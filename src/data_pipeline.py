# # # data pipeline


# src/data_pipeline.py
import os
import polars as pl
from datasets import load_dataset

# -----------------------------
# Config
# -----------------------------
DATASET_NAME = "irlspbru/RFSD"
CACHE_DIR    = "data/hf_cache"
OUTPUT_PATH  = "data/raw/rfsd_2011_2024.parquet"
START_YEAR   = 2011
END_YEAR     = 2024

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# -----------------------------
# RFSD raw line - concept mapping
# The dataset stores financial statement lines as "line_XXXX".
# We derive our required columns from these using standard Russian
# accounting form (OKUD 0710001/0710002) definitions.
# -----------------------------

# Year is encoded in creation_date or must be derived from the filing
# The dataset has no explicit "year" column — we extract it from creation_date.

# Derived column formulas (all line_XXXX are from the balance sheet /
# income statement / cash flow statement):
#
#   total_assets        = line_1600   (Balance sheet total assets)
#   total_liabilities   = line_1400 + line_1500  (long-term + short-term liabilities)
#   current_assets      = line_1200
#   current_liabilities = line_1500
#   cash                = line_1250
#   inventory           = line_1210
#   receivables         = line_1230
#   operating_cash_flow = line_4100  (net cash from operating activities)
#   free_cash_flow      = line_4100 + line_4200  (operating + investing CF)
#   revenue             = line_2110
#   net_income          = line_2400
#
# Liquidity ratios:
#   current_ratio  = line_1200 / line_1500
#   quick_ratio    = (line_1200 - line_1210) / line_1500
#   cash_ratio     = line_1250 / line_1500
#   working_capital = line_1200 - line_1500
#
# Industry is okved_section (1-letter OKVED2 section code)


def prepare_rfsd_subset(cache_dir=CACHE_DIR,
                         output_path=OUTPUT_PATH,
                         start_year=START_YEAR,
                         end_year=END_YEAR):
    """
    Load RFSD from HuggingFace, derive required columns from raw line codes,
    filter by year, and save to Parquet.
    """
    if os.path.exists(output_path):
        print(f"Loading cached subset from {output_path}")
        return pl.read_parquet(output_path)

    print("⏳ Downloading RFSD dataset from HuggingFace...")
    dataset = load_dataset(DATASET_NAME, split="train",
                           cache_dir=cache_dir, num_proc=1)
    df = pl.from_pandas(dataset.to_pandas())
    print(f"   Raw shape: {df.shape}")
    print(f"   Columns sample: {df.columns[:10]}")

    # ------------------------------------------------------------------
    # 1. Extract year from creation_date
    #    creation_date appears to be a string like "2015-01-01" or int year
    # ------------------------------------------------------------------
    if "creation_date" in df.columns:
        if df["creation_date"].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col("creation_date").str.slice(0, 4).cast(pl.Int32).alias("year")
            )
        elif df["creation_date"].dtype in (pl.Int32, pl.Int64, pl.Float64):
            df = df.with_columns(
                pl.col("creation_date").cast(pl.Int32).alias("year")
            )
        else:
            # Try casting whatever it is
            df = df.with_columns(
                pl.col("creation_date").cast(pl.Utf8).str.slice(0, 4).cast(pl.Int32).alias("year")
            )
    else:
        raise ValueError(
            "Cannot find 'creation_date' to derive year. "
            f"Available columns: {df.columns}"
        )

    # ------------------------------------------------------------------
    # 2. Industry from okved_section
    # ------------------------------------------------------------------
    if "okved_section" in df.columns:
        df = df.rename({"okved_section": "industry"})
    elif "okved" in df.columns:
        # Fall back: use first 2 chars of OKVED code as industry proxy
        df = df.with_columns(
            pl.col("okved").cast(pl.Utf8).str.slice(0, 2).alias("industry")
        )
    else:
        raise ValueError("Cannot find industry column (okved_section or okved).")

    # ------------------------------------------------------------------
    # 3. Derive financial variables from line codes
    #    Guard each with a null-safe coalesce in case lines are missing
    # ------------------------------------------------------------------
    def col_or_zero(name):
        """Return column if it exists, else a zero literal."""
        return pl.col(name) if name in df.columns else pl.lit(0.0)

    df = df.with_columns([
        # Core balance sheet
        col_or_zero("line_1600").alias("total_assets"),
        (col_or_zero("line_1400") + col_or_zero("line_1500")).alias("total_liabilities"),
        col_or_zero("line_1200").alias("current_assets"),
        col_or_zero("line_1500").alias("current_liabilities"),
        col_or_zero("line_1250").alias("cash"),
        col_or_zero("line_1210").alias("inventory"),
        col_or_zero("line_1230").alias("receivables"),

        # Cash flow
        col_or_zero("line_4100").alias("operating_cash_flow"),
        (col_or_zero("line_4100") + col_or_zero("line_4200")).alias("free_cash_flow"),

        # Income statement
        col_or_zero("line_2110").alias("revenue"),
        col_or_zero("line_2400").alias("net_income"),
    ])

    # Liquidity ratios — avoid division by zero
    df = df.with_columns([
        (pl.col("current_assets") / pl.when(pl.col("current_liabilities") != 0)
         .then(pl.col("current_liabilities")).otherwise(pl.lit(None))
         ).alias("current_ratio"),

        ((pl.col("current_assets") - pl.col("inventory")) /
         pl.when(pl.col("current_liabilities") != 0)
         .then(pl.col("current_liabilities")).otherwise(pl.lit(None))
         ).alias("quick_ratio"),

        (pl.col("cash") / pl.when(pl.col("current_liabilities") != 0)
         .then(pl.col("current_liabilities")).otherwise(pl.lit(None))
         ).alias("cash_ratio"),

        (pl.col("current_assets") - pl.col("current_liabilities")).alias("working_capital"),
    ])

    # ------------------------------------------------------------------
    # 4. Filter by year
    # ------------------------------------------------------------------
    df = df.filter(
        (pl.col("year") >= start_year) & (pl.col("year") <= end_year)
    )
    print(f"   After year filter ({start_year}–{end_year}): {df.shape}")

    if df.is_empty():
        raise ValueError(
            f"No rows remain after year filter. "
            f"Check that creation_date encodes the filing year. "
            f"Year range in data: {df['year'].min()}–{df['year'].max()}"
        )

    # ------------------------------------------------------------------
    # 5. Keep only needed columns
    # ------------------------------------------------------------------
    keep = [
        "year", "industry",
        "free_cash_flow", "operating_cash_flow",
        "current_ratio", "quick_ratio", "cash_ratio",
        "working_capital", "total_assets", "total_liabilities",
        "revenue", "net_income", "cash", "inventory", "receivables",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df.select(keep)

    # Drop rows where ALL financial columns are null/zero
    fin_cols = [c for c in keep if c not in ("year","industry")]
    df = df.filter(
        pl.any_horizontal([pl.col(c).is_not_null() & (pl.col(c) != 0) for c in fin_cols])
    )
    print(f"   After null/zero filter: {df.shape}")

    # ------------------------------------------------------------------
    # 6. Aggregate to industry-year level
    #    (RFSD is firm-level; we need panel aggregates for time series)
    # ------------------------------------------------------------------
    df_agg = df.group_by(["year", "industry"]).agg([
        pl.col("free_cash_flow").median().alias("free_cash_flow"),
        pl.col("current_ratio").median().alias("current_ratio"),
        pl.col("quick_ratio").median().alias("quick_ratio"),
        pl.col("cash_ratio").median().alias("cash_ratio"),
        pl.col("working_capital").median().alias("working_capital"),
        pl.col("total_assets").median().alias("total_assets"),
        pl.col("total_liabilities").median().alias("total_liabilities"),
        pl.col("revenue").median().alias("revenue"),
        pl.col("net_income").median().alias("net_income"),
        pl.len().alias("n_firms"),
    ]).sort(["industry", "year"])

    print(f"   Aggregated (industry-year) shape: {df_agg.shape}")
    print(f"   Industries: {df_agg['industry'].n_unique()}")
    print(f"   Years: {sorted(df_agg['year'].unique().to_list())}")

    df_agg.write_parquet(output_path)
    print(f"   Saved to {output_path}")
    return df_agg


def load_rfsd_2011_2024(cache_dir=CACHE_DIR,
                         output_path=OUTPUT_PATH,
                         start_year=START_YEAR,
                         end_year=END_YEAR):
    """
    Full pipeline entry point. Returns aggregated industry-year Polars DataFrame.
    """
    return prepare_rfsd_subset(
        cache_dir=cache_dir,
        output_path=output_path,
        start_year=start_year,
        end_year=end_year,
    )







