import polars as pl
import numpy as np
from omegaconf import OmegaConf
import hydra
from pathlib import Path
from polars import col
from tqdm import tqdm

# ───────────────────────────────────────────────────────────────────────────────
# Feature Calculation Functions

def get_ofi(df, buy_col="buy_qty", sell_col="sell_qty", last_price_col="price_last", n=60):
    """Calculate the Order Flow Imbalance (OFI) using Polars."""
    result = df.with_columns([
        # Calculate buy and sell flows
        (pl.col(buy_col) * pl.col(last_price_col)).rolling_sum(window_size=n, min_samples=1).alias("buy_flow"),
        (pl.col(sell_col) * pl.col(last_price_col)).rolling_sum(window_size=n, min_samples=1).alias("sell_flow")
    ]).with_columns([
        # Calculate OFI
        ((pl.col("buy_flow") - pl.col("sell_flow")) / (pl.col("buy_flow") + pl.col("sell_flow"))).alias("ofi")
    ]).select("ofi")
    
    # Handle both DataFrame and LazyFrame
    if hasattr(result, 'to_series'):
        return result.to_series()
    else:
        return result.collect().to_series()

def get_log_ret(df, price_col="price_last", n=60):
    """Calculate log returns using Polars."""
    result = df.with_columns([
        (pl.col(price_col) / pl.col(price_col).shift(n)).log().alias("log_ret")
    ]).select("log_ret")
    
    # Handle both DataFrame and LazyFrame
    if hasattr(result, 'to_series'):
        return result.to_series()
    else:
        return result.collect().to_series()

def get_log_volume(df, buy_col="buy_qty", sell_col="sell_qty", last_price_col="price_last", n=60):
    """Calculate log-transformed volume using Polars."""
    result = df.with_columns([
        # Calculate total volume
        (pl.col(buy_col) * pl.col(last_price_col) + pl.col(sell_col) * pl.col(last_price_col)).alias("volume")
    ]).with_columns([
        # Replace zeros with 1, take log, then rolling sum
        pl.col("volume").map_elements(lambda x: max(x, 1), return_dtype=pl.Float64).log()
        .rolling_sum(window_size=n, min_samples=1).alias("log_volume")
    ]).select("log_volume")
    
    # Handle both DataFrame and LazyFrame
    if hasattr(result, 'to_series'):
        return result.to_series()
    else:
        return result.collect().to_series()


# Alternative version that works with a Polars Series directly
def normalize_zscore(series, window=60):
    """Normalize a Polars Series using z-score normalization."""
    mean = series.rolling_mean(window_size=window, min_samples=window)
    std = series.rolling_std(window_size=window, min_samples=window)
    return (series - mean) / std

def create_features(df, cfg):
    """Create features from a Polars DataFrame based on configuration."""
    features = []

    # Handle log returns
    if "return" in cfg:
        for window in cfg["return"]["windows"]:
            log_ret = get_log_ret(df, price_col=cfg["return"]["input_col"], n=window)
            name = f"log_ret_{window}"
            if cfg["return"].get("norm", {}).get("type") == "rolling_zscore":
                norm_window = int(cfg["return"]["norm"]["window"]) * window
                log_ret = normalize_zscore(log_ret, window=norm_window)
                name += f"_zscore_{norm_window}"
            features.append((name, log_ret))

    # Handle volume features
    if "volume" in cfg:
        for agg_window in cfg["volume"]["windows"]:
            log_volume = get_log_volume(df, buy_col=cfg["volume"]["input_cols"][0], 
                                        sell_col=cfg["volume"]["input_cols"][1], 
                                        last_price_col=cfg["volume"].get("price_col", "price_last"),
                                        n=agg_window)
            name = f"log_volume_{agg_window}"
            if cfg["volume"].get("norm", {}).get("type") == "zscore":
                norm_window = int(cfg["volume"]["norm"]["window"]) * agg_window
                log_volume = normalize_zscore(log_volume, window=norm_window)
                name += f"_zscore_{norm_window}"
            features.append((name, log_volume))

    # Handle OFI features
    if "ofi" in cfg:
        for agg_window in cfg["ofi"]["windows"]:
            ofi = get_ofi(df, buy_col=cfg["ofi"]["buy_col"], 
                          sell_col=cfg["ofi"]["sell_col"],
                          last_price_col=cfg["ofi"].get("price_col", "price_last"),
                          n=agg_window)
            name = f"ofi_{agg_window}"
            if cfg["ofi"].get("norm", {}).get("type") == "zscore":
                norm_window = int(cfg["ofi"]["norm"]["window"]) * agg_window
                ofi = normalize_zscore(ofi, window=norm_window)
                name += f"_zscore_{norm_window}"
            features.append((name, ofi))

    return features


def inject_family(df, cfg):
    """
    This function takes a LazyFrame, applies the feature generation, 
    and returns a transformed LazyFrame with calculated features.
    """
    df = df.lazy()

    # Add features to the dataframe based on the configuration
    for feature_name, feature_data in create_features(df, cfg):
        df = df.with_columns(pl.lit(feature_data).alias(feature_name))
    
    return df


# ───────────────────────────────────────────────────────────────────────────────
@hydra.main(config_path="../configs", config_name="data", version_base=None)
def run(cfg):
    in_freq   = cfg.resample.freq            # "1s", "600s", etc.
    in_root   = Path(cfg.resample.dest.format(freq=in_freq))
    out_root  = Path(cfg.features.dest) / in_freq
    
    # Clean output directory to avoid mixing old and new feature files
    if out_root.exists():
        print(f"Cleaning existing output directory: {out_root}")
        import shutil
        shutil.rmtree(out_root)
    
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- group input files by symbol -----------------------------------------
    files_by_sym = {}
    # Handle nested directory structure: data/processed/freq/symbol/symbol-freq-month.parquet
    for symbol_dir in in_root.iterdir():
        if symbol_dir.is_dir():
            sym = symbol_dir.name  # e.g., "BTCUSDT"
            for fp in symbol_dir.glob("*.parquet"):
                files_by_sym.setdefault(sym, []).append(fp)
    
    # Fallback: handle flat structure for backward compatibility
    if not files_by_sym:
        for fp in in_root.glob("**/*.parquet"):
            sym = fp.stem.split("-")[0]          # "BTCUSDT" from "BTCUSDT‑1s‑2025‑05"
            files_by_sym.setdefault(sym, []).append(fp)

    print("creating features...")
    for sym, filelist in tqdm(files_by_sym.items(), desc="Symbols"):
        print(f"\nProcessing {sym} with {len(filelist)} files")
        #for fp in sorted(filelist):
            #print(f"  - {fp}")
        
        # 1) concat, sort -------------------------------------------------------
        dfs_to_concat = []
        for fp in sorted(filelist):
            df_temp = pl.read_parquet(fp).lazy()
            print(f"  File {fp.name}: {df_temp.collect().shape} rows")
            dfs_to_concat.append(df_temp)
        
        if not dfs_to_concat:
            print(f"  No files for {sym}, skipping...")
            continue
            
        lf = pl.concat(dfs_to_concat).sort("ts")
        total_rows = lf.collect().shape[0]
        print(f"  Total concatenated rows: {total_rows}")
        
        if total_rows == 0:
            print(f"  No data for {sym}, skipping...")
            continue

        # 2) resample -----------------------------------------------------------
        # ensure ts is datetime
        lf = lf.with_columns(pl.col("ts").cast(pl.Datetime("ms")))
        # dynamic groupby keeps lazy execution
        # Choose appropriate aggregations for your raw columns:
        # after we have a single, long LazyFrame = lf_resampled
        # ------------------------------------------------------
        lf_resampled = (lf                  # <- still the concat of all months
                        .with_columns(pl.col("ts").cast(pl.Datetime("ms")))
                        .group_by_dynamic("ts", every=in_freq, closed="left")
                        .agg([
                            pl.col("price_last").last().alias("price_last"),
                            pl.col("buy_qty").sum().alias("buy_qty"),
                            pl.col("sell_qty").sum().alias("sell_qty"),
                            pl.col("best_bid").last().alias("best_bid"),
                            pl.col("best_ask").last().alias("best_ask"),
                        ]))


        # ---- feature engineering on the FULL dataset ----
        print(f"  Calculating features on full dataset...")
        lf_with_features = inject_family(lf_resampled, cfg.features)
        full_features = lf_with_features.collect()
        print(f"  Full dataset after features: {full_features.shape[0]} rows")

        # ---- drop NaNs and add date column ----
        full_features_clean = full_features.drop_nulls().drop_nans()
        print(f"  After dropping nulls/nans: {full_features_clean.shape[0]} rows")
        
        if full_features_clean.shape[0] == 0:
            print(f"  No data left after cleaning for {sym}, skipping...")
            continue
            
        # Add date column for splitting
        full_features_clean = full_features_clean.with_columns(
            pl.col("ts").dt.date().alias("date")
        )

        # ---------- split by DAY and write one parquet per day ----------
        dates = full_features_clean.select("date").unique()["date"].to_list()
        print(f"  Splitting into {len(dates)} daily files...")
        total_rows = 0
        for d in dates:
            day_df = full_features_clean.filter(pl.col("date") == d).drop("date")
            
            if day_df.shape[0] > 0:
                out_path = out_root / f"{sym}-{in_freq}-{d}.parquet"
                day_df.write_parquet(out_path, compression="zstd")
                #print(f"    Wrote {out_path} with {day_df.shape[0]} rows")
                total_rows += day_df.shape[0]
            else:
                print(f"    Skipped {d} - no data")
        print(f"  Total rows written for {sym}: {total_rows}")


if __name__ == "__main__":
    run()
