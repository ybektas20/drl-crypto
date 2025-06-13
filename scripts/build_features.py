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
# Hydra config for the paths and feature settings
@hydra.main(config_path="../configs", config_name="data", version_base=None)
def main(cfg):
    print("creating features...")
    processed_root = Path(cfg.resample.dest.format(freq=cfg.resample.freq))
    features_root = Path(cfg.features.dest) / cfg.resample.freq

    files = [str(file) for file in processed_root.glob("**/*.parquet")]

    for file in tqdm(files, desc="Processing files"):
        df = pl.read_parquet(file).lazy()
        df = df.sort("ts")

        df_with_features = inject_family(df, cfg.features)

        features_root.mkdir(parents=True, exist_ok=True)

        output_file = features_root / Path(file).name
        df_with_features.collect().write_parquet(output_file, compression="zstd")


if __name__ == "__main__":
    main()
