#!/usr/bin/env python
"""
Generic resampler: Binance aggTrades ZIP → Parquet bars.

• Frequency (1 s, 5 s, …) is read from Hydra config:  data.resample.freq
• Output path auto-includes the freq so different granularities can coexist
"""

from __future__ import annotations
import io, zipfile, os, glob
from pathlib import Path
import polars as pl

import hydra
from omegaconf import DictConfig, OmegaConf
import polars as pl
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# 8-column schema:   0 id | 1 price | 2 qty | 3 firstId | 4 lastId | 5 time
#                    6 isBuyerMaker | 7 isBestMatch
SCHEMA = {
    "0": pl.Int64,
    "1": pl.Float64,
    "2": pl.Float64,
    "3": pl.Int64,
    "4": pl.Int64,
    "5": pl.Int64,      # ms or µs
    "6": pl.Boolean,
    "7": pl.Boolean,
}
KEEP_PRICE_QTY_FLAG = ["1", "2", "6"]          # keep only these after parsing

# --------------------------------------------------------------------------- #
def read_zip(path: Path) -> pl.LazyFrame:
    """Return a LazyFrame with cols: ts, price, qty, is_buyer_maker."""
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(zf.namelist()[0])

    df = pl.read_csv(
        io.BytesIO(raw),
        has_header=False,
        new_columns=list(SCHEMA.keys()),
        schema_overrides=SCHEMA,
        ignore_errors=True,
    )

    # µs threshold: any timestamp > 1e15 → divide by 1_000
    if df["5"].max() > 1_000_000_000_000_000:
        df = df.with_columns((pl.col("5") // 1_000).alias("5"))

    return (
        df.lazy()
          .with_columns(pl.from_epoch("5", time_unit="ms").alias("ts"))
          .select(["ts", *KEEP_PRICE_QTY_FLAG])
          .rename({"1": "price", "2": "qty", "6": "is_buyer_maker"})
          .sort("ts")
    )



def resample(lf: pl.LazyFrame, freq: str, drop_empty: bool) -> pl.DataFrame:
    # ------------------------------------------------------------------ #
    # 1) bar + synthetic BBO
    # ------------------------------------------------------------------ #
    bars_lz = (
        lf.group_by_dynamic("ts", every=freq, closed="left", label="left")
          .agg(
              price_last = pl.col("price").last(),
              buy_qty    = (
                  pl.when(~pl.col("is_buyer_maker"))
                    .then(pl.col("qty")).otherwise(0.0)
              ).sum(),
              sell_qty   = (
                  pl.when(pl.col("is_buyer_maker"))
                    .then(pl.col("qty")).otherwise(0.0)
              ).sum(),
              best_bid = (
                  pl.when(pl.col("is_buyer_maker"))
                    .then(pl.col("price")).otherwise(None)
              ).last(),
              best_ask = (
                  pl.when(~pl.col("is_buyer_maker"))
                    .then(pl.col("price")).otherwise(None)
              ).last(),
          )
          .sort("ts")
    )
    if drop_empty:
        bars_lz = bars_lz.drop_nulls("price_last")

    bars = (
        bars_lz.collect(engine="streaming")
               .with_columns(
                   pl.col("best_bid").forward_fill(),
                   pl.col("best_ask").forward_fill(),
                   pl.col("ts").cast(pl.Datetime("ms")),
               )
    )

    # ------------------------------------------------------------------ #
    # 2) empirical tick  =  most-frequent positive (ask − bid)
    # ------------------------------------------------------------------ #
    tick_df = (
        bars.select(spread = pl.col("best_ask") - pl.col("best_bid"))
            .filter(pl.col("spread") > 0)
            .group_by("spread")
            .len()
            .sort("len", descending=True)
    )
    tick = tick_df["spread"][0] if len(tick_df) else 0.0
    if tick == 0.0:                       # ultra-quiet session fallback
        tick = bars["price_last"].mean() * 1e-8

    # ------------------------------------------------------------------ #
    # 3) enforce strictly-positive spread = one tick
    # ------------------------------------------------------------------ #
    bars = bars.with_columns(
        pl.when(pl.col("best_ask") <= pl.col("best_bid"))
          .then(pl.col("best_bid") + tick)
          .otherwise(pl.col("best_ask"))
          .alias("best_ask")
    )

    return bars



def process_one(zip_path: Path, processed_root: Path, cfg):
    sym   = zip_path.parent.name
    month = "-".join(zip_path.stem.split("-")[-2:])          # e.g. 2025-05
    out_dir = processed_root / sym
    outfile = out_dir / f"{sym}-{cfg.resample.freq}-{month}.parquet"
    if outfile.exists():    
        #print(f"Skipping {outfile} (already exists)")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = resample(read_zip(zip_path), cfg.resample.freq, cfg.resample.drop_empty)    
    bars.write_parquet(outfile, compression="zstd")


# --------------------------------------------------------------------------- #
@hydra.main(config_path="../configs", config_name="data", version_base=None)
def run(cfg: DictConfig):
    #print(OmegaConf.to_yaml(cfg, resolve=True))
    print("Resampling Binance aggTrades ZIPs to Parquet bars...")
    raw_processed_root = Path(cfg.download.dest)
    processed_root = Path(cfg.resample.dest.format(freq=cfg.resample.freq))

    files = glob.glob(os.path.join(raw_processed_root, "**/*.zip"), recursive=True)
    for fp in tqdm(files, desc=f"Resampling → {cfg.resample.freq}", unit="file"):
        process_one(Path(fp), processed_root, cfg)


if __name__ == "__main__":
    run()