#!/usr/bin/env python
"""
Generic resampler: Binance aggTrades ZIP → Parquet bars.

• Frequency (1 s, 5 s, …) is read from Hydra config:  data.resample.freq
• Output path auto-includes the freq so different granularities can coexist
"""

from __future__ import annotations
import io, zipfile, os, glob
from pathlib import Path

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
    bars = (
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
          )
          .sort("ts")
    )
    if drop_empty:
        bars = bars.drop_nulls("price_last")

    return (
        bars.collect(streaming=True)
            .with_columns(pl.col("ts").cast(pl.Datetime("ms")))
    )


def process_one(zip_path: Path, out_root: Path, cfg):
    sym   = zip_path.parent.name
    month = "-".join(zip_path.stem.split("-")[-2:])          # e.g. 2025-05
    out_dir = out_root / sym
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = resample(read_zip(zip_path), cfg.resample.freq, cfg.resample.drop_empty)
    outfile = out_dir / f"{sym}-{cfg.resample.freq}-{month}.parquet"
    bars.write_parquet(outfile, compression="zstd")


# --------------------------------------------------------------------------- #
@hydra.main(config_path="configs", config_name="data", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    raw_root = Path(cfg.paths.raw_root)
    out_root = Path(cfg.paths.out_root.format(freq=cfg.resample.freq))

    files = glob.glob(os.path.join(raw_root, "**/*.zip"), recursive=True)
    for fp in tqdm(files, desc=f"Resampling → {cfg.resample.freq}", unit="file"):
        process_one(Path(fp), out_root, cfg)


if __name__ == "__main__":
    main()
