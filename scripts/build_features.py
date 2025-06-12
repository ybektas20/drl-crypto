#!/usr/bin/env python
"""
Feature builder
===============

Reads resampled bars in  data/interim/{freq}/{sym}/{sym}-{freq}-{YYYY-MM}.parquet
and writes engineered features to
    data/processed/{freq}/{sym}/{sym}-feat-{freq}-{YYYY-MM}.parquet

Everything – lags, rolling windows, norms – comes from configs/features.yaml.
"""

from __future__ import annotations
from pathlib import Path
import glob

import polars as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


# ───────────────────────────── helper exprs ──────────────────────────────
def rolling_norm(expr: pl.Expr, ntype: str, win: int) -> pl.Expr:
    if ntype == "zscore":
        expr = (expr - expr.rolling_mean(win)) / (expr.rolling_std(win) + 1e-9)
    elif ntype == "minmax":
        expr = (expr - expr.rolling_min(win)) / (
            expr.rolling_max(win) - expr.rolling_min(win) + 1e-9
        )
    return expr


def make_lagged(col: str, lag: int, transform: str | None) -> pl.Expr:
    if transform == "log":
        expr = (pl.col(col) / pl.col(col).shift(lag)).log()
        return expr.alias(f"{col}_logret_l{lag}")
    if transform == "pct":
        expr = (pl.col(col) / pl.col(col).shift(lag) - 1.0)
        return expr.alias(f"{col}_pct_l{lag}")
    return pl.col(col).shift(lag).alias(f"{col}_l{lag}")


# ────────────────────────── feature family handlers ──────────────────────
def single_series_family(lf: pl.LazyFrame, spec: DictConfig) -> pl.LazyFrame:
    col = spec.input_col
    for win in spec.agg_windows:
        base_alias = f"{col}_agg{win}"
        lf = lf.with_columns(
            pl.col(col).rolling_sum(win, center=False).alias(base_alias)
        )
        # normalise
        normed = rolling_norm(
            pl.col(base_alias), spec.norm.type, spec.norm.get("window", 1)
        ).alias(base_alias)  # keeps same name
        lf = lf.with_columns(normed)
        # lags
        for lag in spec.lags:
            lf = lf.with_columns(make_lagged(base_alias, lag, spec.get("transform")))
    return lf


def ofi_family(lf: pl.LazyFrame, spec: DictConfig) -> pl.LazyFrame:
    for win in spec.agg_windows:
        bf = f"buy_flow_{win}"
        sf = f"sell_flow_{win}"
        ofi = f"ofi_{win}"

        lf = lf.with_columns(
            [
                pl.col(spec.buy_col).rolling_sum(win, center=False).alias(bf),
                pl.col(spec.sell_col).rolling_sum(win, center=False).alias(sf),
            ]
        )
        ofi_expr = (
            (pl.col(bf) - pl.col(sf)) / (pl.col(bf) + pl.col(sf) + 1e-9)
        ).alias(ofi)
        lf = lf.with_columns(ofi_expr)

        # normalise and (only) lag-0
        normed = rolling_norm(
            pl.col(ofi), spec.norm.type, spec.norm.get("window", 1)
        ).alias(f"{ofi}_l0")
        lf = lf.with_columns(normed)
    return lf

# ───────────────────────── exposed helper for unit-tests ───────────────────
def inject_family(lf: pl.LazyFrame, spec: DictConfig) -> pl.LazyFrame:
    """Public wrapper so tests can add one feature family to a LazyFrame."""
    if "input_col" in spec:
        return single_series_family(lf, spec)
    if "buy_col" in spec:          # OFI
        return ofi_family(lf, spec)
    raise ValueError("Unknown feature spec")


# ───────────────────────────── Hydra entrypoint ───────────────────────────
@hydra.main(config_path="configs", config_name="features", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    freq = cfg.get("resample", {}).get("freq", "1s")
    interim_root = Path(f"data/interim/{freq}")
    out_root = Path(f"data/processed/{freq}")
    out_root.mkdir(parents=True, exist_ok=True)

    dtype = getattr(pl, cfg.dtype.capitalize())  # "float32" → pl.Float32

    shards = sorted(glob.glob(str(interim_root / "*" / f"*{freq}-*.parquet")))
    if not shards:
        raise SystemExit(f"No input shards under {interim_root}")

    for fp in tqdm(shards, desc=f"Features {freq}", unit="file"):
        sym = Path(fp).parent.name
        month = "-".join(Path(fp).stem.split("-")[-2:])
        out_file = out_root / sym / f"{sym}-feat-{freq}-{month}.parquet"
        if out_file.exists():
            continue

        lf = pl.scan_parquet(fp)

        # apply every feature family
        for fam_name, spec in cfg.features.items():
            if "input_col" in spec:      # returns / volumes
                lf = single_series_family(lf, spec)
            elif "buy_col" in spec:      # OFI
                lf = ofi_family(lf, spec)
            else:
                raise ValueError(f"Unknown feature block {fam_name}")

        # collect → cast dtype → drop raw cols → write
        df = lf.collect(engine="streaming")

         # 1) cast *feature* columns to float32 but leave ts untouched
        feat_cols = [c for c in df.columns if c != "ts"]
        df = df.with_columns([pl.col(feat_cols).cast(dtype)])
    
         # 2) drop raw bar columns so only engineered features remain
        drop = {"price_last", "buy_qty", "sell_qty", "price_last_agg1"}
        df = df.select([c for c in df.columns if c not in drop])
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_file, compression="zstd")
        tqdm.write(f"✔ {out_file}")


if __name__ == "__main__":
    main()
