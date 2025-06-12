import numpy as np
import polars as pl
from omegaconf import OmegaConf

from scripts.build_features import inject_family


# ───────────────────────── 1 · log-returns (lag-1) ──────────────────────────
def test_returns_log_lag1():
    df = pl.DataFrame(
        {
            "ts": [0, 1, 2, 3],
            "price_last": [10.0, 11.0, 12.0, 13.0],
            "buy_qty": [0, 0, 0, 0],
            "sell_qty": [0, 0, 0, 0],
        }
    ).lazy()

    cfg = OmegaConf.create(
        {
            "input_col": "price_last",
            "transform": "log",
            "lags": [1],
            "agg_windows": [1],
            "norm": {"type": "none"},
        }
    )

    out = inject_family(df, cfg).collect()["price_last_agg1_logret_l1"].to_numpy()

    expected = np.array(
        [
            np.nan,
            np.log(11.0 / 10.0),  # ≈0.09531018
            np.log(12.0 / 11.0),  # ≈0.08701138
            np.log(13.0 / 12.0),  # ≈0.08004271
        ],
        dtype=float,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-8, equal_nan=True)


# ─────────── 2 · buy_volume (sum2) + 2-point rolling z-score ───────────────
def test_buy_volume_zscore():
    df = pl.DataFrame(
        {
            "ts": [0, 1, 2],
            "price_last": [0, 0, 0],
            "buy_qty": [1.0, 2.0, 3.0],
            "sell_qty": [0, 0, 0],
        }
    ).lazy()

    cfg = OmegaConf.create(
        {
            "input_col": "buy_qty",
            "lags": [0],
            "agg_windows": [2],
            "norm": {"type": "zscore", "window": 2},
        }
    )

    col = inject_family(df, cfg).collect()["buy_qty_agg2_l0"].to_numpy()
    expected = np.array([np.nan, np.nan, 0.70710678])  # hard-coded
    np.testing.assert_allclose(col, expected, atol=1e-6, equal_nan=True)


# ──────── 3 · OFI (window2) + 2-point rolling min-max normalise ────────────
def test_ofi_minmax():
    df = pl.DataFrame(
        {
            "ts": [0, 1, 2],
            "price_last": [0, 0, 0],
            "buy_qty": [1.0, 2.0, 3.0],
            "sell_qty": [3.0, 1.0, 1.0],
        }
    ).lazy()

    cfg = OmegaConf.create(
        {
            "buy_col": "buy_qty",
            "sell_col": "sell_qty",
            "agg_windows": [2],
            "norm": {"type": "minmax", "window": 2},
        }
    )

    out = inject_family(df, cfg).collect()["ofi_2_l0"].to_numpy()
    expected = np.array([np.nan, np.nan, 1.0])  # raw OFI → min-max over last 2
    np.testing.assert_allclose(out, expected, equal_nan=True, atol=1e-6)
