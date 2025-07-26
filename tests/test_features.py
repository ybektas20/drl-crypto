import pytest
import polars as pl
import numpy as np
from omegaconf import OmegaConf
from data_pipeline.build_features import inject_family


# Helper functions
def get_ofi(df, n=60):
    buy_flow = (df['buy_qty'] * df['last_price']).rolling(n).sum()
    sell_flow = (df['sell_qty'] * df['last_price']).rolling(n).sum()
    ofi = (buy_flow - sell_flow) / (buy_flow + sell_flow)
    return ofi

def get_log_ret(df, n=60):
    log_ret = np.log(df['price_last'] / df['price_last'].shift(n))
    return log_ret

def get_log_volume(df, n=60):
    volume = df['buy_qty'] * df['last_price'] + df['sell_qty'] * df['last_price']
    log_volume = np.log(volume.replace([0], 1)).rolling(n).sum()
    return log_volume


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# 1. Test `returns` feature (lag1, log transformation)
def test_returns_lag1():
    df = pl.DataFrame(
        {
            "ts": [0, 1, 2, 3],
            "price_last": [10.0, 11.0, 12.0, 13.0],
            "buy_qty": [0, 0, 0, 0],
            "sell_qty": [0, 0, 0, 0],
        }
    ).lazy()

    cfg = OmegaConf.create(
        {"input_col": "price_last", "lags": [1], "agg_windows": [1], "norm": {"type": "none"}}
    )
    out = inject_family(df, cfg).collect()

    # log returns
    expected = np.array([np.nan, np.log(11 / 10), np.log(12 / 11), np.log(13 / 12)])
    result = out["price_last_agg1_l1"].to_numpy()

    np.testing.assert_allclose(result, expected, equal_nan=True)


# 2. Test `buy_qty` z-score normalization with window 2
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
        {"input_col": "buy_qty", "lags": [0], "agg_windows": [2], "norm": {"type": "zscore", "window": 2}}
    )
    out = inject_family(df, cfg).collect()
    
    # manually calculating z-score with window 2
    expected = np.array([np.nan, np.nan, 0.70710678])  # Z-score for value 3 with window of 2
    result = out["buy_qty_agg2_z_l0"].to_numpy()

    np.testing.assert_allclose(result, expected, equal_nan=True, atol=1e-6)


# 3. Test `OFI` (Order Flow Imbalance) with min-max normalization (window 2)
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
        {"buy_col": "buy_qty", "sell_col": "sell_qty", "agg_windows": [2], "norm": {"type": "minmax", "window": 2}}
    )
    out = inject_family(df, cfg).collect()

    # raw OFI values (None, -0.142857, 0.428571)
    # min-max over last 2 raw values -> [nan, nan, 1]
    expected = np.array([np.nan, np.nan, 1.0])
    result = out["ofi_2_l0"].to_numpy()

    np.testing.assert_allclose(result, expected, equal_nan=True, atol=1e-6)


# 4. Test `log_volume`
def test_log_volume():
    df = pl.DataFrame(
        {
            "ts": [0, 1, 2, 3],
            "price_last": [10.0, 11.0, 12.0, 13.0],
            "buy_qty": [1.0, 2.0, 3.0, 4.0],
            "sell_qty": [2.0, 1.0, 1.0, 2.0],
        }
    ).lazy()

    cfg = OmegaConf.create(
        {"input_cols": ["buy_qty", "sell_qty"], "agg_windows": [2], "norm": {"type": "none"}}
    )
    out = inject_family(df, cfg).collect()

    # manually calculating log volume (buy + sell volume)
    volume = df["buy_qty"] * df["price_last"] + df["sell_qty"] * df["price_last"]
    expected = np.log(volume).rolling(2).sum().to_numpy()  # log(volume) and sum over window size of 2
    result = out["log_volume_agg2_l0"].to_numpy()

    np.testing.assert_allclose(result, expected, equal_nan=True)


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main()
