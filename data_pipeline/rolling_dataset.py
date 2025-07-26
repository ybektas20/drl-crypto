# data_pipeline/rolling_dataset.py
from pathlib import Path
import polars as pl
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class RollingWindowDataset(Dataset):
    """
    • Outer‑joins all symbols on 'ts'
    • Forward‑fills missing rows per column
    • Drops 'best_bid'/'best_ask' from the channel list
    """
    def __init__(self, root: str, symbols: list[str],
                window: int = 50,
                sampling_freq: str = "1s",
                start: str | None = None,
                end:   str | None = None):
        """
        start / end : optional strings "YYYY-MM" or "YYYY-MM-DD".
                    Only files whose month is within [start, end] are loaded.
        """
        self.syms, self.n = symbols, window
        root_path = Path(root)

        # -------- date filter helpers --------
        start_ts = None
        end_ts = None
        if start:
            start_ts = pd.to_datetime(start)
        if end:
            end_ts   = pd.to_datetime(end) + pd.offsets.MonthEnd(0)

        def file_in_range(fp: Path) -> bool:
            parts = fp.stem.split("-")
            if len(parts) < 4:        # safety check
                return False
            
            # Handle both monthly (YYYY-MM) and daily (YYYY-MM-DD) formats
            if len(parts) >= 5:  # Daily format: symbol-freq-YYYY-MM-DD
                # For BNBUSDT-600s-2025-05-14, parts = ['BNBUSDT', '600s', '2025', '05', '14']
                date_str = "-".join(parts[-3:])  # "2025-05-14"
                file_ts = pd.to_datetime(date_str)
            else:  # Monthly format: symbol-freq-YYYY-MM
                month_str = "-".join(parts[-2:])  # "2025-02"
                file_ts = pd.to_datetime(month_str + "-01")
            
            if start_ts and file_ts < start_ts: 
                return False
            if end_ts and file_ts > end_ts:   
                return False
            return True

        # -------- load each symbol (concat selected months) --------
        dfs = {}
        for sym in symbols:
            files = sorted(
                fp for fp in root_path.glob(f"{sym}-{sampling_freq}-*.parquet")
                if file_in_range(fp)
            )
            if not files:
                raise FileNotFoundError(f"No files for {sym} in range.")
            
            #for file in files:
            #    print(f"Loading {file} for {sym}...")

            df = pl.concat([pl.read_parquet(fp) for fp in files])
            df = df.sort("ts").unique(subset=["ts"], keep="first")
            dfs[sym] = df


        timeline = pl.concat([d.select("ts") for d in dfs.values()]).unique().sort("ts")
        lf = timeline.lazy()
        for sym, df in dfs.items():
            keep_cols = [c for c in df.columns
                        if c != "ts" and c not in {"best_bid", "best_ask"}]
            rename = {c: f"{sym}_{c}" for c in keep_cols}
            lf = lf.join(df.select(["ts"] + keep_cols).rename(rename).lazy(),
                        on="ts", how="left")

        self.df = (lf.collect()
                    .sort("ts")
                    .fill_null(strategy="forward")
                    .drop_nulls())
        
        
        # ─── cyclical calendar features (global) ──────────────────
        two_pi = 2 * np.pi
        self.df = (self.df
            .with_columns([
            # seconds since midnight
            (pl.col("ts").dt.hour() * 3600 +
                pl.col("ts").dt.minute() * 60 +
                pl.col("ts").dt.second()
            ).alias("sec_mid"),
            pl.col("ts").dt.weekday().alias("dow_idx")
            ])
            .with_columns([
            ( (pl.col("sec_mid") / 86_400) * two_pi ).sin().alias("tod_sin"),
            ( (pl.col("sec_mid") / 86_400) * two_pi ).cos().alias("tod_cos"),
            ( (pl.col("dow_idx") / 7)      * two_pi ).sin().alias("dow_sin"),
            ( (pl.col("dow_idx") / 7)      * two_pi ).cos().alias("dow_cos"),
            ])
            .drop(["sec_mid", "dow_idx"])
        )


        # ---------- auto‑discover channels ----------
        global_cols = {"tod_sin", "tod_cos", "dow_sin", "dow_cos"}
        self.global_chs = sorted(global_cols)

        # keep only columns that REALLY have a symbol prefix,
        # *and* are not one of the calendar feature names
        sym_cols = {
            c.split("_", 1)[1]                     # strip prefix
            for c in self.df.columns
            if "_" in c and c.split("_", 1)[0] in self.syms   # true prefix
            and c not in global_cols                        # not calendar
        }
        self.sym_chs = sorted(sym_cols)

        # full list: calendar first, then per‑asset
        self.chs = self.global_chs + self.sym_chs
        self.C, self.m = len(self.chs), len(self.syms)

        # ---------- burn‑in ----------
        self.df = self.df.slice(self.n + 1)

        ## write df to a parquet file to debug
        #out_path =  "dbg_rolling_dataset.parquet"
        #self.df.write_parquet(out_path)

    # ------------- torch dataset -------------
    def __len__(self):
        return len(self.df) - self.n

    # ------------- torch dataset -------------
    # ----------------------------------------------------------------------
    # torch dataset
    # ----------------------------------------------------------------------
    def __getitem__(self, idx: int):
        """
        Returns one sample (X, y).

        • X  : tensor [C, m, n]
            C = len(self.chs)  (global + per‑asset feature channels)
            m = # assets, n = window length
        • y  : price‑ratio vector [m]

        The routine
        • pads/trim the history slice to exactly n rows,
        • expresses 'price_last' channels as *relative* to the penultimate
            price (so the last entry is always 1),
        • broadcasts the four calendar channels across assets,
        • retries automatically when it hits an NaN or an extreme jump.
        """
        while True:                                 # retry until sample is clean
            if idx >= len(self):                    # len() already subtracts self.n
                raise IndexError("Ran out of clean samples")

            slice_df = self.df.slice(idx, self.n + 1)      # n+1 contiguous bars
            X = torch.empty(self.C, self.m, self.n, dtype=torch.float32)

            # ───────────────── build feature cube ──────────────────
            for c_idx, ch in enumerate(self.chs):

                # -------- 1) GLOBAL calendar channels --------------
                if ch in self.global_chs:
                    lst = slice_df[ch][:-1].to_list()       # drop newest bar
                    need = self.n - len(lst)
                    if need > 0:
                        lst = [lst[0]] * need + lst         # forward‑pad first value
                    elif len(lst) > self.n:
                        lst = lst[-self.n:]
                    X[c_idx] = torch.tensor([lst] * self.m) # broadcast
                    continue

                # -------- 2) PER‑ASSET channels --------------------
                for s_idx, sym in enumerate(self.syms):
                    col = slice_df[f"{sym}_{ch}"][:-1]      # first n rows
                    lst = col.to_list()

                    # forward‑pad with first realised value
                    need = self.n - len(lst)
                    if need > 0:
                        pad_val = lst[0] if lst else float("nan")
                        lst = [pad_val] * need + lst
                    elif len(lst) > self.n:
                        lst = lst[-self.n:]

                    # convert close prices to *relative* series
                    if ch == "price_last":
                        ref = slice_df[f"{sym}_{ch}"][-2]
                        lst = [v / ref for v in lst]

                    X[c_idx, s_idx] = torch.tensor(lst, dtype=torch.float32)

            # ───────────────── build y (price ratios) ──────────────
            y = torch.tensor(
                [
                    slice_df[f"{sym}_price_last"][-1] /
                    slice_df[f"{sym}_price_last"][-2]
                    for sym in self.syms
                ],
                dtype=torch.float32,
            )

            # ───────────────── sanity check ────────────────────────
            if (
                torch.isnan(y).any()
                or (y <= 0.8).any()
                or (y >= 1.2).any()
            ):
                idx += 1                      # skip pathological bar
                continue

            return X, y
