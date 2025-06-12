#!/usr/bin/env python
"""
Asynchronously mirror Binance monthly aggTrades ZIPs.

Why async?
----------
`https://data.binance.vision` has ~200-400 ms latency per object.  Downloading
dozens of files concurrently hides that round-trip time and keeps the CPU busy.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import date
from itertools import product
from pathlib import Path
import aiohttp
import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm
import hydra
from omegaconf import OmegaConf


BASE = "https://data.binance.vision/data/spot/monthly/aggTrades"


# --------------------------------------------------------------------------- #
async def _fetch(session: aiohttp.ClientSession, url: str, dest: Path):
    """Download one ZIP if it doesn’t exist yet."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    async with session.get(url) as r:
        r.raise_for_status()
        async with aiofiles.open(dest, "wb") as f:
            async for chunk in r.content.iter_chunked(1 << 15):
                await f.write(chunk)


# --------------------------------------------------------------------------- #
async def main(cfg):
    print("Downloading Binance aggTrades ZIPs...")
    # Build a list like ["2024-01", "2024-02", …] using pandas' fast date_range
    months = (
        pd.date_range(start=cfg.download.start, end=cfg.download.end, freq="MS")
          .to_period("M").strftime("%Y-%m")
          .tolist()
    )

    tasks = []
    async with aiohttp.ClientSession() as sess:
        for sym, m in product(cfg.download.symbols, months):
            file = f"{sym}-aggTrades-{m}.zip"
            url  = f"{BASE}/{sym}/{file}"
            dest = Path(cfg.download.dest) / sym / file
            tasks.append(_fetch(sess, url, dest))

        # tqdm.asyncio gives a neat progress bar while awaiting gather-like tasks
        for fut in tqdm.as_completed(tasks, total=len(tasks)):
            await fut


# --------------------------------------------------------------------------- #
@hydra.main(config_path="../configs", config_name="data.yaml", version_base=None)
def run(cfg):
    """Runs the main function with the given configuration."""
    print(OmegaConf.to_yaml(cfg, resolve=True))  # Print the current configuration

    # Run the main function
    asyncio.run(main(cfg))


if __name__ == "__main__":
    run()
