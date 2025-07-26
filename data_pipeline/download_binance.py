#!/usr/bin/env python
"""
Asynchronously mirror Binance monthly aggTrades ZIPs with robust timeout,
retry and concurrency control.

Why async?
----------
`https://data.binance.vision` sits behind an S3 static site that adds
~200‑400 ms latency per object.  Downloading many files concurrently hides that
round‑trip delay and keeps the CPU & network busy without overloading the
remote host.
"""
from __future__ import annotations

import asyncio
from datetime import date
from itertools import product
from pathlib import Path
from typing import Iterable
import argparse

import aiofiles  # type: ignore
import aiohttp   # type: ignore
import pandas as pd
from tqdm.asyncio import tqdm
import hydra
from omegaconf import OmegaConf

# --------------------------------------------------------------------------- #
# Configuration constants — override in `data.yaml` if desired
# --------------------------------------------------------------------------- #
BASE_URL      = "https://data.binance.vision/data/spot/monthly/aggTrades"
CHUNK_SIZE    = 1 << 17   # 128 KiB — a good balance of syscalls vs. memory
MAX_PARALLEL  = 8          # simultaneous sockets (per event‑loop)
RETRIES       = 3          # exponential back‑off attempts on timeout/reset
CONNECT_WAIT  = 30         # seconds allowed for DNS + TLS handshake

# --------------------------------------------------------------------------- #
# Low‑level download helper
# --------------------------------------------------------------------------- #
async def _fetch(
    session: aiohttp.ClientSession,
    sema: asyncio.Semaphore,
    url: str,
    dest: Path,
) -> None:
    """Download *url* → *dest* with retries, temp‑file and 404 skip.

    The file is written to ``dest.with_suffix('.part')`` first and atomically
    renamed on success to avoid leaving zero‑byte stubs that would otherwise
    confuse incremental reruns.
    """
    if dest.exists():
        #return  # already in cache
        #print(f" {dest} (already exists)")
        return
    async with sema:  # enforce *global* concurrency limit
        tmp = dest.with_suffix(".part")
        for attempt in range(1, RETRIES + 1):
            try:
                async with session.get(url) as resp:
                    if resp.status == 404:
                        return  # month not yet published for this symbol
                    resp.raise_for_status()

                    # ensure parent dir exists exactly once
                    tmp.parent.mkdir(parents=True, exist_ok=True)

                    async with aiofiles.open(tmp, "wb") as f:
                        async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                            await f.write(chunk)

                # success — move into place atomically
                tmp.replace(dest)
                return

            except (asyncio.TimeoutError, aiohttp.ClientError):
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

                if attempt == RETRIES:
                    raise  # bubble up after final attempt

                # back‑off: 2, 4 … seconds between attempts
                await asyncio.sleep(2 ** attempt)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
async def _download_all(cfg) -> None:
    """Produce the task list from config and drive concurrent downloads."""
    # Build ["2024‑01", "2024‑02", …] fast with Pandas
    months: list[str] = (
        pd.date_range(start=cfg.download.start, end=cfg.download.end, freq="MS")
        .to_period("M")
        .strftime("%Y-%m")
        .tolist()
    )

    # aiohttp session tuned for big files
    timeout = aiohttp.ClientTimeout(
        total=None,                 # no global 5‑minute cap
        sock_connect=CONNECT_WAIT,  # fail fast on unreachable host
        sock_read=None,             # stream as long as bytes arrive
    )
    connector = aiohttp.TCPConnector(limit=MAX_PARALLEL)
    sema      = asyncio.Semaphore(MAX_PARALLEL)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as ses:
        tasks: list[asyncio.Task] = []
        for sym, month in product(cfg.download.symbols, months):
            file = f"{sym}-aggTrades-{month}.zip"
            url  = f"{BASE_URL}/{sym}/{file}"
            dest = Path(cfg.download.dest) / sym / file
            tasks.append(asyncio.create_task(_fetch(ses, sema, url, dest)))

        for fut in tqdm.as_completed(tasks, total=len(tasks)):
            await fut  # propagate any exceptions


# --------------------------------------------------------------------------- #
# CLI / Hydra entry‑point
# --------------------------------------------------------------------------- #
@hydra.main(config_path="../configs", config_name="data.yaml", version_base=None)
def run(cfg):  # noqa: D401 — simple name for Hydra
    """Hydra wrapper that prints the effective config then launches downloads."""
    print(OmegaConf.to_yaml(cfg, resolve=True))
    asyncio.run(_download_all(cfg))


if __name__ == "__main__":
    run()
