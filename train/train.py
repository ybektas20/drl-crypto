# train/train.py — fully revised for explicit cash column
# -----------------------------------------------------------------------------
from __future__ import annotations
import itertools, logging, math, random, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from agents.eiie_agent import EIIEAgent
from data_pipeline.rolling_dataset import RollingWindowDataset
from models.eiie_cnn import EIIE_CNN

# ───────────────────────── helper utils ───────────────────────────

def set_seed(seed: int) -> None:
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def bars_per_year(freq: str) -> float:
    secs = int(freq.rstrip("s"))
    return 365 * 24 * 3600 / secs


def sharpe(log_r: np.ndarray, ann_factor: float) -> float:
    if len(log_r) < 2:
        return float("nan")
    mu, sd = log_r.mean(), log_r.std(ddof=1)
    return float("nan") if sd == 0 else (mu * ann_factor) / (sd * math.sqrt(ann_factor))

# ------------------------------------------------------------------
#  single grid‑point run (executes in worker)
# ------------------------------------------------------------------

def single_run(cfg: DictConfig, grid_params: Dict[str, Any]) -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)
    logger = logging.getLogger(__name__)

    # Set seed in worker process
    set_seed(cfg.seed)

    # clone and override cfg ---------------------------------------------------
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    OmegaConf.set_struct(cfg, False)
    for k, v in grid_params.items():
        OmegaConf.update(cfg, {
            "window": "data.window",
            "lr": "agent.lr",
            "beta": "agent.beta",
            "batch_size": "agent.batch_size",
            "commission": "agent.commission",
        }.get(k, k), v, merge=True)
    OmegaConf.set_struct(cfg, True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # datasets -----------------------------------------------------------------
    freq   = cfg.data.resample.freq
    root   = Path(cfg.data.root.format(freq=freq))
    ds_tr  = RollingWindowDataset(root, cfg.data.symbols, cfg.data.window, freq, cfg.data.start_train, cfg.data.end_train)
    ds_val = RollingWindowDataset(root, cfg.data.symbols, cfg.data.window, freq, cfg.data.start_val,   cfg.data.end_val)

    # model & agent ------------------------------------------------------------
    # Set seed again before model creation to ensure deterministic initialization
    set_seed(cfg.seed)
    model = EIIE_CNN(m_assets=len(cfg.data.symbols), in_channels=ds_tr.C, window=ds_tr.n, long_short=cfg.model.long_short).to(device)
    agent = EIIEAgent(
        model,
        commission=cfg.agent.commission,
        lr=cfg.agent.lr,
        batch_size=cfg.agent.batch_size,
        beta=cfg.agent.beta,
        grad_clip=cfg.agent.grad_clip,
        lambda_turnover=cfg.agent.lambda_turnover,   
    )

    ann = bars_per_year(freq)

    # ---------------------------- training loop ------------------------------
    logP, rewards = 0.0, []
    for step, (X_cpu, y_cpu) in enumerate(ds_tr):
        if cfg.train.max_steps and step >= cfg.train.max_steps:
            break
        r, _ = agent.step(X_cpu, y_cpu)
        rewards.append(r)
        logP += r
        if cfg.train.log_every and ((step + 1) % cfg.train.log_every == 0 or step == 0):
            logger.info(
                f"params={grid_params}  step={step}  r={r:+.4f}  logP={logP:+.2f}  sharpe={sharpe(np.asarray(rewards), ann):+.2f}"
                f"\nweights={agent.w_prev.cpu().numpy() if agent.w_prev is not None else None}"
                )

    train_sharpe = sharpe(np.asarray(rewards), ann)

    # --------------------------- validation loop (online learning ON) ------
    logP_val, rewards_val = 0.0, []
    val_weights, val_price_ratios = [], []

    for X_cpu, y_cpu in ds_val:                       # NOTE: we call agent.step → continues learning
        r_val, w_t = agent.step(X_cpu, y_cpu)         # learning ON
        logP_val += r_val
        rewards_val.append(r_val)
        val_weights.append(w_t.detach().numpy())      # detach before converting to numpy
        val_price_ratios.append(y_cpu.numpy())        # cash ratio is always 1, not stored

    val_sharpe = sharpe(np.asarray(rewards_val), ann)

    return {
        "params": grid_params,        
        "train_sharpe": train_sharpe,
        "val_sharpe": val_sharpe,
        "train_eq": math.exp(logP),
        "val_eq": math.exp(logP_val),
        "model_state": model.state_dict(),
        "val_weights": np.array(val_weights),
        "val_price_ratios": np.array(val_price_ratios),
        "val_rewards": np.array(rewards_val),
        "symbols": cfg.data.symbols,
    }

# ---------------------------------------------------------------------------
#  main entry (Hydra)
# ---------------------------------------------------------------------------
@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)

    # Set seed early before any random operations
    set_seed(cfg.seed)

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    # merge data cfg -----------------------------------------------------------
    data_cfg = OmegaConf.load(Path(__file__).resolve().parent.parent / "configs" / "data.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.data = OmegaConf.merge(cfg.data, data_cfg)
    OmegaConf.set_struct(cfg, True)

    # build grid ---------------------------------------------------------------
    trials = [dict()] if not cfg.grid else [dict(zip(cfg.grid.keys(), combo)) for combo in itertools.product(*cfg.grid.values())]
    logger.info(f"Running {len(trials)} trial(s)…")

    results: list[dict] = []
    workers = cfg.get("num_workers", max(1, mp.cpu_count() - 2))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(single_run, cfg, p): p for p in trials}
        for i, fut in enumerate(as_completed(futures), 1):
            params = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                logger.info(f"[{i}/{len(trials)}]  train_Sharpe={res['train_sharpe']:+.3f}  val_Sharpe={res['val_sharpe']:+.3f} train_eq={res['train_eq']:.6f}  val_eq={res['val_eq']:.6f}  params={params}")
            except Exception as e:
                logger.error(f"[{i}/{len(trials)}]  FAILED  params={params}  error: {e}")

    if not results:
        logger.warning("No successful results to report.")
        return

    # ---------------- save artefacts ----------------------------------------
    results.sort(key=lambda r: r["val_eq"], reverse=True)

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode        = "longshort" if cfg.model.long_short else "softmax"
    freq        = cfg.data.resample.freq
    folder      = Path("results") / f"{ts}_{mode}_{freq}_c{cfg.agent.commission:.4f}".replace(".", "")
    folder.mkdir(parents=True, exist_ok=True)

    # dataframe without bulky arrays
    df_rows = []
    for r in results:
        row = {**r["params"], **{k: v for k, v in r.items() if k not in {"params", "model_state", "val_weights", "val_price_ratios", "val_rewards", "symbols"}}}
        df_rows.append(row)
    pd.DataFrame(df_rows).to_csv(folder / "results.csv", index=False)

    best = results[0]
    best_dir = folder / "best_model"
    best_dir.mkdir()

    logger.info(f"Best model: train_Sharpe={best['train_sharpe']:+.3f} val_Sharpe={best['val_sharpe']:+.3f} train_eq={best['train_eq']:.6f}, val_eq={best['val_eq']:.6f}  params={best['params']}")

    # save model
    torch.save(best["model_state"], best_dir / "model_weights.pth")

    # save validation artefacts
    cols_assets = [*best["symbols"], "CASH_weight"]
    pd.DataFrame(best["val_weights"], columns=cols_assets).to_csv(best_dir / "validation_weights.csv", index=False)
    pd.DataFrame(best["val_price_ratios"], columns=best["symbols"]).to_csv(best_dir / "validation_price_ratios.csv", index=False)
    pd.DataFrame({"step": np.arange(len(best["val_rewards"])), "log_return": best["val_rewards"], "cum_log_return": np.cumsum(best["val_rewards"])}).to_csv(best_dir / "validation_rewards.csv", index=False)

    OmegaConf.save(cfg, best_dir / "config.yaml")

    logger.info(f"Saved all artefacts to {folder}")

if __name__ == "__main__":
    main()
