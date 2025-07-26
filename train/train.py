# train/train.py
from __future__ import annotations
import math, random, itertools
from pathlib import Path
from typing import Dict, Any
import sys
import logging

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from data_pipeline.rolling_dataset import RollingWindowDataset
from models.eiie_cnn import EIIE_CNN
from agents.eiie_agent import EIIEAgent

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm   # add at top of file
import pandas as pd
from datetime import datetime
import pickle
import os

# ───────────────────────── utility helpers ─────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bars_per_year(freq: str) -> float:
    """Convert '600s' → approx. number of bars per year."""
    secs = int(freq.rstrip("s"))
    return 365 * 24 * 3600 / secs


def sharpe(log_r: np.ndarray, ann_factor: float) -> float:
    if len(log_r) < 2:
        return float("nan")
    mu, sd = log_r.mean(), log_r.std(ddof=1)
    return float("nan") if sd == 0 else (mu * ann_factor) / (sd * math.sqrt(ann_factor))


def max_drawdown(log_equity: np.ndarray) -> float:
    wealth = np.exp(log_equity - log_equity[0])
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    return dd.min()  # negative


# ───────────────────────── single trial run ────────────────────────
def single_run(cfg: DictConfig, grid_params: Dict[str, Any]) -> Dict[str, Any]:
    # Set up logging for worker process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Force reconfiguration in worker process
    )
    logger = logging.getLogger(__name__)
    
    # ---- clone cfg and allow writes temporarily -------------------
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    OmegaConf.set_struct(cfg, False)

    # ---- map flat grid keys -> nested paths ----------------------
    mapping = {
        "window":      "data.window",
        "lr":          "agent.lr",
        "batch_size":  "agent.batch_size",
        "beta":        "agent.beta",
        "commission":  "agent.commission",
    }
    for k, v in grid_params.items():
        target = mapping.get(k, k)         # fall back to plain key
        OmegaConf.update(cfg, target, v, merge=True)

    OmegaConf.set_struct(cfg, True)        # lock again (optional)
    
    # Get sampling frequency from data config
    sampling_freq = cfg.data.resample.freq
    
    # Format the root path to replace {freq} placeholder
    formatted_root = cfg.data.root.format(freq=sampling_freq)
    
    ds_tr = RollingWindowDataset(
        root          = formatted_root,
        symbols       = cfg.data.symbols,
        window        = cfg.data.window,
        sampling_freq = sampling_freq,
        start         = cfg.data.start_train,
        end           = cfg.data.end_train,
    )
    ds_val = RollingWindowDataset(
        root          = formatted_root,
        symbols       = cfg.data.symbols,
        window        = cfg.data.window,
        sampling_freq = sampling_freq,
        start         = cfg.data.start_val,
        end           = cfg.data.end_val,
    )

    # ---- model & agent ------------------------------------------
    model = EIIE_CNN(
        m_assets   = len(cfg.data.symbols),
        in_channels= ds_tr.C,
        window     = ds_tr.n,
        long_short = cfg.model.long_short,
    )
    agent = EIIEAgent(
        model      = model,
        commission = cfg.agent.commission,
        lr         = cfg.agent.lr,
        batch_size = cfg.agent.batch_size,
        beta       = cfg.agent.beta,
        grad_clip  = cfg.agent.grad_clip,
    )

    ann = bars_per_year(sampling_freq)

    # -------- training loop --------------------------------------
    logP, rewards = 0.0, []
    for step, (X, y) in enumerate(ds_tr):
        if cfg.train.max_steps and step >= cfg.train.max_steps:
            break

        r, _ = agent.step(X, y)
        rewards.append(r)
        logP += r

        # print progress every log_every steps
        if cfg.train.log_every and (step + 1) % cfg.train.log_every == 0 or step == 0:
            logger.info(
                f"params={grid_params} "
                f"step={step}  r={r:+.4f}  logP={logP:+.2f}  sharpe={sharpe(np.asarray(rewards), ann):+.2f}"
                f"w_t={agent.w_prev.tolist()} "
            )

    train_sharpe = sharpe(np.asarray(rewards), ann)

    # -------- validation loop (no updates) -----------------------
    w_prev = torch.full((model.m,), 1.0 / model.m)
    logP_val, rewards_val = 0.0, []
    val_weights = []  # Store validation weights
    val_price_ratios = []  # Store validation price ratios
    
    for X, y in ds_val:
        X, y = X.float(), y.float()
        w_t = model(X.unsqueeze(0), w_prev.unsqueeze(0)).squeeze(0)

        cash_prev = 1.0 - w_prev.sum()
        dot = (w_prev * y).sum() + cash_prev
        mu = 1.0 - cfg.agent.commission * (w_t - w_prev).abs().sum() / 2
        r_val = math.log(max(mu.item() * dot.item(), 1e-12))

        rewards_val.append(r_val)
        logP_val += r_val
        val_weights.append(w_t.detach().cpu().numpy())
        val_price_ratios.append(y.detach().cpu().numpy())
        w_prev = w_t

    val_sharpe = sharpe(np.asarray(rewards_val), ann)

    return {
        "params": grid_params,
        "train_sharpe": train_sharpe,
        "val_sharpe":   val_sharpe,
        "val_eq":       math.exp(logP_val),
        "model_state": model.state_dict(),  # Save model weights
        "val_weights": np.array(val_weights),
        "val_price_ratios": np.array(val_price_ratios),
        "val_rewards": np.array(rewards_val),
        "symbols": cfg.data.symbols,
    }



# ────────────────────────────────────────────────────────────────────
@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    mp.set_start_method("spawn", force=True)   # needed on macOS/Win
    set_seed(cfg.seed)

    # ---- load data configuration ---------------------------------
    # Get the absolute path to the configs directory
    current_file = Path(__file__).resolve()
    configs_dir = current_file.parent.parent / "configs"
    config_path = configs_dir / "data.yaml"
    data_cfg = OmegaConf.load(config_path)
    
    # Temporarily disable struct mode to allow merging
    OmegaConf.set_struct(cfg, False)
    # Merge data config into the main config
    cfg.data = OmegaConf.merge(cfg.data, data_cfg)
    # Re-enable struct mode
    OmegaConf.set_struct(cfg, True)

    # ---- expand the grid -----------------------------------------
    grid_space = cfg.grid
    keys, vals = list(grid_space.keys()), list(grid_space.values())
    trials = [dict()] if not keys else [
        dict(zip(keys, combo)) for combo in itertools.product(*vals)
    ]
    logger.info(f"Running {len(trials)} trial(s) ...")
    logger.info("Starting worker processes...")
    
    # ---- launch processes ----------------------------------------
    results = []
    try:
        with ProcessPoolExecutor(max_workers=cfg.get("num_workers", mp.cpu_count()-2)) as pool:
            futures = {pool.submit(single_run, cfg, p): p for p in trials}
            logger.info(f"Submitted {len(futures)} tasks to worker pool")
            count = 0
            for fut in as_completed(futures):
                try:
                    logger.info("------------------------------------")
                    count += 1
                    logger.info(f"Finished trial {count}/{len(futures)} ...")
                    res = fut.result(timeout=3600)  # 1 hour timeout per task
                    results.append(res)
                    logger.info(f"⇢ params: {res['params']}  val_Sharpe={res['val_sharpe']:+.3f}, val_eq={res['val_eq']:.6f}")
                    logger.info("------------------------------------")
                    logger.info("")                
                          
                except Exception as e:
                    params = futures[fut]
                    logger.error(f"⇢ FAILED params {params}  error: {e}")
                    # Continue with other processes
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user - shutting down processes...")
        # ProcessPoolExecutor will handle cleanup automatically
        raise

    # ---- sort & report -------------------------------------------
    if results:  # Only sort if we have results
        results.sort(key=lambda r: r["val_eq"], reverse=True)
        
        # Create organized output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        longshort_str = "longshort" if cfg.model.long_short else "softmax"
        freq_str = cfg.data.resample.freq
        commission_str = f"c{cfg.agent.commission:.4f}".replace(".", "")
        
        output_folder = Path("results") / f"{timestamp}_{longshort_str}_{freq_str}_{commission_str}"
        output_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to: {output_folder}")
        
        # Prepare results dataframe (without large arrays)
        results_for_csv = []
        for row in results:
            csv_row = {**row["params"]}
            csv_row.update({k: v for k, v in row.items() 
                           if k not in ["params", "model_state", "val_weights", "val_price_ratios", "val_rewards", "symbols"]})
            results_for_csv.append(csv_row)
        
        df = pd.DataFrame(results_for_csv)
        csv_path = output_folder / "results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")
        
        # Save best model and additional data
        best_result = results[0]
        best_folder = output_folder / "best_model"
        best_folder.mkdir(exist_ok=True)
        
        # Save model weights
        model_path = best_folder / "model_weights.pth"
        torch.save(best_result["model_state"], model_path)
        logger.info(f"Best model weights saved to: {model_path}")
        
        # Save validation weights
        weights_df = pd.DataFrame(
            best_result["val_weights"],
            columns=[f"{symbol}_weight" for symbol in best_result["symbols"]]
        )
        weights_path = best_folder / "validation_weights.csv"
        weights_df.to_csv(weights_path, index=False)
        logger.info(f"Validation weights saved to: {weights_path}")
        
        # Save price ratios
        price_ratios_df = pd.DataFrame(
            best_result["val_price_ratios"],
            columns=[f"{symbol}_price_ratio" for symbol in best_result["symbols"]]
        )
        price_ratios_path = best_folder / "validation_price_ratios.csv"
        price_ratios_df.to_csv(price_ratios_path, index=False)
        logger.info(f"Price ratios saved to: {price_ratios_path}")
        
        # Save validation rewards
        rewards_df = pd.DataFrame({
            "step": range(len(best_result["val_rewards"])),
            "log_return": best_result["val_rewards"],
            "cumulative_log_return": np.cumsum(best_result["val_rewards"])
        })
        rewards_path = best_folder / "validation_rewards.csv"
        rewards_df.to_csv(rewards_path, index=False)
        logger.info(f"Validation rewards saved to: {rewards_path}")
        
        # Save configuration
        config_path = best_folder / "config.yaml"
        with open(config_path, 'w') as f:
            OmegaConf.save(cfg, f)
        logger.info(f"Configuration saved to: {config_path}")
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "longshort_mode": cfg.model.long_short,
            "frequency": cfg.data.resample.freq,
            "symbols": best_result["symbols"],
            "best_params": best_result["params"],
            "best_val_sharpe": best_result["val_sharpe"],
            "best_val_equity": best_result["val_eq"],
            "num_trials": len(results)
        }
        metadata_path = best_folder / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Metadata saved to: {metadata_path}")
        
        logger.info("\n=========== LEADERBOARD (val Sharpe) ===========")
        for r in results:
            logger.info(f"{r['val_sharpe']:+.3f}  eq={r['val_eq']:.6f}  params={r['params']}")
        logger.info(f"BEST: {results[0]['params']}")
        logger.info(f"All results saved in folder: {output_folder}")
    else:
        logger.warning("\n⚠️  No successful results to report")


if __name__ == "__main__":
    main()
