# agents/eiie_agent.py
# ========================================================================== #
from __future__ import annotations
import random
from collections import deque
from typing import Deque, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# utility
# --------------------------------------------------------------------------- #
def _device_of(module: torch.nn.Module) -> torch.device:
    """Return the device of the first parameter of *module*."""
    return next(module.parameters()).device


# --------------------------------------------------------------------------- #
# main class
# --------------------------------------------------------------------------- #
class EIIEAgent:
    """
    Ensemble‑of‑Identical‑Independent‑Evaluators (EIIE) trading agent.

    • Network outputs a weight vector of length m + 1 (m assets + cash).  
    • We use *time‑shifted* replay:     (Xₜ , wₜ) ─▶ reward of wₜ₋₁.  
    • Replay buffer stores full transitions so training can run off‑policy.
    """

    # ------------------------------------------------------------------ #
    # constructor
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        commission: float = 2.5e-3,   # round‑trip fee (e.g. 0.0025 = 25 bp)
        lr: float = 1e-4,
        batch_size: int = 32,
        beta: float = 0.2,            # geometric sampling decay
        grad_clip: float = 5.0,
        buffer_cap: int = 10_000,
        lambda_turnover: float = 1.0e-3,  # regularisation for turnover
    ) -> None:
        # store args ----------------------------------------------------
        self.net   = model
        self.c     = commission
        self.batch = batch_size
        self.beta  = beta
        self.grad_clip = grad_clip
        self.lam_t = lambda_turnover
        # optimiser -----------------------------------------------------
        self.opt = Adam(self.net.parameters(), lr=lr)

        # device & starting allocation ---------------------------------
        self.device = _device_of(model)
        m_plus_1    = model.m + 1
        self.w_prev = torch.zeros(m_plus_1, device=self.device)
        self.w_prev[-1] = 1.0  # start with full cash

        # replay buffers: (Xₜ, yₜ₊₁, wₜ) -------------------------------
        self.X_buf: Deque[Tensor] = deque(maxlen=buffer_cap)
        self.y_buf: Deque[Tensor] = deque(maxlen=buffer_cap)
        self.w_buf: Deque[Tensor] = deque(maxlen=buffer_cap)

        # pending state waiting for yₜ₊₁ -------------------------------
        self._pending: Tuple[Tensor, Tensor] | None = None   # (Xₜ , wₜ)

        ##dbg
        self.update_count = 0

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _commission_factor(self, w_new: Tensor,
                        w_prev: Tensor,              # w_{t-1}
                        y: Tensor) -> Tensor:        # y_t (m,)
        """
        μ_t = 1 - c * || w'_{t} - w_{t} ||_1   (round‑trip commission)
        w'_{t} = (y_t ⊙ w_{t-1}) / (y_t·w_{t-1})
        """
        # append cash =1 so we can reuse code        
        w_prime = (w_prev * y) / (w_prev * y).sum()       # drifted weights

        turnover = (w_new[..., :-1] - w_prime[..., :-1]).abs().sum(dim=-1, keepdim=True)
        mu = torch.clamp(1.0 - self.c * turnover, min=1e-6)
        return mu
                                  

    @torch.no_grad()
    def act(self, X: Tensor) -> Tensor:
        """
        Convenience wrapper: return the *deterministic* action the current
        policy takes for a single observation X (no grad, shape (C,m,n)).
        """
        X = X.to(self.device).unsqueeze(0)                    # → (1,C,m,n)
        w = self.net(X, self.w_prev.unsqueeze(0)).squeeze(0)  # (m+1,)
        return w.cpu()

    # ------------------------------------------------------------------ #
    # environment interface (one online step)
    # ------------------------------------------------------------------ #
    def step(self,
             X_cpu: Tensor,    # state at t        (C,m,n)
             y_cpu: Tensor     # price relatives   (m,)
             ) -> Tuple[float, Tensor]:
        """
        Feed one new bar to the agent.

        Returns:
            • realised log‑return of *previous* allocation (float)
            • new allocation wₜ (on CPU, for logging)
        """
        # move to device -----------------------------------------------
        X = X_cpu.to(self.device).float()
        y = y_cpu.to(self.device).float()

        # 1) current policy action -------------------------------------
        w_t = self.net(X.unsqueeze(0), self.w_prev.unsqueeze(0)).squeeze(0)

        # 2) compute reward for w_{t‑1} once yₜ is known ---------------
        if self._pending is not None:
            X_prev_cpu, w_prev_detached = self._pending             # w_{t‑1}
            y_full = torch.cat([y, torch.ones(1, device=self.device)])  # cash=1
            mu_prev  = self._commission_factor(w_t, w_prev_detached, y_full)
            dot_prev = (w_prev_detached * y_full).sum()
            reward_prev = torch.log(torch.clamp(mu_prev * dot_prev, 1e-8))
            # store transition in replay buffer
            self.X_buf.append(X_prev_cpu)
            self.y_buf.append(y_cpu)
            self.w_buf.append(w_prev_detached.cpu())
        else:
            reward_prev = torch.tensor(0.0, device=self.device)

        # 3) cache current (Xₜ, wₜ) for the next step ------------------
        self.w_prev = w_t.detach()                    # detach for stability
        self._pending = (X_cpu, self.w_prev)

        # 4) learn off one mini‑batch if enough data -------------------
        if len(self.X_buf) >= self.batch:
            self._update()

        return reward_prev.item(), w_t.cpu()

    # ------------------------------------------------------------------ #
    # internal update (online stochastic‑batch learning)
    # ------------------------------------------------------------------ #
    def _update(self) -> None:
        """
        Sample a geometrically‑weighted mini‑batch from the replay buffer
        and take one policy‑gradient step on the *batch average* log‑return.
        """
        N = len(self.X_buf)
        # geometric sampling prob ~ (1‑β)ᶦ  (newer points preferred)
        geom = torch.tensor([(1.0 - self.beta) ** i
                             for i in range(N - 1, -1, -1)])
        idx = random.choices(range(N),
                             weights=(geom / geom.sum()).tolist(),
                             k=self.batch)

        # assemble mini‑batch -----------------------------------------
        Xb  = torch.stack([self.X_buf[i] for i in idx]).to(self.device)
        yb  = torch.stack([self.y_buf[i] for i in idx]).to(self.device)
        wpb = torch.stack([self.w_buf[i] for i in idx]).to(self.device)  # prev w

        # forward pass -------------------------------------------------
        wb = self.net(Xb, wpb)                              # (B, m+1)

        # --- already in your code ---
        yb_full = torch.cat([yb, torch.ones_like(yb[:, :1])], dim=1)  # (B, m+1)

                # drift the *previous* allocation for fee calculation only
        w_prime = (wpb * yb_full) / (wpb * yb_full).sum(dim=1, keepdim=True)

        # ------- gradient flows through wb -------
        dot = (wb * yb_full).sum(dim=1, keepdim=True)          #  ← NO .detach()


        w_prime = (wpb * yb_full) / (wpb * yb_full).sum(dim=1, keepdim=True)

        turnover = (wb[..., :-1] - w_prime[..., :-1]).abs().sum(dim=1, keepdim=True)
        mu       = torch.clamp(1.0 - self.c * turnover, 1e-6)   # 0.5 = per‑side

        base_loss = -torch.log(mu * dot).mean()
        l2_turn   = self.lam_t * (turnover ** 2).mean()
        loss      = base_loss + l2_turn

        if self.update_count % self.batch == 0:
            print(f"turn {turnover.mean():.3f}  mu {mu.mean():.6f} "
                  f"dot {dot.mean():.4f}  loss {loss.item():+.4f}")
        self.update_count += 1


        # backward + optimiser step -------------------------
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        """ grad_norm = 0.0
        for p in self.net.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().norm().item() ** 2
        grad_norm **= 0.5
        print(f"[dbg] grad_norm = {grad_norm:.4e}")
        clip_grad_norm_(self.net.parameters(), self.grad_clip) """
        self.opt.step()

