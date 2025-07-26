# agents/eiie_agent.py
# ================================================================
from __future__ import annotations
import random
from collections import deque
from typing import Deque, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

Tensor = torch.Tensor


class EIIEAgent:
    """Reinforcement‑learning wrapper around an EIIE network."""

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        commission: float = 2.5e-3,
        lr: float = 1e-4,
        batch_size: int = 32,
        beta: float = 0.2,
        grad_clip: float = 5.0,
        buffer_cap: int = 10_000,
    ) -> None:
        self.net = model
        self.c = commission
        self.opt = Adam(self.net.parameters(), lr=lr)

        self.batch = batch_size
        self.beta = beta
        self.grad_clip = grad_clip

        # start with uniform CRP
        self.w_prev = torch.full((model.m,), 1 / model.m)

        # FIFO replay buffers
        self.X_buf: Deque[Tensor] = deque(maxlen=buffer_cap)
        self.y_buf: Deque[Tensor] = deque(maxlen=buffer_cap)
        self.w_buf: Deque[Tensor] = deque(maxlen=buffer_cap)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _commission_factor(self, w: Tensor) -> Tensor:
        """μ_t = 1 − c · |w_t − w_{t‑1}|₁ / 2"""
        cost = (w - self.w_prev).abs().sum() / 2.0
        return 1.0 - self.c * cost

    @torch.no_grad()
    def act(self, X: Tensor) -> Tensor:
        """Greedy action without gradient tracking."""
        return self.net(X.unsqueeze(0), self.w_prev.unsqueeze(0)).squeeze(0)

    # ------------------------------------------------------------------ #
    # environment interface
    # ------------------------------------------------------------------ #
    def step(self, X: Tensor, y: Tensor) -> Tuple[float, Tensor]:
        """
        X : (C, m, n)  features at bar t
        y : (m,)       price‑ratios p_t / p_{t-1}
        """
        X = X.float()
        y = y.float()

        # 1) net proposes new allocation
        w_t = self.net(X.unsqueeze(0), self.w_prev.unsqueeze(0)).squeeze(0)

        # 2) realised reward
        mu = self._commission_factor(w_t)
        cash_prev = 1.0 - self.w_prev.sum()
        dot = (self.w_prev * y).sum() + cash_prev
        reward = torch.log(torch.clamp(mu * dot, min=1e-8))

        # 3) store transition
        self.X_buf.append(X)
        self.y_buf.append(y)
        self.w_buf.append(self.w_prev.clone())

        # 4) update memory
        self.w_prev = w_t.detach()

        # 5) learn
        if len(self.X_buf) >= self.batch:
            self._update()

        return reward.item(), w_t.detach()

    # ------------------------------------------------------------------ #
    # internal update
    # ------------------------------------------------------------------ #
    # ───────────────── gradient / replay update ───────────────────
    def _update(self) -> None:
        """
        Sample a β‑geometric mini‑batch, compute minus‑log‑return loss,
        and run one optimiser step.
        """
        # ----- 1) geometric‑decay sampling -------------------------
        N = len(self.X_buf)
        geom = torch.tensor([(1.0 - self.beta) ** i for i in range(N - 1, -1, -1)])
        probs = geom / geom.sum()
        idx = random.choices(range(N), weights=probs, k=self.batch)

        # ----- 2) build mini‑batch --------------------------------
        Xb  = torch.stack([self.X_buf[i] for i in idx])    # (B, C, m, n)
        yb  = torch.stack([self.y_buf[i] for i in idx])    # (B, m)   price ratios
        wpb = torch.stack([self.w_buf[i] for i in idx])    # (B, m)   w_{t‑1}

        wb  = self.net(Xb, wpb)                            # (B, m)   w_t (current)

        # ----- 3) log‑return with commission ----------------------
        cash_post = 1.0 - wb.sum(dim=1, keepdim=True)                  # after rebalance
        dot       = (wb * yb).sum(dim=1, keepdim=True) + cash_post

        mu = 1.0 - self.c * (wb - wpb).abs().sum(dim=1, keepdim=True) / 2.0

        ret  = torch.log(torch.clamp(mu * dot, min=1e-8))              # (B, 1)
        loss = -ret.mean()

        # ----- 4) optimiser step ----------------------------------
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.opt.step()
