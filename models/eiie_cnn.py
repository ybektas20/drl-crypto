import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EIIE_CNN(nn.Module):
    def __init__(self, *, m_assets: int, in_channels: int,
                 window: int, long_short: bool = False):
        super().__init__()
        self.m = m_assets
        self.long_short = long_short

        # ───────── CNN feature extractor ─────────
        self.conv1 = nn.Conv2d(in_channels,  8, kernel_size=(1, 3))
        self.in1   = nn.InstanceNorm2d(8, affine=False)          # NEW
        eff = max(1, window - 2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=(1, eff))
        self.in2   = nn.InstanceNorm2d(32, affine=False)         # NEW
        self.conv3 = nn.Conv2d(33, 1, kernel_size=1)  # Updated to accept 33 channels (32 from conv2 + 1 from w_prev)

        # Kaiming init
        for c in (self.conv1, self.conv2, self.conv3):
            nn.init.kaiming_uniform_(c.weight, a=math.sqrt(5))
            if c.bias is not None:
                nn.init.zeros_(c.bias)

        # learnable bias to break symmetry
        self.asset_bias = nn.Parameter(torch.empty(m_assets))
        nn.init.uniform_(self.asset_bias, -0.3, 0.3)

        # learnable cash bias for both modes
        self.cash_bias = nn.Parameter(torch.zeros(1))

    # ---------- projections ----------
    def _l1_projection(self, x):
        # Include cash as an option in L1 projection too
        logits = torch.cat([x, self.cash_bias.expand(x.size(0), 1)], 1)
        u = logits
        weights = u / (u.abs().sum(1, keepdim=True) + 1e-8)
        return weights[:, :self.m]  # Return only asset weights, cash is implicit

    def _softmax_cash(self, s):
        logits = torch.cat([s, self.cash_bias.expand(s.size(0), 1)], 1)
        return F.softmax(logits, 1)

    # ---------- forward ----------
    def forward(self, X, w_prev):
        z = F.relu(self.in1(self.conv1(X)))   # norm‑after‑conv
        z = F.relu(self.in2(self.conv2(z)))
        # Expand w_prev to match shape
        w_prev_exp = w_prev.unsqueeze(-1).unsqueeze(1)  # (B, 1, m, 1)
        z = torch.cat([z, w_prev_exp], dim=1)           # add as extra feature channel

        scores = self.conv3(z).squeeze(1).squeeze(-1) + self.asset_bias

        if self.long_short:
            return self._l1_projection(scores)
        return self._softmax_cash(scores)[:, : self.m]
