# test_commission.py
import torch
import math


def commission_factor(w_new, w_prev, y, c):
    """
    Reference implementation of the *correct* formula (paper Eq. 7 & 8):

        w'ₜ  = (y ⊙ w_{t-1}) / (y · w_{t-1})
        μₜ  = 1 – c · ‖ w'ₜ – wₜ ‖₁      (assets slice only)

    Cash (last column) is appended as 1.
    """
    device = w_new.device
    y_full = torch.cat([y, torch.ones(1, device=device)])
    w_prime = (w_prev * y_full) / (w_prev * y_full).sum()        # drifted weights

    turnover = (w_new[:-1] - w_prime[:-1]).abs().sum()           # assets only
    mu = 1.0 - c * turnover
    return mu


def test_commission_factor():
    # --- scenario ---------------------------------------------------------
    # start with 40 % + 40 % assets, 20 % cash
    w_prev = torch.tensor([0.40, 0.40, 0.20])
    # market moves: asset‑0 +10 %, asset‑1 −10 %
    y      = torch.tensor([1.10, 0.90])
    # agent wants to end with 45 % / 35 % / 20 %
    w_new  = torch.tensor([0.45, 0.35, 0.20])
    c      = 2e-4   # 0.02 % per side  → 0.04 % round‑trip

    mu_ref = commission_factor(w_new, w_prev, y, c)

    # manual hand‑calc:  drifted w'  = [0.44, 0.36, 0.20]
    # turnover          = |0.45‑0.44| + |0.35‑0.36| = 0.02
    # μ                 = 1 – 0.0002·0.02 = 0.999996
    mu_manual = 1 - c * 0.02

    assert math.isclose(mu_ref, mu_manual, rel_tol=1e-8)

    # show the contrast to the **naïve** formula that ignores drift
    naive_turnover = (w_new[:-1] - w_prev[:-1]).abs().sum()      # 0.10
    mu_naive = 1 - c * naive_turnover                           # 0.99998
    print(f"μ_correct={mu_ref:.6f}   μ_naive={mu_naive:.6f}")


if __name__ == "__main__":
    test_commission_factor()
