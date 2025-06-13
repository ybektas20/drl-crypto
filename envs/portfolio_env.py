# drl_crypto/envs/portfolio_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BinancePortfolioEnv(gym.Env):
    """
    Passive-maker/occasional-taker portfolio environment.

    Observation:
        frame  : (A, F, W)   z-score feature cube
        w_prev : (A+1,)      realised weights at t-1
        bid    : (A,)        synthetic best bids at t
        ask    : (A,)        synthetic best asks at t
    Action  (length 2A+1):
        [ w_raw(1..A), w_raw_cash , m_frac(1..A) ]
        • w_raw  ∈ [-1, 1]   desired net weights before projection
        • m_frac ∈ [0, 1]    fraction of Δweight to execute immediately (taker)
    Reward:
        Sharpe-like   r̃_t = log(V_t/V_{t-1})  –  λ·√var_{t-1}
    """

    metadata = {"render_modes": []}

    def __init__(self, feature_cube, bids, asks,
                 window=50,
                 maker_fee=-1e-4,
                 taker_fee=4e-4,
                 lambda_risk=0.3,
                 var_mu=0.1):
        super().__init__()

        # --- static data ---------------------------------------------------
        self.X     = feature_cube            # (T, A, F)
        self.bids  = bids.astype(np.float32)
        self.asks  = asks.astype(np.float32)
        self.T, self.A, self.F = feature_cube.shape
        self.W     = window

        # --- hyper-parameters ---------------------------------------------
        self.maker_fee   = maker_fee
        self.taker_fee   = taker_fee
        self.lambda_risk = lambda_risk
        self.var_mu      = var_mu

        # --- gym spaces ----------------------------------------------------
        self.observation_space = spaces.Dict({
            "frame":  spaces.Box(-np.inf, np.inf, (self.A, self.F, window), np.float32),
            "w_prev": spaces.Box(-1., 1., (self.A+1,), np.float32),
            "bid":    spaces.Box(0.,  np.inf, (self.A,), np.float32),
            "ask":    spaces.Box(0.,  np.inf, (self.A,), np.float32),
        })
        low  = np.concatenate([np.full(self.A+1, -1.), np.zeros(self.A)])
        high = np.ones(2*self.A+1, dtype=np.float32)
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        # --- runtime state -------------------------------------------------
        self.pointer   = None          # current time index
        self.w_prev    = None          # realised weights at t-1
        self.pos       = None          # units of each asset
        self.cash      = None
        self.value     = None
        self.var       = 0.0
        self.open_orders = []          # list[tuple(asset, side, px, qty)]

    # ------------------------------------------------------------------ #
    # helper: L1-projection  |w_assets|₁ + cash = 1, cash ≥ 0
    def _project(self, w):
        w = w.copy()
        w[-1] = 0.
        l1 = np.sum(np.abs(w[:-1]))
        if l1 > 1.:
            w[:-1] /= l1
        w[-1] = 1. - np.sum(np.abs(w[:-1]))
        return w

    # ------------------------------------------------------------------ #
    def _mark_to_market(self):
        bid, ask = self.bids[self.pointer], self.asks[self.pointer]
        price = np.where(self.pos >= 0, bid, ask)        # long@bid, short@ask
        inv_val = np.sum(self.pos * price)
        return self.cash + inv_val

    # ------------------------------------------------------------------ #
    def _get_obs(self):
        frame_slice = self.X[self.pointer-self.W:self.pointer]   # (W,A,F)
        frame = frame_slice.transpose(1, 2, 0).astype(np.float32)
        return {
            "frame":  frame,
            "w_prev": self.w_prev.astype(np.float32),
            "bid":    self.bids[self.pointer],
            "ask":    self.asks[self.pointer],
        }

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pointer = self.W
        self.pos   = np.zeros(self.A, dtype=np.float32)
        self.cash  = 1.0                                    # 1 unit NAV
        self.value = 1.0
        self.w_prev = np.zeros(self.A+1, dtype=np.float32)
        self.w_prev[-1] = 1.0
        self.open_orders.clear()
        self.var = 0.0
        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    def step(self, action):
        action = np.asarray(action, np.float32)
        w_target = self._project(action[:self.A+1])
        m_frac   = np.clip(action[self.A+1:], 0., 1.)

        delta = w_target[:-1] - self.w_prev[:-1]         # desired change
        taker_delta = m_frac * delta
        maker_delta = (1. - m_frac) * delta

        bid, ask = self.bids[self.pointer], self.asks[self.pointer]

        # ---- 1) execute taker immediately ------------------------------
        taker_qty   = taker_delta * self.value           # qty in notional terms
        taker_pnl   = 0.0
        for i, q in enumerate(taker_qty):
            if q == 0:
                continue
            if q > 0:   # buying
                fill_px = ask[i]
                fee = self.taker_fee * abs(q)
                self.pos[i] += q / fill_px
                self.cash   -= q + fee
                taker_pnl   -= fee
            else:       # selling
                fill_px = bid[i]
                fee = self.taker_fee * abs(q)
                self.pos[i] += q / fill_px
                self.cash   -= q + fee
                taker_pnl   -= fee

        # ---- 2) queue maker orders -------------------------------------
        for i, q in enumerate(maker_delta * self.value):
            if q == 0:
                continue
            if q > 0:   # buy order at bid price
                self.open_orders.append((i, +1, bid[i], q))
            else:       # sell order at ask price
                self.open_orders.append((i, -1, ask[i], -q))

        # ---- 3) check existing maker orders for fill -------------------
        filled_orders = []
        for idx, (i, side, px, qty) in enumerate(self.open_orders):
            # side +1 => buy, needs ask <= px
            if side == +1 and self.asks[self.pointer][i] <= px:
                fill_px = self.asks[self.pointer][i]
            # side -1 => sell, needs bid >= px
            elif side == -1 and self.bids[self.pointer][i] >= px:
                fill_px = self.bids[self.pointer][i]
            else:
                continue

            # execute
            notional = qty
            fee = self.maker_fee * abs(notional)
            if side == +1:      # buy
                self.pos[i] += qty / fill_px
                self.cash   -= notional + fee
            else:               # sell
                self.pos[i] -= qty / fill_px
                self.cash   += notional - fee
            filled_orders.append(idx)

        # remove filled orders (reverse iterate)
        for idx in reversed(filled_orders):
            self.open_orders.pop(idx)

        # ---- 4) PnL & reward -------------------------------------------
        prev_val = self.value
        self.value = self._mark_to_market()
        raw_ret = np.log(self.value / prev_val)
        self.var = self.var_mu * raw_ret**2 + (1-self.var_mu)*self.var
        reward = raw_ret - self.lambda_risk * np.sqrt(self.var + 1e-12)

        # ---- 5) update realised weights for next observation -----------
        w_assets = (self.pos * np.where(self.pos>=0, bid, ask)) / self.value
        cash_w   = self.cash / self.value
        self.w_prev = np.append(w_assets, cash_w)

        # ---- 6) step pointer & termination -----------------------------
        self.pointer += 1
        terminated = (self.pointer >= self.T)
        truncated  = False
        return self._get_obs(), float(reward), terminated, truncated, {}
