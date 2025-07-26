import torch
from models.eiie_cnn import EIIE_CNN

def test_forward_shapes():
    B, C, m, n = 4, 5, 3, 50        # batch, channels, assets, window
    X = torch.randn(B, C, m, n)
    w_prev = torch.full((B, m), 1/m)

    net = EIIE_CNN(m_assets=m, in_channels=C, window=n)
    w = net(X, w_prev)

    assert w.shape == (B, m)
    # weights should be non‑negative and sum to 1 (long‑only)
    assert torch.all(w >= 0)
    s = w.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-6)
