from torch import Tensor
import torch


def _commission_factor(
                        w_new: Tensor,
                        w_old: Tensor) -> Tensor:
    """
    Single‑period wealth retention factor μ given two allocations.
    """
    turnover = (w_new[..., :-1] - w_old[..., :-1]).abs().sum(dim=-1, keepdim=True)
    mu = torch.clamp(1.0 - 0.5 * turnover, min=1e-6)   # already round‑trip
    return mu                                             # shape (·, 1)

if __name__ == "__main__":
    mu=_commission_factor(torch.tensor([0.3, 0.2, 0.5]), torch.tensor([0.2, 0.3, 0.5]))
    print(mu)