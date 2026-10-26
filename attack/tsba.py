"""TSBA-adapted: a PyTorch adaptation of TSBA for WiFi-CSI pose regression.

Reference: Jiang, Ma, Erfani, Bailey, "Backdoor Attacks on Time Series: A
Generative Approach", SaTML 2023 — https://github.com/yujingmarkjiang/Time_Series_Backdoor_Attack

What is preserved from the public implementation
------------------------------------------------
* The sample-specific multiplicative trigger, ``clip_add`` in models/noise_gan.py::

      x' = x * (1 + 0.1 * tanh(G(x)))

  Our ``inject_tensor`` at ``dose=1.0, eps=0.1`` reproduces this rule exactly.
  Lower doses extend it so the attack can be evaluated as a dose-response.
* A generator emitting a same-shaped, tanh-bounded pattern per sample.
* Alternating generator/victim optimization with the victim frozen while the
  generator is updated (``bd_trainable=False`` upstream).
* A clean warm-up of the victim before the alternating phase (10 epochs).
* Adam at lr 1e-3 for the generator.

What differs, and why
---------------------
* **Generator architecture.** Upstream is Conv1D with filter counts scaled by
  the variable count (``512 * n_vars``); CSI has 114-180 subcarriers, so a
  literal port is prohibitively large. This is a compact Conv2D generator that
  learns over both CSI axes (subcarrier x packet). The two generators are
  therefore *different functions*: only the injection rule is shared, not the
  end-to-end trigger.
* **Objective.** Upstream optimizes categorical cross-entropy toward a target
  label; this optimizes target-pose MPJPE, because the victim is a regressor.
* **Alternation schedule.** Upstream alternates in large blocks (generator 20
  epochs, victim 5 epochs, then a further victim pass on clean data each outer
  round). We use finer-grained online alternating updates (a few generator
  steps per victim step) to accommodate the computational cost of CSI pose
  regression. This is an approximation of the upstream schedule, not an
  equivalent of it, and the upstream clean-data victim pass has no counterpart
  here.

The 0.1 factor is a multiplicative relative bound (|dx/x| <= 0.1), not an
additive norm budget or a fraction of the data range.

Reported as ``TSBA-adapted``; it is not an exact reproduction.
"""

from __future__ import annotations

import torch
from torch import nn


class TSBAGenerator(nn.Module):
    """Generate a bounded perturbation pattern with the same shape as CSI."""

    def __init__(self, n_ant: int = 3, hidden: int = 32):
        super().__init__()
        if hidden < 4:
            raise ValueError(f'tsba hidden width must be >= 4, got {hidden}')
        self.net = nn.Sequential(
            nn.Conv2d(n_ant, hidden, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden * 2, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_ant, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        if csi.ndim != 4:
            raise ValueError(f'TSBA expects (B,A,S,T), got {tuple(csi.shape)}')
        return self.net(csi)


class TSBATrigger(nn.Module):
    """Differentiable, sample-specific TSBA trigger.

    Injection is deliberately deferred until after dataset normalization: the
    generator must stay inside the autograd graph, and TSBA attacks the victim's
    actual input representation rather than raw complex CSI.
    """

    requires_deferred_injection = True

    def __init__(self, n_ant: int = 3, hidden: int = 32):
        super().__init__()
        self.generator = TSBAGenerator(n_ant=n_ant, hidden=hidden)

    def pattern(self, csi: torch.Tensor) -> torch.Tensor:
        return self.generator(csi)

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        return self.pattern(csi)

    def inject_tensor(self, csi: torch.Tensor, dose, eps: float = 0.1) -> torch.Tensor:
        """Apply TSBA's multiplicative trigger at a scalar or per-sample dose."""
        if not torch.is_tensor(dose):
            dose = torch.as_tensor(dose, dtype=csi.dtype, device=csi.device)
        else:
            dose = dose.to(device=csi.device, dtype=csi.dtype)
        if dose.ndim == 0:
            dose = dose.expand(csi.shape[0])
        if dose.ndim != 1 or dose.shape[0] != csi.shape[0]:
            raise ValueError(
                f'dose must be scalar or shape ({csi.shape[0]},), got {tuple(dose.shape)}')
        scale = dose[:, None, None, None] * float(eps)
        return csi * (1.0 + scale * self.pattern(csi))

    def inject(self, *_args, **_kwargs):
        raise RuntimeError(
            'TSBA is differentiable and must be injected after collation via '
            'inject_tensor(); construct PoisonedDataset normally and it will '
            'defer injection automatically.')


def build_tsba_trigger(cfg: dict) -> TSBATrigger:
    return TSBATrigger(
        n_ant=int(cfg.get('n_ant', 3)),
        hidden=int(cfg.get('tsba_hidden', 32)),
    )
