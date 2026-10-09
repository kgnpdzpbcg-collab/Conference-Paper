"""Complex-valued building blocks implemented as (real, imaginary) float tensors.

Complex convolution and affine transforms follow Gao et al. Eqs. (1),(2),(12).
Complex BN uses channelwise whitening of the 2x2 real/imag covariance (Eqs. 5-8).
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

ComplexPair = tuple[torch.Tensor, torch.Tensor]


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.wr = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.wi = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.br = nn.Parameter(torch.zeros(out_channels))
        self.bi = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: ComplexPair) -> ComplexPair:
        xr, xi = x
        real = self.wr(xr) - self.wi(xi) + self.br[None, :, None]
        imag = self.wi(xr) + self.wr(xi) + self.bi[None, :, None]
        return real, imag


class ComplexLinear(nn.Module):
    def __init__(self, inp: int, out: int):
        super().__init__()
        self.wr = nn.Linear(inp, out)
        self.wi = nn.Linear(inp, out)

    def forward(self, x: ComplexPair) -> ComplexPair:
        xr, xi = x
        # Two Linear biases implement (bR, bI), not double counted.
        return self.wr(xr) - F.linear(xi, self.wi.weight), self.wi(xr) + F.linear(xi, self.wr.weight)


class ComplexBatchNorm1d(nn.Module):
    """Full 2x2 covariance whitening across batch and time for each channel.

    Uses running means/covariances for evaluation. Eigh is performed in float32
    even under AMP for stability. Affine mixing is symmetric, initialized I/sqrt(2).
    """
    def __init__(self, channels: int, eps: float = 1e-4, momentum: float = 0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.register_buffer("running_mean", torch.zeros(channels, 2))
        self.register_buffer("running_cov", torch.eye(2).repeat(channels, 1, 1))
        self.gamma_rr = nn.Parameter(torch.full((channels,), 2 ** -0.5))
        self.gamma_ii = nn.Parameter(torch.full((channels,), 2 ** -0.5))
        self.gamma_ri = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels, 2))

    def forward(self, x: ComplexPair) -> ComplexPair:
        xr, xi = x
        original_dtype = xr.dtype
        z = torch.stack([xr.float(), xi.float()], dim=-1)  # B,C,T,2
        if self.training:
            mean = z.mean(dim=(0, 2))
            centered = z - mean[None, :, None, :]
            cov = torch.einsum("bcti,bctj->cij", centered, centered) / (z.shape[0] * z.shape[2])
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_cov.lerp_(cov.detach(), self.momentum)
        else:
            mean, cov = self.running_mean, self.running_cov
            centered = z - mean[None, :, None, :]
        eye = torch.eye(2, device=z.device, dtype=z.dtype)[None]
        eigvals, eigvecs = torch.linalg.eigh(cov + self.eps * eye)
        invsqrt = eigvecs @ torch.diag_embed(eigvals.clamp_min(self.eps).rsqrt()) @ eigvecs.transpose(-1, -2)
        z = torch.einsum("cij,bctj->bcti", invsqrt, centered)
        gamma = torch.stack((torch.stack((self.gamma_rr, self.gamma_ri), -1),
                             torch.stack((self.gamma_ri, self.gamma_ii), -1)), -2)
        z = torch.einsum("cij,bctj->bcti", gamma.float(), z) + self.beta[None, :, None, :]
        return z[..., 0].to(original_dtype), z[..., 1].to(original_dtype)


class ACON(nn.Module):
    """Learnable activate-or-not (Gao Eq. 3)."""
    def __init__(self, channels: int):
        super().__init__()
        self.p1 = nn.Parameter(torch.ones(1, channels, 1))
        self.p2 = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.p1 - self.p2
        return delta * x * torch.sigmoid(self.beta * delta * x) + self.p2 * x


class ComplexACON(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.r, self.i = ACON(channels), ACON(channels)

    def forward(self, x: ComplexPair) -> ComplexPair:
        return self.r(x[0]), self.i(x[1])


class ComplexBlock(nn.Module):
    def __init__(self, inp: int, out: int, kernel: int, stride: int = 1):
        super().__init__()
        self.conv = ComplexConv1d(inp, out, kernel, stride)
        self.bn = ComplexBatchNorm1d(out)
        self.act = ComplexACON(out)

    def forward(self, x: ComplexPair) -> ComplexPair:
        return self.act(self.bn(self.conv(x)))