"""CCSAE structure inspired by Gao et al., IEEE TIE 2024.

Exact strides, channel schedule, attention heads and decoder layout are not
reported in the paper; choices here are marked as independent implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .complex_layers import ComplexPair, ComplexBlock, ComplexLinear


class RealAttentionLocal(nn.Module):
    """Real-valued CSA component: global MHA plus local k=3 convolution (Eq. 18).

    This is an explicit approximation of Fig. 2's local shift/aggregation.
    """
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        assert channels % heads == 0
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.local = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1, groups=channels),
                                   nn.Conv1d(channels, channels, 1))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_out, _ = self.attn(x.transpose(1, 2), x.transpose(1, 2), x.transpose(1, 2), need_weights=False)
        return self.alpha * global_out.transpose(1, 2) + self.beta * self.local(x)


class ComplexSelfAttention(nn.Module):
    """Complex extension CSA(X) = (CSA_R(Xr)-CSA_I(Xi)) + i(CSA_I(Xr)+CSA_R(Xi)), Eq. 19."""
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.real = RealAttentionLocal(channels, heads)
        self.imag = RealAttentionLocal(channels, heads)

    def forward(self, x: ComplexPair) -> ComplexPair:
        xr, xi = x
        return self.real(xr) - self.imag(xi), self.imag(xr) + self.real(xi)


@dataclass(frozen=True)
class CCSAEConfig:
    classes: int = 13
    sequence_length: int = 1000
    large_kernel: int = 64
    large_channels: int = 20
    small_channels: int = 20
    feature_dim: int = 500
    attention_heads: int = 4
    input_imag_kernel: int = 3

    @classmethod
    def for_case(cls, case: str) -> "CCSAEConfig":
        if case == "skl":
            return cls(classes=13, large_kernel=64, large_channels=20, small_channels=20, feature_dim=500)
        if case == "abl":
            return cls(classes=11, large_kernel=128, large_channels=50, small_channels=20, feature_dim=300)
        raise ValueError("case must be skl or abl")


class CCSAE(nn.Module):
    def __init__(self, cfg: CCSAEConfig):
        super().__init__()
        self.cfg = cfg
        self.imag_init = nn.Sequential(nn.BatchNorm1d(1),
                                      nn.Conv1d(1, 1, cfg.input_imag_kernel, padding=cfg.input_imag_kernel // 2))
        self.encoder = nn.Sequential(
            ComplexBlock(1, cfg.large_channels, cfg.large_kernel, stride=4),
            ComplexBlock(cfg.large_channels, cfg.small_channels, 5, stride=4),
            ComplexBlock(cfg.small_channels, cfg.small_channels, 5, stride=2),
        )
        self.attention = ComplexSelfAttention(cfg.small_channels, cfg.attention_heads)
        # Determine flattened feature length from convolution sizes without updating BN state.
        def conv_length(length: int, kernel: int, stride: int) -> int:
            return (length + 2 * (kernel // 2) - kernel) // stride + 1
        reduced_length = conv_length(cfg.sequence_length, cfg.large_kernel, 4)
        reduced_length = conv_length(reduced_length, 5, 4)
        reduced_length = conv_length(reduced_length, 5, 2)
        flatten_dim = cfg.small_channels * reduced_length
        self.projection = ComplexLinear(flatten_dim, cfg.feature_dim)
        self.decoder = ComplexLinear(cfg.feature_dim, cfg.sequence_length)
        self.classifier = nn.Linear(cfg.feature_dim * 2, cfg.classes)

    def encode(self, x: torch.Tensor) -> ComplexPair:
        if x.ndim != 2 or x.shape[1] != self.cfg.sequence_length:
            raise ValueError(f"expected [batch, {self.cfg.sequence_length}], got {tuple(x.shape)}")
        real = x.unsqueeze(1)
        imag = self.imag_init(real)
        h = self.encoder((real, imag))
        h = self.attention(h)
        h = (h[0].flatten(1), h[1].flatten(1))
        return self.projection(h)

    @staticmethod
    def feature_vector(h: ComplexPair) -> torch.Tensor:
        return torch.cat(h, dim=1)

    def reconstruct(self, h: ComplexPair, output_mode: str = "magnitude") -> torch.Tensor:
        """V1 signed_real is a documented adaptation, not an official paper implementation."""
        rr, ri = self.decoder(h)
        if output_mode == "signed_real":
            return rr  # Signed output from the existing complex decoder: no added parameters.
        if output_mode == "magnitude":
            return torch.sqrt(rr.square() + ri.square() + 1e-8)
        raise ValueError(f"unknown reconstruction output mode: {output_mode}")

    def forward(self, x: torch.Tensor, reconstruction: bool = False,
                reconstruction_output: str = "magnitude") -> dict[str, torch.Tensor]:
        h = self.encode(x)
        features = self.feature_vector(h)
        out = {"features": features, "logits": self.classifier(features)}
        if reconstruction:
            out["reconstruction"] = self.reconstruct(h, output_mode=reconstruction_output)
        return out