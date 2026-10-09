"""Eq. 21-24 loss implementations, with documented paper ambiguities."""
import torch
from torch import nn
from torch.nn import functional as F


def supervised_contrastive(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Standard *minimized* supervised contrastive loss, exclude self comparisons.

    Published Eq. (22) omits the conventional minus sign. We restore it to agree
    with the authors' stated goal of pulling positives closer.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    features = F.normalize(features, dim=1)
    logits = (features @ features.T) / temperature
    n = features.shape[0]
    if n <= 1:
        return features.sum() * 0
    eye = torch.eye(n, device=features.device, dtype=torch.bool)
    positives = (labels[:, None] == labels[None, :]) & ~eye
    scores = logits.masked_fill(eye, -torch.inf)
    log_prob = scores - torch.logsumexp(scores, dim=1, keepdim=True)
    counts = positives.sum(dim=1)
    valid = counts > 0
    if not valid.any():
        return features.sum() * 0
    losses = -(torch.where(positives, log_prob, 0.0).sum(dim=1)[valid] / counts[valid])
    return losses.mean()


def stage1_loss(outputs: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor,
                scl_weight: float, temperature: float = 0.1,
                reconstruction_target: str = "paper_signed") -> tuple[torch.Tensor, dict[str, float]]:
    if reconstruction_target not in {"paper_signed", "magnitude"}:
        raise ValueError("reconstruction_target must be paper_signed or magnitude")
    # Paper Eq.13 is |R| >= 0 yet raw responses (Fig.5) can be negative.
    # paper_signed matches printed Eqs.13 & 21; magnitude is a documented alternative.
    target = x if reconstruction_target == "paper_signed" else x.abs()
    recon = F.mse_loss(outputs["reconstruction"], target)
    scl = supervised_contrastive(outputs["features"], y, temperature)
    total = recon + scl_weight * scl
    return total, {"reconstruction": float(recon.detach()), "scl": float(scl.detach())}