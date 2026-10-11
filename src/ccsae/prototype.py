"""Differentiable episodic prototypical classification, independent of the backbone."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def sample_episode(
    labels: torch.Tensor,
    num_classes: int,
    support_per_class: int,
    query_per_class: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return disjoint support and query indices; all known classes appear."""
    if support_per_class < 1 or query_per_class < 1:
        raise ValueError("support and query counts must be positive")
    s: list[int] = []
    q: list[int] = []
    y = labels.cpu().numpy()
    for c in range(num_classes):
        pool = np.flatnonzero(y == c)
        n = support_per_class + query_per_class
        if len(pool) < n:
            raise ValueError(f"class {c} has {len(pool)} records; need {n}")
        picked = rng.choice(pool, size=n, replace=False)
        s.extend(picked[:support_per_class].tolist())
        q.extend(picked[support_per_class:].tolist())
    return torch.tensor(s, dtype=torch.long), torch.tensor(q, dtype=torch.long)


def build_prototypes(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if support_features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    centers: list[torch.Tensor] = []
    for c in range(num_classes):
        mask = support_labels == c
        if not bool(mask.any()):
            raise ValueError(f"no support examples of class {c}")
        centers.append(support_features[mask].mean(dim=0))
    return torch.stack(centers, dim=0)


def prototype_logits(
    query_features: torch.Tensor,
    prototypes: torch.Tensor,
    metric: str = "euclidean",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return [Q,C] logits; euclidean is squared distance / embedding width.

    The division controls scale and prevents large 600/1000-D distances from
    destroying the softmax gradient. Cosine is available as a controlled ablation.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if metric == "euclidean":
        sq = torch.cdist(query_features, prototypes, p=2).square()
        return -sq / (query_features.shape[-1] * temperature)
    if metric == "cosine":
        a = F.normalize(query_features, dim=-1)
        b = F.normalize(prototypes, dim=-1)
        return (a @ b.T) / temperature
    raise ValueError("metric must be euclidean or cosine")


def macro_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    if not y_true:
        raise ValueError("empty evaluation set")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(num_classes)),
                                   average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
