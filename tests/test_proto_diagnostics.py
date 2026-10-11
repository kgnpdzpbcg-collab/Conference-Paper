"""Regression tests: keep P0 unchanged, diagnose instead of claiming correction."""
import math
from types import SimpleNamespace

import pytest
import torch

from ccsae.model import CCSAE, CCSAEConfig
from ccsae.prototype import prototype_logits
from ccsae.proto_diagnostics import diagnose_model


def test_distance_normalization_preserves_P0_and_is_d_times_on_fixed_embeddings():
    torch.manual_seed(7)
    query = torch.randn(6, 12, requires_grad=True)
    centers = torch.randn(3, 12, requires_grad=True)
    historical = prototype_logits(query, centers)
    explicit = prototype_logits(query, centers, distance_normalization="dim")
    unscaled = prototype_logits(query, centers, distance_normalization="none")
    assert torch.allclose(historical, explicit, atol=0, rtol=0)
    assert torch.allclose(unscaled, historical * 12, atol=1e-5, rtol=1e-5)
    assert historical.argmax(1).tolist() == unscaled.argmax(1).tolist()
    assert math.isfinite(unscaled.mean().item())
    with pytest.raises(ValueError, match="distance_normalization"):
        prototype_logits(query, centers, distance_normalization="bad")
    for logits in (historical, unscaled):
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 0, 1, 2]))
        d1, d2 = torch.autograd.grad(loss, [query, centers], retain_graph=True)
        assert d1.isfinite().all() and d2.isfinite().all()


def test_diagnose_is_read_only_and_returns_all_mechanism_checks():
    torch.manual_seed(12)
    cfg = CCSAEConfig(classes=3, sequence_length=64, large_kernel=8,
                      large_channels=8, small_channels=8, feature_dim=12, attention_heads=2)
    model = CCSAE(cfg)
    labels = torch.arange(3).repeat_interleave(5)
    val_labels = torch.arange(3).repeat_interleave(2)
    split = SimpleNamespace(
        train_x=torch.randn(15, 64) + labels.float().unsqueeze(1) * 0.1,
        train_y=labels,
        val_x=torch.randn(6, 64) + val_labels.float().unsqueeze(1) * 0.1,
        val_y=val_labels,
    )
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    result = diagnose_model(model, split, torch.device("cpu"),
                            seed=12, support=2, query=2, repeats=3)
    assert result["validation_only"]
    assert result["num_train"] == 15
    assert result["num_validation"] == 6
    assert result["prototype_dim"] == 24
    assert set(result["one_episode_train_mode_gradients"]) == {"dim", "none"}
    assert result["fixed_embedding_logits"]["dim_vs_none_prediction_differences"] == 0
    assert result["support_stability"]["repeats"] == 3
    assert 0 <= result["stage1_full_K_val_macro_f1"] <= 1
    for key, v in model.state_dict().items():
        assert torch.equal(v, before[key]), "diagnostic must not mutate checkpoint state"
