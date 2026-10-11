"""Tiny end-to-end CPU regression for the V1 + prototype runner."""
from argparse import Namespace
from types import SimpleNamespace

import torch

from ccsae.model import CCSAE, CCSAEConfig
from ccsae.fewshot_train import pretrain_shared, train_one, set_seed


def test_v1_shared_stage1_and_both_heads_one_step(tmp_path):
    set_seed(12)
    cfg = CCSAEConfig(
        classes=3, sequence_length=64, large_kernel=8, large_channels=8,
        small_channels=8, feature_dim=12, attention_heads=2,
    )
    train_y = torch.arange(3).repeat_interleave(5)
    val_y = torch.arange(3).repeat_interleave(2)
    train_x = torch.randn(15, 64) + train_y.unsqueeze(1).float() * 0.25
    val_x = torch.randn(6, 64) + val_y.unsqueeze(1).float() * 0.25
    split = SimpleNamespace(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)
    args = Namespace(
        seed=12, case="skl", k=5, stage1_epochs=1, stage1_batch_size=15,
        lr=0.0022, scl_weight=0.0017, epochs=1, steps_per_epoch=1,
        support=2, query=2, metric="euclidean", temperature=1.0,
    )
    model = CCSAE(cfg)
    stage1_history = pretrain_shared(model, split, args, torch.device("cpu"), tmp_path)
    assert len(stage1_history) == 1
    assert (tmp_path / "shared_stage1_last.pt").exists()

    # Clone the exact same V1 encoder state for both classifier choices.
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    fc = train_one("fc", initial, split, cfg, args, torch.device("cpu"), tmp_path)
    proto = train_one("proto", initial, split, cfg, args, torch.device("cpu"), tmp_path)
    assert 0.0 <= fc["val"]["macro_f1"] <= 1.0
    assert 0.0 <= proto["val"]["macro_f1"] <= 1.0
    assert (tmp_path / "fc_best.pt").exists()
    assert (tmp_path / "proto_best.pt").exists()
    assert (tmp_path / "fc_val_predictions.json").exists()
    assert (tmp_path / "proto_val_predictions.json").exists()
