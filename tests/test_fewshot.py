"""Focused regression tests for the limited-label experiment."""
from __future__ import annotations
import io
import zipfile
import numpy as np
import pytest
import torch

from ccsae.fewshot_data import make_fewshot_split
from ccsae.prototype import (
    build_prototypes, macro_metrics, prototype_logits, sample_episode,
)


def _write_dataset(path):
    rng = np.random.default_rng(77)
    with zipfile.ZipFile(path, "w") as zf:
        for prefix in ("SKLCircuit", "ABLCircuit"):
            for split, per_class in (("train", 20), ("test", 10)):
                xs, ys = [], []
                for c in range(3):
                    arr = rng.normal(loc=c, scale=1.0, size=(per_class, 1000)).astype(np.float32)
                    if split == "test":
                        # Exact duplicate of first TRAIN row in the same class:
                        # duplicate source is deterministic by resetting in test below.
                        arr[-1] = np.float32(c + 100)
                    else:
                        arr[-1] = np.float32(c + 100)
                    xs.append(arr)
                    ys.extend([c] * per_class)
                x = np.concatenate(xs)
                y = np.eye(3, dtype=np.float32)[ys]
                for typ, a in (("data", x), ("label", y)):
                    buf = io.StringIO()
                    np.savetxt(buf, a, delimiter=",", fmt="%.9g")
                    zf.writestr(f"{prefix}_INCI_{split}_{typ}.csv", buf.getvalue())


def test_actual_dedup_and_k_only_scaler(tmp_path):
    file = tmp_path / "Data.zip"
    _write_dataset(file)
    r = make_fewshot_split(file, "skl", k=5, val_per_class=2,
                           split_seed=1234, sample_seed=42)
    assert r.audit["original_rows"] == 90
    assert r.audit["unique_rows"] == 87
    assert r.audit["duplicate_rows_removed"] == 3
    assert len(r.train_x) == 15
    assert len(r.val_x) == 6
    assert r.train_y.bincount(minlength=3).tolist() == [5, 5, 5]
    assert r.val_y.bincount(minlength=3).tolist() == [2, 2, 2]
    assert abs(r.train_x.mean().item()) < 1e-5
    assert abs(r.train_x.std(unbiased=False).item() - 1.0) < 1e-5
    hashes_train = sum(r.audit["selected_train_hashes_by_class"].values(), [])
    hashes_val = sum(r.audit["selected_val_hashes_by_class"].values(), [])
    assert not set(hashes_train) & set(hashes_val)
    same = make_fewshot_split(file, "skl", k=5, val_per_class=2,
                              split_seed=1234, sample_seed=42)
    assert r.audit["selected_train_hashes_by_class"] == same.audit["selected_train_hashes_by_class"]
    larger = make_fewshot_split(file, "skl", k=8, val_per_class=2,
                                split_seed=1234, sample_seed=42)
    for c in ("0", "1", "2"):
        assert r.audit["selected_train_hashes_by_class"][c] == (
            larger.audit["selected_train_hashes_by_class"][c][:5]
        )


def test_episode_disjoint_balanced_and_reproducible():
    y = torch.arange(3).repeat_interleave(8)
    first = sample_episode(y, 3, 3, 2, np.random.default_rng(51))
    second = sample_episode(y, 3, 3, 2, np.random.default_rng(51))
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    s, q = first
    assert not set(s.tolist()) & set(q.tolist())
    assert y[s].bincount(minlength=3).tolist() == [3, 3, 3]
    assert y[q].bincount(minlength=3).tolist() == [2, 2, 2]
    with pytest.raises(ValueError, match="need"):
        sample_episode(y, 3, 7, 2, np.random.default_rng(0))


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_prototype_end_to_end_gradient(metric):
    torch.manual_seed(0)
    xs = torch.randn(9, 12, requires_grad=True)
    xq = torch.randn(6, 12, requires_grad=True)
    ys = torch.arange(3).repeat_interleave(3)
    yq = torch.arange(3).repeat_interleave(2)
    prototypes = build_prototypes(xs, ys, 3)
    logits = prototype_logits(xq, prototypes, metric=metric, temperature=1.0)
    assert logits.shape == (6, 3)
    loss = torch.nn.functional.cross_entropy(logits, yq)
    loss.backward()
    assert torch.isfinite(xs.grad).all() and torch.isfinite(xq.grad).all()
    assert xs.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="no support"):
        build_prototypes(xs, ys, 4)
    with pytest.raises(ValueError, match="metric"):
        prototype_logits(xq.detach(), prototypes.detach(), metric="invalid")


def test_macro_metrics():
    result = macro_metrics([0, 1, 2], [0, 1, 0], 3)
    assert 0 <= result["macro_f1"] <= 1
    assert result["accuracy"] == pytest.approx(2 / 3)
