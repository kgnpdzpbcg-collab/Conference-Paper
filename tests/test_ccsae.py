import io
import zipfile
from pathlib import Path
import numpy as np
import pytest
import torch
from ccsae.model import CCSAE, CCSAEConfig
from ccsae.complex_layers import ComplexConv1d, ComplexBatchNorm1d
from ccsae.losses import supervised_contrastive, stage1_loss
from ccsae.data import split_data


def test_complex_conv_has_cross_terms():
    conv = ComplexConv1d(1, 1, 1)
    with torch.no_grad():
        conv.wr.weight.fill_(2)
        conv.wi.weight.fill_(3)
        conv.br.zero_()
        conv.bi.zero_()
    r, i = conv((torch.tensor([[[4.]]]), torch.tensor([[[5.]]])))
    assert r.item() == -7
    assert i.item() == 22


def test_complex_bn_backprop():
    bn = ComplexBatchNorm1d(4)
    real = torch.randn(3, 4, 7, requires_grad=True)
    imag = torch.randn(3, 4, 7, requires_grad=True)
    r, i = bn((real, imag))
    (r.square().mean() + i.square().mean()).backward()
    assert torch.isfinite(real.grad).all() and torch.isfinite(imag.grad).all()
    bn.eval()
    rr, ii = bn((real.detach(), imag.detach()))
    assert torch.isfinite(rr).all() and torch.isfinite(ii).all()


def test_model_stage1_and_stage2_gradients():
    torch.manual_seed(0)
    cfg = CCSAEConfig(classes=4, sequence_length=64, large_kernel=8,
                      large_channels=8, small_channels=8, feature_dim=12, attention_heads=2)
    model = CCSAE(cfg)
    x = torch.randn(6, 64)
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    out = model(x, reconstruction=True)
    assert out["features"].shape == (6, 24)
    assert out["reconstruction"].shape == (6, 64)
    assert torch.all(out["reconstruction"] >= 0)
    assert out["logits"].shape == (6, 4)
    loss, _ = stage1_loss(out, x, y, .002)
    loss.backward()
    assert model.encoder[0].conv.wr.weight.grad is not None
    model.zero_grad(set_to_none=True)
    nnloss = torch.nn.functional.cross_entropy(model(x)["logits"], y)
    nnloss.backward()
    assert model.classifier.weight.grad is not None


def test_contrastive_loss_edge_cases():
    x = torch.randn(4, 6, requires_grad=True)
    labels = torch.arange(4)
    l = supervised_contrastive(x, labels)
    assert l.item() == 0
    assert supervised_contrastive(x, torch.tensor([0, 0, 1, 1])).item() > 0
    supervised_contrastive(x, torch.tensor([0, 0, 1, 1])).backward()
    assert torch.isfinite(x.grad).all()


def _make_zip(path: Path):
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(path, "w") as z:
        for case in ("SKLCircuit", "ABLCircuit"):
            for mode, n in (("train", 30), ("test", 12)):
                a = rng.normal(size=(n, 1000)).astype(np.float32)
                y = np.arange(n) % 3
                # Repeat a sample of class 0 in both official files.
                if mode == "test":
                    a[0] = np.zeros(1000, dtype=np.float32)
                else:
                    a[0] = np.zeros(1000, dtype=np.float32)
                labels = np.eye(3, dtype=np.float32)[y]
                for typ, arr in (("data", a), ("label", labels)):
                    text = io.StringIO()
                    np.savetxt(text, arr, delimiter=",")
                    z.writestr(f"{case}_INCI_{mode}_{typ}.csv", text.getvalue())


def test_data_grouped_dedup_and_official_guard(tmp_path):
    p = tmp_path / "Data.zip"
    _make_zip(p)
    s = split_data(p, "skl", seed=1)
    assert s.report["original_test_overlap"] >= 1
    assert not any(s.report["cross_split_duplicate_groups"].values())
    assert sum(s.report["split_sizes"].values()) == 42
    assert s.train.tensors[0].shape[1] == 1000
    with pytest.raises(ValueError, match="duplicates"):
        split_data(p, "skl", mode="official")