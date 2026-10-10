"""Two-stage CCSAE training and reproducible CLI.

Stage 1: reconstruct + supervised contrast (feature extraction and enhancement).
Stage 2: finetune encoder + classifier using cross entropy, decoder unused.
"""
from __future__ import annotations
import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from .data import split_data
from .model import CCSAE, CCSAEConfig
from .losses import stage1_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: CCSAE, loader: DataLoader, device: torch.device,
             feature_objective: bool = False, scl_weight: float = 0.0017,
             temperature: float = 0.1, reconstruction_target: str = "paper_signed") -> dict:
    model.eval()
    losses, total = 0.0, 0
    component_sums = {"reconstruction": 0.0, "scl": 0.0}
    labels, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output_mode = "signed_real" if reconstruction_target == "signed_real" else "magnitude"
            out = model(x, reconstruction=feature_objective, reconstruction_output=output_mode)
            if feature_objective:
                loss, parts = stage1_loss(out, x, y, scl_weight, temperature, reconstruction_target)
                for name in component_sums:
                    component_sums[name] += parts[name] * x.shape[0]
            else:
                loss = nn.functional.cross_entropy(out["logits"], y)
            losses += float(loss) * x.shape[0]
            total += x.shape[0]
            labels.extend(y.cpu().tolist())
            preds.extend(out["logits"].argmax(dim=1).cpu().tolist())
    metrics = {"loss": losses / max(total, 1)}
    if feature_objective:
        metrics.update({f"{name}_loss": v / max(total, 1) for name, v in component_sums.items()})
    if not feature_objective:
        metrics.update({"accuracy": accuracy_score(labels, preds),
                        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0)})
    return metrics


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    splits = split_data(args.data, args.case, seed=args.split_seed, mode=args.split,
                        allow_official_overlap=args.allow_official_overlap)
    outdir = Path(args.output) / args.case / f"seed_{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "data_audit.json").write_text(json.dumps(splits.report, indent=2), encoding="utf-8")
    if any(splits.report["cross_split_duplicate_groups"].values()) and args.split == "grouped":
        raise AssertionError("grouped split must have no exact cross-split duplicates")
    cfg = CCSAEConfig.for_case(args.case)
    if args.tiny:
        cfg = CCSAEConfig(classes=cfg.classes, sequence_length=1000, large_kernel=16,
                          large_channels=8, small_channels=8, feature_dim=32, attention_heads=2)
    model = CCSAE(cfg).to(device)
    train_loader = DataLoader(splits.train, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(splits.val, batch_size=args.batch_size, shuffle=False)
    history = []
    # Stage 1: encoder, attention, decoder; excludes classifier parameters.
    first_params = list(model.imag_init.parameters()) + list(model.encoder.parameters()) + \
                   list(model.attention.parameters()) + list(model.projection.parameters()) + \
                   list(model.decoder.parameters())
    optimizer = torch.optim.SGD(first_params, lr=args.lr, momentum=0.9)
    best_stage1 = float("inf")
    for epoch in range(1, args.feature_epochs + 1):
        model.train()
        accum = 0.0
        component_sums = {"reconstruction": 0.0, "scl": 0.0}
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            output_mode = "signed_real" if args.reconstruction_target == "signed_real" else "magnitude"
            out = model(x, reconstruction=True, reconstruction_output=output_mode)
            loss, parts = stage1_loss(out, x, y, (args.scl_weight if args.scl_weight is not None else (0.0017 if args.case == "skl" else 0.0059)),
                                  args.temperature, args.reconstruction_target)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite stage1 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm_(first_params, args.grad_clip)
            optimizer.step()
            accum += float(loss.detach()) * x.shape[0]
            for name in component_sums:
                component_sums[name] += parts[name] * x.shape[0]
        val = evaluate(model, val_loader, device, feature_objective=True,
                       scl_weight=(args.scl_weight if args.scl_weight is not None else (0.0017 if args.case == "skl" else 0.0059)),
                       temperature=args.temperature, reconstruction_target=args.reconstruction_target)
        history.append({"stage": 1, "epoch": epoch, "train_loss": accum / len(splits.train),
                        **{f"train_{name}_loss": v / len(splits.train) for name, v in component_sums.items()}, **val})
        print(f"stage1 epoch={epoch} train={accum / len(splits.train):.5f} val={val['loss']:.5f}", flush=True)
        if val["loss"] < best_stage1:
            best_stage1 = val["loss"]
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "epoch": epoch}, outdir / "stage1_best.pt")
    if args.feature_epochs > 0:
        model.load_state_dict(torch.load(outdir / "stage1_best.pt", map_location=device, weights_only=True)["state_dict"])
    # Stage 2: encoder + attention + projection + classifier; decoder frozen/excluded.
    second_params = list(model.imag_init.parameters()) + list(model.encoder.parameters()) + \
                    list(model.attention.parameters()) + list(model.projection.parameters()) + \
                    list(model.classifier.parameters())
    optimizer = torch.optim.SGD(second_params, lr=args.lr, momentum=0.9)
    best_f1 = -1.0
    best_epoch = None
    for epoch in range(1, args.classification_epochs + 1):
        model.train()
        accum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)["logits"]
            loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite stage2 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm_(second_params, args.grad_clip)
            optimizer.step()
            accum += float(loss.detach()) * x.shape[0]
        val = evaluate(model, val_loader, device)
        history.append({"stage": 2, "epoch": epoch, "train_loss": accum / len(splits.train), **val})
        print(f"stage2 epoch={epoch} train={accum / len(splits.train):.5f} val_F1={val['macro_f1']:.5f}", flush=True)
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch = val["macro_f1"], epoch
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "epoch": epoch,
                        "data_audit": splits.report}, outdir / "stage2_best.pt")
    summary = {"case": args.case, "seed": args.seed, "best_val_macro_f1": best_f1,
               "best_classification_epoch": best_epoch, "test_evaluated": False,
               "protocol": splits.report, "args": vars(args)}
    if args.evaluate_test:
        if args.classification_epochs < 1:
            raise ValueError("test evaluation requires classification training")
        model.load_state_dict(torch.load(outdir / "stage2_best.pt", map_location=device, weights_only=True)["state_dict"])
        test_loader = DataLoader(splits.test, batch_size=args.batch_size, shuffle=False)
        metrics = evaluate(model, test_loader, device)
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x.to(device))["logits"]
                ps.extend(logits.argmax(dim=1).cpu().tolist())
                ys.extend(y.tolist())
        labels = list(range(cfg.classes))
        metrics["confusion_matrix"] = confusion_matrix(ys, ps, labels=labels).tolist()
        metrics["per_class"] = classification_report(ys, ps, labels=labels, output_dict=True, zero_division=0)
        summary["test"] = metrics
        summary["test_evaluated"] = True
    (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Independent CCSAE 2-stage reproduction")
    p.add_argument("--data", type=str, required=True, help="Data.zip or directory of CSV files")
    p.add_argument("--case", choices=["skl", "abl"], required=True)
    p.add_argument("--split", choices=["grouped", "official"], default="grouped")
    p.add_argument("--split-seed", type=int, default=20261009,
                   help="Partition seed independent of model initialization --seed")
    p.add_argument("--allow-official-overlap", action="store_true", help="Unsafe, for comparison only")
    p.add_argument("--output", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--feature-epochs", type=int, default=100)
    p.add_argument("--classification-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--scl-weight", type=float, default=None)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--reconstruction-target", choices=["signed_real", "magnitude", "paper_signed"], default="signed_real")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--tiny", action="store_true", help="Fast smoke-test network, NOT paper settings")
    args = p.parse_args()
    if args.lr is None:
        args.lr = 0.0022 if args.case == "skl" else 0.0102
    result = run(args)
    print(json.dumps({"best_val_macro_f1": result["best_val_macro_f1"],
                      "best_epoch": result["best_classification_epoch"],
                      "test_evaluated": result["test_evaluated"]}, indent=2))


if __name__ == "__main__":
    main()