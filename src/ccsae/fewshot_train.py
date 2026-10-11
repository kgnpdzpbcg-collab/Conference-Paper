"""Quick limited-label comparison: CCSAE V1 + FC vs CCSAE V1 + Proto.

Only training subsets and a fixed 5-per-class validation subset are consumed.
No test-set inference is implemented here by design.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .fewshot_data import make_fewshot_split
from .losses import stage1_loss
from .model import CCSAE, CCSAEConfig
from .prototype import build_prototypes, macro_metrics, prototype_logits, sample_episode


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(case: str, tiny: bool) -> CCSAEConfig:
    cfg = CCSAEConfig.for_case(case)
    if tiny:
        return CCSAEConfig(classes=cfg.classes, sequence_length=1000,
                           large_kernel=16, large_channels=8,
                           small_channels=8, feature_dim=32, attention_heads=2)
    return cfg


def _feature_params(model: CCSAE, include_classifier: bool) -> list[nn.Parameter]:
    modules: list[nn.Module] = [
        model.imag_init, model.encoder, model.attention, model.projection
    ]
    if include_classifier:
        modules.append(model.classifier)
    return [p for module in modules for p in module.parameters()]


@torch.no_grad()
def _forward_features(
    model: CCSAE, x: torch.Tensor, device: torch.device, batch_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    feats, logits = [], []
    for chunk in x.split(batch_size):
        out = model(chunk.to(device))
        feats.append(out["features"])
        logits.append(out["logits"])
    return torch.cat(feats, dim=0), torch.cat(logits, dim=0)


@torch.no_grad()
def validate(
    model: CCSAE,
    method: str,
    split,
    cfg: CCSAEConfig,
    device: torch.device,
    metric: str,
    temperature: float,
    distance_normalization: str = "dim",
) -> tuple[dict[str, float], list[int]]:
    model.eval()
    val_features, fc_logits = _forward_features(model, split.val_x, device)
    if method == "fc":
        scores = fc_logits
    elif method == "proto":
        train_features, _ = _forward_features(model, split.train_x, device)
        centers = build_prototypes(train_features, split.train_y.to(device), cfg.classes)
        scores = prototype_logits(val_features, centers, metric=metric, temperature=temperature,
                                  distance_normalization=distance_normalization)
    else:
        raise ValueError(method)
    pred = scores.argmax(dim=1).cpu().tolist()
    return macro_metrics(split.val_y.tolist(), pred, cfg.classes), pred


def pretrain_shared(
    model: CCSAE,
    split,
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
) -> list[dict]:
    """Stage 1 on K-only training data; save last epoch, no validation-based tuning."""
    if args.stage1_epochs == 0:
        return []
    dataset = TensorDataset(split.train_x, split.train_y)
    loader = DataLoader(dataset, batch_size=args.stage1_batch_size, shuffle=True,
                        drop_last=False)
    params = _feature_params(model, include_classifier=False) + list(model.decoder.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    history: list[dict] = []
    model.train()
    for epoch in range(1, args.stage1_epochs + 1):
        totals = {"total": 0.0, "reconstruction": 0.0, "scl": 0.0}
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            result = model(x, reconstruction=True, reconstruction_output="signed_real")
            loss, parts = stage1_loss(
                result, x, y, scl_weight=args.scl_weight, temperature=0.1,
                reconstruction_target="signed_real"
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite stage1 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            totals["total"] += loss.detach().item() * len(y)
            for key in ("reconstruction", "scl"):
                totals[key] += parts[key] * len(y)
        row = {"epoch": epoch, **{k: v / len(dataset) for k, v in totals.items()}}
        history.append(row)
        print(f"[stage1] epoch={epoch} total={row['total']:.5f} "
              f"recon={row['reconstruction']:.5f} scl={row['scl']:.5f}", flush=True)
    torch.save({"state_dict": model.state_dict(), "source": "CCSAE V1 signed_real",
                "stage1_epochs": args.stage1_epochs}, output / "shared_stage1_last.pt")
    return history


def train_one(
    method: str,
    initial_state: dict[str, torch.Tensor],
    split,
    cfg: CCSAEConfig,
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
) -> dict:
    """Same initialization and same episode index stream for both methods."""
    set_seed(args.seed)
    model = CCSAE(cfg).to(device)
    model.load_state_dict(initial_state)
    model.train()
    params = _feature_params(model, include_classifier=(method == "fc"))
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    rng = np.random.default_rng(args.seed + 901)
    labels = split.train_y
    best_f1, best_epoch = -1.0, None
    history: list[dict] = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for _ in range(args.steps_per_epoch):
            s_idx, q_idx = sample_episode(
                labels, cfg.classes, args.support, args.query, rng
            )
            indices = torch.cat((s_idx, q_idx))
            all_x = split.train_x[indices].to(device)
            all_y = split.train_y[indices].to(device)
            n_support = len(s_idx)
            out = model(all_x)
            if method == "fc":
                # Same examples and one optimizer step as the Proto method.
                loss = nn.functional.cross_entropy(out["logits"], all_y)
            else:
                z_s, z_q = out["features"][:n_support], out["features"][n_support:]
                prototypes = build_prototypes(z_s, all_y[:n_support], cfg.classes)
                scores = prototype_logits(
                    z_q, prototypes, metric=args.metric, temperature=args.temperature,
                    distance_normalization=getattr(args, "distance_normalization", "dim")
                )
                loss = nn.functional.cross_entropy(scores, all_y[n_support:])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {method} loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            train_loss += float(loss.detach())

        metrics, _ = validate(
            model, method, split, cfg, device, args.metric, args.temperature,
            distance_normalization=getattr(args, "distance_normalization", "dim")
        )
        record = {"epoch": epoch, "train_loss": train_loss / args.steps_per_epoch, **metrics}
        history.append(record)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(), "epoch": epoch,
                "method": method, "case": args.case, "k": args.k,
                "metric": args.metric, "temperature": args.temperature,
                "distance_normalization": getattr(args, "distance_normalization", "dim")
            }, output / f"{method}_best.pt")
        print(f"[{method}] epoch={epoch} train={record['train_loss']:.5f} "
              f"val_F1={metrics['macro_f1']:.4f} best={best_f1:.4f}", flush=True)

    elapsed = time.perf_counter() - start
    state = torch.load(output / f"{method}_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    final_metrics, val_predictions = validate(
        model, method, split, cfg, device, args.metric, args.temperature,
        distance_normalization=getattr(args, "distance_normalization", "dim")
    )
    (output / f"{method}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output / f"{method}_val_predictions.json").write_text(
        json.dumps({"truth": split.val_y.tolist(), "predicted": val_predictions}, indent=2),
        encoding="utf-8"
    )
    return {
        "method": method, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "val": final_metrics, "stage2_seconds": elapsed, "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "val_predictions_file": f"{method}_val_predictions.json",
        "checkpoint": f"{method}_best.pt",
    }


def run(args: argparse.Namespace) -> dict:
    if args.k < args.support + args.query:
        raise ValueError("K must allow disjoint support and query samples")
    if args.epochs < 1 or args.steps_per_epoch < 1 or args.stage1_epochs < 0:
        raise ValueError("invalid training budget")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    distance_normalization = getattr(args, "distance_normalization", "dim")
    if distance_normalization not in {"dim", "none"}:
        raise ValueError("distance_normalization must be dim or none")
    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    split = make_fewshot_split(
        args.data, args.case, k=args.k, val_per_class=args.val_per_class,
        split_seed=args.split_seed,
        sample_seed=args.sample_seed if args.sample_seed is not None else args.seed,
        val_seed=args.val_seed,
    )
    cfg = make_config(args.case, args.tiny)
    run_label = (f"k{args.k}_sample{args.sample_seed if args.sample_seed is not None else args.seed}"
                 f"_train{args.seed}" +
                 ("_distnone" if distance_normalization == "none" else ""))
    save_dir = Path(args.output) / args.case / run_label
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "split_audit.json").write_text(
        json.dumps(split.audit, indent=2), encoding="utf-8"
    )

    if args.lr is None:
        args.lr = 0.0022 if args.case == "skl" else 0.0102
    if args.scl_weight is None:
        args.scl_weight = 0.0017 if args.case == "skl" else 0.0059

    (save_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    base = CCSAE(cfg).to(device)
    stage1_history = pretrain_shared(base, split, args, device, save_dir)
    initial_state = deepcopy(base.state_dict())
    del base
    (save_dir / "shared_stage1_history.json").write_text(
        json.dumps(stage1_history, indent=2), encoding="utf-8"
    )

    methods = ["fc", "proto"] if args.mode == "both" else [args.mode]
    results = {}
    for method in methods:
        results[method] = train_one(
            method, initial_state, split, cfg, args, device, save_dir
        )
    report = {
        "experiment": "CCSAE V1 limited-label FC vs episodic Proto (validation only)",
        "test_evaluated": False,
        "data": split.audit,
        "config": vars(args),
        "models": results,
        "paired_val_macro_f1_delta_proto_minus_fc": (
            results["proto"]["val"]["macro_f1"] - results["fc"]["val"]["macro_f1"]
            if args.mode == "both" else None
        ),
    }
    (save_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps({
        "case": args.case, "k": args.k,
        "val_macro_f1": {m: r["val"]["macro_f1"] for m, r in results.items()},
        "output": str(save_dir), "test_evaluated": False
    }, indent=2), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="K-shot CCSAE V1 FC/Proto validation comparison")
    p.add_argument("--data", default="data/Data.zip")
    p.add_argument("--case", choices=["skl", "abl"], required=True)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--val-per-class", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=20261009)
    p.add_argument("--val-seed", type=int, default=20261009)
    p.add_argument("--sample-seed", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["both", "fc", "proto"], default="both")
    p.add_argument("--stage1-epochs", type=int, default=10)
    p.add_argument("--stage1-batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40, help="stage2 epochs")
    p.add_argument("--steps-per-epoch", type=int, default=10)
    p.add_argument("--support", type=int, default=3)
    p.add_argument("--query", type=int, default=2)
    p.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--distance-normalization", choices=["dim", "none"], default="dim",
                   help="Euclidean squared-distance scaling: dim=historical P0, none=scale ablation")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--scl-weight", type=float, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="outputs/fewshot_proto")
    p.add_argument("--tiny", action="store_true", help="tiny shape smoke-test, not paper setting")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
