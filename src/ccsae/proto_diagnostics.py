"""Read-only diagnostics for an existing CCSAE K-shot run.

Requires locally saved shared_stage1_last.pt and original Data.zip. Does not
train, change existing checkpoints, read the held-out Test set, or make claims
about held-out test performance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .fewshot_data import make_fewshot_split
from .fewshot_train import _forward_features, make_config, set_seed
from .model import CCSAE
from .prototype import build_prototypes, macro_metrics, prototype_logits, sample_episode


def _stats(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def _logit_stats(logits: torch.Tensor) -> dict[str, float]:
    logits = logits.detach().float()
    gap = torch.topk(logits, k=2, dim=1).values
    return {
        "mean_class_logit_spread": float(logits.std(dim=1, unbiased=False).mean()),
        "mean_top1_minus_top2_logit": float((gap[:, 0] - gap[:, 1]).mean()),
        "mean_max_softmax_probability": float(logits.softmax(dim=1).max(dim=1).values.mean()),
    }


def diagnose_model(
    model: CCSAE,
    split,
    device: torch.device,
    seed: int = 42,
    support: int = 3,
    query: int = 2,
    repeats: int = 32,
) -> dict:
    """Use K Train for all prototype construction and a fixed labelled Val for descriptive metrics."""
    if repeats < 2 or support < 1 or query < 1:
        raise ValueError("repeats must be >=2 and support/query >=1")
    classes = model.cfg.classes
    y_train = split.train_y.to(device)
    y_val = split.val_y.tolist()
    if int(y_train.bincount(minlength=classes).min()) < support + query:
        raise ValueError("insufficient K for disjoint Support/Query")
    model.eval()
    with torch.no_grad():
        z_train, _ = _forward_features(model, split.train_x, device)
        z_val, _ = _forward_features(model, split.val_x, device)
        centers = build_prototypes(z_train, y_train, classes)
        dim_logits = prototype_logits(z_val, centers, distance_normalization="dim")
        raw_logits = prototype_logits(z_val, centers, distance_normalization="none")
        pred_dim = dim_logits.argmax(dim=1).cpu().tolist()
        pred_raw = raw_logits.argmax(dim=1).cpu().tolist()
        # Fixed embeddings: positive logit rescaling must preserve argmax in exact arithmetic.
        reference64 = -torch.cdist(z_val.double(), centers.double()).square()
        reference_pred = reference64.argmax(dim=1).cpu().tolist()
        separation = torch.cdist(centers, centers, p=2)
        separation.fill_diagonal_(float("inf"))
        nearest_center = separation.min(dim=1).values
        within_rms = torch.stack([
            ((z_train[y_train == c] - centers[c]).square().sum(dim=1).mean()).sqrt()
            for c in range(classes)
        ])
        ratio = within_rms / nearest_center.clamp_min(1e-12)
        # Classify the unchanged Val under 3-support random subsampling without retraining.
        rng = np.random.default_rng(seed + 114)
        per_draw_f1, drifts = [], []
        labels_np = split.train_y.numpy()
        for _ in range(repeats):
            indices = []
            for c in range(classes):
                pool = np.flatnonzero(labels_np == c)
                indices.extend(rng.choice(pool, support, replace=False).tolist())
            ids = torch.tensor(indices, device=device, dtype=torch.long)
            sampled = build_prototypes(z_train[ids], y_train[ids], classes)
            drifts.extend(
                (sampled - centers).norm(dim=1).div(nearest_center.clamp_min(1e-12)).cpu().tolist()
            )
            preds = prototype_logits(z_val, sampled).argmax(dim=1).cpu().tolist()
            per_draw_f1.append(macro_metrics(y_val, preds, classes)["macro_f1"])
    base_state = deepcopy(model.state_dict())
    rng = np.random.default_rng(seed + 901)
    s_idx, q_idx = sample_episode(split.train_y, classes, support, query, rng)
    batch_ids = torch.cat([s_idx, q_idx])
    x = split.train_x[batch_ids].to(device)
    y = split.train_y[batch_ids].to(device)
    ns = len(s_idx)

    grad_report = {}
    for norm in ("dim", "none"):
        model.load_state_dict(base_state)
        model.zero_grad(set_to_none=True)
        model.train()
        z = model(x)["features"]
        protos = build_prototypes(z[:ns], y[:ns], classes)
        logits = prototype_logits(z[ns:], protos, distance_normalization=norm)
        loss = F.cross_entropy(logits, y[ns:])
        grads = torch.autograd.grad(
            loss, [model.projection.wr.weight, model.encoder[0].conv.wr.weight],
            allow_unused=True,
        )
        if any(g is None for g in grads):
            raise RuntimeError("missing gradient path from Prototype CE to CCSAE encoder")
        grad_report[norm] = {
            "episode_ce": float(loss.detach()),
            "query_logit_stats": _logit_stats(logits),
            "projection_gradient_l2": float(grads[0].norm()),
            "first_complex_conv_gradient_l2": float(grads[1].norm()),
        }

    model.load_state_dict(base_state)
    model.eval()
    with torch.no_grad():
        features_eval = model(x)["features"]
    model.train()
    with torch.no_grad():
        features_train = model(x)["features"]
    # train-mode normalization updates buffers only inside this disposable diagnostic
    # model. Never save mutated model state.
    norm_shift = {
        "same_batch_train_vs_eval_mean_cosine": float(
            F.cosine_similarity(features_eval, features_train, dim=1).mean()
        ),
        "same_batch_train_vs_eval_mean_relative_l2": float(
            ((features_train - features_eval).norm(dim=1)
             / features_eval.norm(dim=1).clamp_min(1e-12)).mean()
        ),
    }
    model.load_state_dict(base_state)
    model.eval()

    return {
        "validation_only": True,
        "num_train": int(len(y_train)), "num_validation": len(y_val),
        "prototype_dim": int(z_train.shape[1]),
        "stage1_full_K_val_macro_f1": macro_metrics(y_val, pred_dim, classes)["macro_f1"],
        "stage1_full_K_val_accuracy": macro_metrics(y_val, pred_dim, classes)["accuracy"],
        "fixed_embedding_logits": {
            "dim": _logit_stats(dim_logits),
            "none": _logit_stats(raw_logits),
            "dim_vs_none_prediction_differences": int(sum(a != b for a, b in zip(pred_dim, pred_raw))),
            "dim_vs_fp64_reference_prediction_differences": int(
                sum(a != b for a, b in zip(pred_dim, reference_pred))
            ),
        },
        "embedding_geometry": {
            "class_mean_within_rms_distance": float(within_rms.mean()),
            "class_mean_nearest_center_distance": float(nearest_center.mean()),
            "mean_within_to_nearest_center_ratio": float(ratio.mean()),
        },
        "support_stability": {
            "support_per_class": support, "repeats": repeats,
            "three_shot_to_full_center_relative_drift": _stats(drifts),
            "val_macro_f1_across_sampled_supports": _stats(per_draw_f1),
        },
        "one_episode_train_mode_gradients": grad_report,
        "batchnorm_proxy": norm_shift,
        "interpretation_limits": (
            "Feature geometry and stability do not isolate causal training effects. "
            "Fixed-checkpoint logit scaling normally leaves argmax unchanged; only retraining "
            "can test optimization effects. BN proxy is a same-batch mode comparison, "
            "not a full attribution of episode-to-evaluation mismatch."
        ),
    }


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed if args.seed is not None else 42)
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    original_audit = json.loads((run_dir / "split_audit.json").read_text(encoding="utf-8"))
    seed = int(config["seed"])
    if args.seed is not None and seed != args.seed:
        raise ValueError("requested seed differs from run_dir/config.json")
    if not (run_dir / "shared_stage1_last.pt").exists():
        raise FileNotFoundError("shared_stage1_last.pt missing; use local/server output directory with weights")
    split = make_fewshot_split(
        args.data, config["case"], k=int(config["k"]),
        val_per_class=int(config["val_per_class"]),
        split_seed=int(config["split_seed"]),
        sample_seed=int(config["sample_seed"]) if config.get("sample_seed") is not None else seed,
        val_seed=int(config["val_seed"]),
    )
    for key in ("selected_train_hashes_by_class", "selected_val_hashes_by_class",
                "train_pool_sha256", "val_pool_sha256", "test_pool_sha256"):
        if split.audit[key] != original_audit[key]:
            raise ValueError(f"reconstructed data manifest mismatch: {key}")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    model = CCSAE(make_config(config["case"], bool(config.get("tiny", False)))).to(device)
    checkpoint = torch.load(run_dir / "shared_stage1_last.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    report = diagnose_model(
        model, split, device, seed=seed,
        support=int(config["support"]), query=int(config["query"]), repeats=args.repeats
    )
    report.update({
        "source_run_directory": str(run_dir), "case": config["case"], "seed": seed,
        "k": int(config["k"]), "test_evaluated": False,
        "stage1_checkpoint": "shared_stage1_last.pt",
        "data_manifest_verified": True,
    })
    output = (Path(args.output) if args.output else
              Path("outputs/proto_diagnostics") / config["case"] /
              f"{run_dir.name}_stage1.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "case": config["case"], "seed": seed,
        "stage1_full_K_val_macro_f1": report["stage1_full_K_val_macro_f1"],
        "dim_vs_none_prediction_differences":
            report["fixed_embedding_logits"]["dim_vs_none_prediction_differences"],
        "test_evaluated": False,
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Stage1 prototype diagnostics (no Test)")
    parser.add_argument("--run-dir", required=True, help="Existing K20 run with config, audit and Stage1 checkpoint")
    parser.add_argument("--data", default="data/Data.zip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--output", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
