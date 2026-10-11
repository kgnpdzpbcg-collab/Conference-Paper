"""在固定去重划分上运行仓库 V1 signed_real 两阶段训练。

复用 V1 model/losses/train；输入由已冻结缓存注入，六次训练后只汇总验证集。
测试波形仅做数据完整性核验，不生成新的测试预测。
预检查仅运行完整结构的 1+1 轮，不进入正式结果。
"""

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader

from .fixed_data import fingerprint, load_fixed, read_json, write_json
from .model import CCSAE, CCSAEConfig
from .train import run, set_seed


PROJECT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
TRAINING = {"batch_size": 64, "feature_epochs": 100, "classification_epochs": 200,
            "momentum": 0.9, "temperature": 0.1, "grad_clip": 5.0,
            "threads": 4, "workers": 2, "reconstruction_target": "signed_real",
            "learning_rates": {"skl": 0.0022, "abl": 0.0102},
            "scl_weights": {"skl": 0.0017, "abl": 0.0059},
            "bn_statistics": "original_repository_running_statistics",
            "stage2_class_weights": "none; original_repository_cross_entropy"}


def configure(seed: int, threads: int, device: str = "cpu") -> None:
    """设置运算设备和随机种子；GPU 使用确定性 FP32，不启用混合精度。

    CUBLAS 工作区必须在第一次矩阵运算前设置。注意力使用数学后端，
    避免不同融合实现影响逐次重载核验；数据和优化器超参数不在这里调整。
    """
    torch.set_num_threads(threads)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("明确要求 CUDA，但当前环境无法使用 GPU")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    set_seed(seed)


def metrics_from_predictions(labels, predictions, classes: int) -> dict:
    """使用与基线相同的类别集合计算宏平均，空预测类按零处理。"""
    category_ids = list(range(classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=category_ids, zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predictions)), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall.mean()), "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(), "per_class_f1": f1.tolist(), "support": support.tolist(),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=category_ids).tolist()}


def predict(model: CCSAE, dataset, batch_size: int) -> tuple[dict, dict]:
    """固定行序推理，保存真实标签、类别概率和对应的零起始数据行号。"""
    model.eval()
    # 输入送到模型所在设备；保存指标和概率时移回 CPU，保持原行序。
    device = next(model.parameters()).device
    actual, predicted, probabilities, losses = [], [], [], 0.0
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            logits = model(x.to(device))["logits"]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("推理分类分数包含 NaN/Inf")
            losses += float(torch.nn.functional.cross_entropy(logits, y.to(device), reduction="sum"))
            actual.append(y.numpy())
            predicted.append(logits.argmax(1).cpu().numpy())
            probabilities.append(logits.softmax(1).cpu().numpy())
    y, pred = np.concatenate(actual), np.concatenate(predicted)
    metric = metrics_from_predictions(y, pred, model.cfg.classes)
    metric["loss_unweighted"] = losses / len(y)
    arrays = {"row_index": np.arange(len(y)), "label": y, "prediction": pred,
              "probability": np.concatenate(probabilities)}
    return metric, arrays


def checkpoint_model(path: Path, device: str = "cpu") -> tuple[CCSAE, dict]:
    """读取仓库保存的模型配置与权重，不从测试成绩推断任何架构。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = CCSAE(CCSAEConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    # 权重先安全加载到 CPU，随后只移动模型；核验仍可读取 CPU 状态字典。
    return model.to(device), checkpoint


def train_job(plan: dict, output: Path, job: dict) -> dict:
    """调用仓库训练函数，增量保存日志；训练完成后仅重载验证集预测。"""
    cfg = plan["training"]
    configure(job["seed"], cfg["threads"], cfg["device"])
    splits = load_fixed(Path(plan["data_dir"]), Path(plan["baseline_dir"]), job["case"])
    args = argparse.Namespace(data=plan["data_dir"], case=job["case"], split="fixed", split_seed=20261009,
                              allow_official_overlap=False, output=str(output), seed=job["seed"],
                              batch_size=cfg["batch_size"], feature_epochs=cfg["feature_epochs"],
                              classification_epochs=cfg["classification_epochs"],
                              lr=cfg["learning_rates"][job["case"]], scl_weight=cfg["scl_weights"][job["case"]],
                              temperature=cfg["temperature"], reconstruction_target=cfg["reconstruction_target"],
                              device=cfg["device"], grad_clip=cfg["grad_clip"], evaluate_test=False, tiny=False, log_every=10)
    folder = output / job["case"] / f"seed_{job['seed']}"
    history, started = [], time.perf_counter()

    def record(row: dict, best: float) -> None:
        """此回调仅写进度/日志，不接触模型或更改训练决策。"""
        history.append(row)
        write_json(folder / "history_live.json", history)
        write_json(folder / "progress.json", {**job, **row, "best_selection_score": best,
                   "elapsed_seconds": time.perf_counter() - started})

    print(f"START {job['id']}", flush=True)
    summary = run(args, data_splits=splits, progress_callback=record)
    model, checkpoint = checkpoint_model(folder / "stage2_best.pt", cfg["device"])
    validation, prediction = predict(model, splits.val, cfg["batch_size"])
    if abs(validation["macro_f1"] - summary["best_val_macro_f1"]) > 1e-12:
        raise AssertionError("重载权重后的验证 Macro-F1 与选模记录不符")
    np.savez_compressed(folder / "val_predictions.npz", **prediction)
    total = sum(p.numel() for p in model.parameters())
    decoder = sum(p.numel() for p in model.decoder.parameters())
    summary.update({"status": "trained", "id": job["id"], "model": "ccsae_v1_signed",
                    "validation": validation, "parameters_total": total,
                    "parameters_inference": total - decoder, "train_seconds": time.perf_counter() - started,
                    "stage1_sha256": fingerprint(folder / "stage1_best.pt"),
                    "stage2_sha256": fingerprint(folder / "stage2_best.pt"),
                    "experiment_plan_sha256": fingerprint(output / "plan.json"),
                    "best_classification_epoch": checkpoint["epoch"]})
    write_json(folder / "summary.json", summary)
    print(f"TRAINED {job['id']} val_F1={validation['macro_f1']:.4f}", flush=True)
    return summary


def check_frozen(plan: dict, output: Path) -> None:
    """检查当前源码、锁文件、数据及先前基线缓存，拒绝混用不同版本。"""
    for relative, expected in plan["source_sha256"].items():
        if fingerprint(PROJECT / relative) != expected or fingerprint(output / "source_snapshot" / relative) != expected:
            raise ValueError(f"源码或环境配置变化：{relative}")
    # 另核对实际复用的 uv 环境配置，而非仅检查归档副本。
    for name, expected in plan["environment_sha256"].items():
        if fingerprint(Path(plan["environment_project"]) / name) != expected:
            raise ValueError(f"实际 GPU 环境配置变化：{name}")
    baseline = Path(plan["baseline_dir"])
    if fingerprint(baseline / "plan.json") != plan["baseline_plan_sha256"]:
        raise ValueError("前两个基线的计划发生变化")
    for name, expected in plan["baseline_result_sha256"].items():
        if fingerprint(baseline / name) != expected:
            raise ValueError("前两个基线的结果或核验记录发生变化")
    for case in ("skl", "abl"):
        load_fixed(Path(plan["data_dir"]), baseline, case)


def train_all(data_dir: Path, baseline: Path, output: Path, smoke: bool = False, device: str = "cpu",
              reconstruction_target: str = "signed_real") -> None:
    """开始前冻结六次运行的全部设计；预检查只有两个 seed42 的 1+1 轮。"""
    output.mkdir(parents=True, exist_ok=False)
    audits = {case: load_fixed(data_dir, baseline, case).report for case in ("skl", "abl")}
    if reconstruction_target != "signed_real" and not smoke:
        raise ValueError("正式任务仅运行已授权的 V1 signed_real；旧模式仅用于短流程兼容检查")
    settings = {**TRAINING, "reconstruction_target": reconstruction_target}
    # 单张 GPU 顺序运行，避免两个训练进程争用显存和计算资源。
    settings.update(device=device, workers=1 if device == "cuda" else TRAINING["workers"],
                    numeric_precision="float32_ieee_no_amp", deterministic_algorithms=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，拒绝静默转为 CPU")
    if smoke:
        settings.update(feature_epochs=1, classification_epochs=1)
    jobs = [{"id": f"{case}_ccsae_seed{seed}", "case": case, "seed": seed}
            for seed in ((42,) if smoke else SEEDS) for case in ("skl", "abl")]
    snapshot = output / "source_snapshot"
    files = [PROJECT / "pyproject.toml", PROJECT / "run_server_v1.py"]
    files += sorted((PROJECT / "environment_snapshot").iterdir())
    files += sorted((PROJECT / "src").rglob("*.py")) + sorted((PROJECT / "tests").rglob("*.py"))
    hashes = {}
    for path in files:
        relative = path.relative_to(PROJECT)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[relative.as_posix()] = fingerprint(path)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    plan = {"training": settings, "runs": jobs, "data_dir": str(data_dir.resolve()),
            "baseline_dir": str(baseline.resolve()), "baseline_plan_sha256": fingerprint(baseline / "plan.json"),
            "source_commit": commit, "source_sha256": hashes, "audits": audits,
            "environment_project": str(Path(sys.prefix).parent),
            "environment_sha256": {name: fingerprint(Path(sys.prefix).parent / name)
                                   for name in ("pyproject.toml", "uv.lock", ".python-version")},
            "baseline_result_sha256": {name: fingerprint(baseline / name)
                                       for name in ("summary.json", "summary.csv", "validation.json")},
            "model_config": {case: asdict(CCSAEConfig.for_case(case)) for case in ("skl", "abl")},
            "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                            "python_executable": sys.executable, "device": device,
                            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
                            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)) if device == "cuda" else None},
            "smoke_only": smoke,
            "selection": "stage1:min_validation_reconstruction_plus_SCL; stage2:max_validation_macro_f1; first_on_ties",
            "test_evaluated": False, "evaluation_scope": "validation_only; test_integrity_check_only",
            "implementation": "repository_independent_adaptation_v1_signed_real"}
    write_json(output / "plan.json", plan)
    # CUDA 不能在初始化后通过 fork 复用；spawn 在新进程中单独初始化设备。
    with ProcessPoolExecutor(max_workers=settings["workers"], mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(train_job, plan, output, job): job for job in jobs}
        completed = 0
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as error:
                write_json(output / "failure.json", {"job": job, "error": repr(error), "test_evaluated": False})
                print(f"FAILED {job['id']}: {error}", flush=True)
                raise
            completed += 1
            write_json(output / "progress.json", {"completed_runs": completed, "total_runs": len(jobs),
                       "last_completed": job["id"], "phase": "train"})
    write_json(output / "training_complete.json", {"status": "complete", "runs": len(jobs),
               "smoke_only": smoke, "test_evaluated": False})


def aggregate_validation(output: Path) -> None:
    """六次正式训练完成后，只汇总验证集；明确拒绝测试预测及重复覆盖。"""
    plan = read_json(output / "plan.json")
    if plan["smoke_only"] or not (output / "training_complete.json").exists():
        raise ValueError("需要完成全部六次正式训练")
    if (output / "summary.json").exists():
        raise FileExistsError("验证汇总已存在，拒绝覆盖")
    check_frozen(plan, output)
    results = []
    for job in plan["runs"]:
        folder = output / job["case"] / f"seed_{job['seed']}"
        result = read_json(folder / "summary.json")
        if result["status"] != "trained" or result["test_evaluated"] or "test" in result:
            raise ValueError("结果状态不符合验证集实验范围")
        if result["experiment_plan_sha256"] != fingerprint(output / "plan.json"):
            raise ValueError("单次运行未使用冻结计划")
        result["status"] = "complete_validation"
        write_json(folder / "summary.json", result)
        results.append(result)
    groups = []
    for case in ("skl", "abl"):
        selected = [r for r in results if r["case"] == case]
        row = {"case": case, "model": "ccsae_v1_signed", "runs": len(selected),
               "parameters_total": selected[0]["parameters_total"],
               "parameters_inference": selected[0]["parameters_inference"]}
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            values = [r["validation"][metric] for r in selected]
            row[f"val_{metric}_mean"] = float(np.mean(values))
            row[f"val_{metric}_std"] = float(np.std(values, ddof=1))
        groups.append(row)
    write_json(output / "summary.json", {"status": "complete_validation", "groups": groups,
               "runs": results, "test_evaluated": False,
               "std_definition": "sample_std_across_3_training_seeds; fixed_data_split"})
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)


def main():
    """用 uv 运行 smoke/train/aggregate 三阶段；默认路径绑定本工作目录的既有数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train", "aggregate"), required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT.parent / "Data_deduplicated")
    parser.add_argument("--baseline-dir", type=Path, default=PROJECT.parent / "analog-circuit-baselines" / "artifacts" / "baseline_v2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reconstruction-target", choices=("signed_real", "paper_signed"), default="signed_real")
    args = parser.parse_args()
    output = args.output_dir or PROJECT / "outputs" / ("fixed_smoke_v1" if args.mode == "smoke" else "fixed_v1")
    if args.mode == "aggregate":
        aggregate_validation(output)
    else:
        train_all(args.data_dir, args.baseline_dir, output, smoke=args.mode == "smoke", device=args.device,
                  reconstruction_target=args.reconstruction_target)


if __name__ == "__main__":
    main()
