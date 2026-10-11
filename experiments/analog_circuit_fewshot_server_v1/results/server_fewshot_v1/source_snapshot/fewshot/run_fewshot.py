"""固定 54 次少样本实验，复用三个模型的既有训练代码和服务器 uv 环境。

流程为准备、训练、训练核验、统一测试、最终核验和报告。源文件或数据变化、
任何训练失败都会停止；不覆盖旧输出，不根据测试分数调整样本量或超参数。
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(ROOT / "_reference"))
sys.path.insert(0, str(WORKSPACE / "Conference-Paper-ccsae-v1" / "src"))

import numpy as np
import torch
from baselines import run as baseline
from baselines.data import make_loader
from baselines.models import make_model
from ccsae.fixed_data import fingerprint, load_fixed, read_json, write_json
from ccsae.fixed_experiment import checkpoint_model, configure, predict
from ccsae.train import run as run_ccsae
from fewshot_data import CASES, SHOTS, SEEDS, cache_folder, load_splits, prepare


MODELS = ("cnn1d", "resnet1d", "ccsae_v1")
DATA = WORKSPACE / "Data_deduplicated"
REFERENCE = WORKSPACE / "analog-circuit-baselines" / "artifacts" / "baseline_v2"
BASELINE_FULL = WORKSPACE / "analog-circuit-baselines" / "artifacts" / "server_gpu_v1"
CCSAE_FULL = WORKSPACE / "Conference-Paper-ccsae-v1" / "outputs" / "server_v1_signed_v1"
ENVIRONMENT = WORKSPACE / "Conference-Paper"


def ccsae_folder(output: Path, job: dict) -> Path:
    """仓库训练入口自行追加 case/seed，保留该布局并明确权重位置。"""
    return output / job["id"] / "training" / job["case"] / f"seed_{job['seed']}"


def setup(output: Path, smoke: bool) -> dict:
    """运行前冻结抽样、训练预算、原始划分、参考权重及所有执行源码。"""
    output.mkdir(parents=True, exist_ok=False)
    baseline_old = read_json(BASELINE_FULL / "plan.json")
    ccsae_old = read_json(CCSAE_FULL / "plan.json")
    assert read_json(BASELINE_FULL / "validation.json")["status"] == "passed"
    assert read_json(CCSAE_FULL / "validation.json")["status"] == "passed"
    assert ccsae_old["source_commit"] == "d4413979d27160c6a96690f9b544738c6ca92bd7"
    selected = prepare(DATA, REFERENCE, output)
    baseline_cfg, ccsae_cfg = baseline_old["training"].copy(), ccsae_old["training"].copy()
    if smoke:
        baseline_cfg.update(max_epochs=2, patience=3)
        ccsae_cfg.update(feature_epochs=1, classification_epochs=1)
    jobs = [{"id": f"{case}_{model}_k{k}_seed{seed}", "case": case, "circuit": CASES[case],
             "model": model, "k": k, "seed": seed}
            for k in ((10, 40) if smoke else SHOTS) for case in CASES
            for seed in ((42,) if smoke else SEEDS) for model in MODELS]
    # 使用服务器原文件的副本；快照可直接核对是否修改了模型及训练核心。
    sources = {}
    groups = [(ROOT, "fewshot", sorted(ROOT.glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))),
              (ROOT / "_reference", "reference", sorted((ROOT / "_reference" / "baselines").glob("*.py"))),
              (WORKSPACE / "Conference-Paper-ccsae-v1" / "src", "ccsae_v1",
               sorted((WORKSPACE / "Conference-Paper-ccsae-v1" / "src").rglob("*.py"))),
              (ENVIRONMENT, "environment", [ENVIRONMENT / n for n in ("pyproject.toml", "uv.lock", ".python-version")])]
    for source_root, prefix, files in groups:
        for source in files:
            relative = Path(prefix) / source.relative_to(source_root)
            destination = output / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            sources[str(source)] = {"sha256": fingerprint(source), "snapshot": relative.as_posix()}
    refs = {}
    for root in (BASELINE_FULL, CCSAE_FULL):
        for name in ("plan.json", "summary.json", "validation.json"):
            refs[str(root / name)] = fingerprint(root / name)
    for run in read_json(BASELINE_FULL / "summary.json")["runs"]:
        p = BASELINE_FULL / run["id"] / "best.pt"
        assert fingerprint(p) == run["checkpoint_sha256"]
        refs[str(p)] = fingerprint(p)
    for run in read_json(CCSAE_FULL / "summary.json")["runs"]:
        p = CCSAE_FULL / run["case"] / f"seed_{run['seed']}" / "stage2_best.pt"
        assert fingerprint(p) == run["stage2_sha256"]
        refs[str(p)] = fingerprint(p)
    plan = {"protocol": "balanced K-shot from original Train; fixed original Val/Test",
            "created_utc": datetime.now(timezone.utc).isoformat(), "smoke_only": smoke,
            "runs": jobs, "training": {"baseline": baseline_cfg, "ccsae": ccsae_cfg},
            "sampling_seeds": list(SEEDS), "shots": list(SHOTS), "split_seed": 20261009,
            "subset_is_nested": True, "normalization": "selected Train only",
            "unselected_train_usage": "none in training, pretraining or BN recalibration",
            "data_dir": str(DATA), "baseline_reference": str(REFERENCE),
            "source_sha256": sources, "reference_sha256": refs,
            "data_sha256": read_json(DATA / "validation.json")["output_file_sha256"],
            "prepared_sha256": {p.relative_to(output).as_posix(): fingerprint(p)
                                for p in sorted((output / "prepared").rglob("*")) if p.is_file()},
            "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                            "device": "cuda", "deterministic_fp32": True},
            "reference_baseline_metrics": "reuse already verified full-Train Val/Test results",
            "reference_ccsae_metrics": "reuse full-Train Val; evaluate existing weights on Test after all training",
            "test_policy": "all 54 selections and checkpoint hashes frozen before any new Test prediction"}
    write_json(output / "plan.json", plan)
    write_json(output / "preparation.json", {"status": "passed", "subsets": len(selected), "runs": len(jobs)})
    return plan


def check_frozen(output: Path, plan: dict) -> None:
    """任何源码、缓存、数据或历史权重变化都会报错，防止跨版本混用。"""
    for source, expected in plan["source_sha256"].items():
        assert fingerprint(Path(source)) == expected["sha256"], source
        assert fingerprint(output / "source_snapshot" / expected["snapshot"]) == expected["sha256"], source
    for relative, digest in plan["prepared_sha256"].items():
        assert fingerprint(output / relative) == digest, relative
    for source, digest in plan["reference_sha256"].items():
        assert fingerprint(Path(source)) == digest, source
    for name, digest in plan["data_sha256"].items():
        assert fingerprint(DATA / name) == digest, name


def train_ccsae(output: Path, plan: dict, job: dict) -> dict:
    """原样调用 V1 两阶段训练，只注入少样本缓存；每个任务从头初始化。"""
    cfg = plan["training"]["ccsae"]
    configure(job["seed"], cfg["threads"], "cuda")
    folder = output / job["id"]
    folder.mkdir(exist_ok=False)
    splits = load_splits(cache_folder(output, job["case"], job["seed"], job["k"]), job["case"])
    args = argparse.Namespace(data=str(DATA), case=job["case"], split="fixed", split_seed=20261009,
                              allow_official_overlap=False, output=str(folder / "training"), seed=job["seed"],
                              batch_size=cfg["batch_size"], feature_epochs=cfg["feature_epochs"],
                              classification_epochs=cfg["classification_epochs"],
                              lr=cfg["learning_rates"][job["case"]], scl_weight=cfg["scl_weights"][job["case"]],
                              temperature=cfg["temperature"], reconstruction_target="signed_real", device="cuda",
                              grad_clip=cfg["grad_clip"], evaluate_test=False, tiny=False, log_every=25)
    started = time.perf_counter()

    def record(row, best):
        """只记录已有训练指标，不改变优化器、随机状态或选模条件。"""
        write_json(folder / "progress.json", {**job, **row, "best_score": best})

    run_ccsae(args, data_splits=splits, progress_callback=record)
    weights = ccsae_folder(output, job)
    model, checkpoint = checkpoint_model(weights / "stage2_best.pt", "cuda")
    metrics, predictions = predict(model, splits.val, cfg["batch_size"])
    np.savez_compressed(folder / "val_predictions.npz", **predictions)
    result = {"status": "trained", **job, "validation": metrics, "best_epoch": checkpoint["epoch"],
              "epochs_run": cfg["classification_epochs"], "train_seconds": time.perf_counter() - started,
              "parameters": sum(p.numel() for p in model.parameters()),
              "checkpoint_sha256": fingerprint(weights / "stage2_best.pt"),
              "stage1_sha256": fingerprint(weights / "stage1_best.pt"), "test_evaluated": False}
    write_json(folder / "result.json", result)
    return result


def train_all(output: Path, plan: dict) -> None:
    """单 GPU 顺序执行 54 个任务；两种基线直接调用其既有 train_one 函数。"""
    check_frozen(output, plan)
    for index, job in enumerate(plan["runs"], 1):
        print(f"START {index}/{len(plan['runs'])} {job['id']}", flush=True)
        if job["model"] == "ccsae_v1":
            result = train_ccsae(output, plan, job)
        else:
            prepared = cache_folder(output, job["case"], job["seed"], job["k"])
            result = baseline.train_one({"training": plan["training"]["baseline"],
                                         "environment": {"device": "cuda"}}, prepared, output, job,
                                        read_json(prepared / "metadata.json"))
        print(f"DONE {index}/{len(plan['runs'])} {job['id']} val_F1={result['validation']['macro_f1']:.6f}", flush=True)
        write_json(output / "progress.json", {"completed": index, "total": len(plan["runs"]), "last": job["id"]})
    write_json(output / "training_complete.json", {"runs": len(plan["runs"]), "status": "complete"})


def restore(output: Path, job: dict):
    """从已选定的磁盘权重恢复模型，返回其验证或测试推理需要的架构。"""
    if job["model"] == "ccsae_v1":
        return checkpoint_model(ccsae_folder(output, job) / "stage2_best.pt", "cuda")[0]
    classes = read_json(cache_folder(output, job["case"], job["seed"], job["k"]) / "metadata.json")["circuits"][job["circuit"]]["classes"]
    model = make_model(job["model"], classes).to("cuda")
    model.load_state_dict(torch.load(output / job["id"] / "best.pt", map_location="cpu", weights_only=True)["state"])
    return model


def predictions(output: Path, job: dict, split: str):
    """三个模型使用同一少样本标准化缓存、固定原始评估行序。"""
    configure(job["seed"], 4, "cuda")
    model = restore(output, job)
    folder = cache_folder(output, job["case"], job["seed"], job["k"])
    if job["model"] == "ccsae_v1":
        return predict(model, getattr(load_splits(folder, job["case"]), split), 64)
    return baseline.evaluate(model, make_loader(folder, job["circuit"], split, 128, job["seed"], shuffle=False),
                             model.classifier.out_features if job["model"] == "resnet1d" else model.classifier[-1].out_features)


def test_all(output: Path, plan: dict) -> None:
    """全部少样本权重通过验证后统一测试；完整 V1 只评估既有权重，不重训。"""
    assert not plan["smoke_only"]
    assert read_json(output / "training_validation.json")["status"] == "passed"
    check_frozen(output, plan)
    frozen_weights = read_json(output / "training_validation.json")["checkpoint_sha256"]
    # 在任何新测试预测之前，先确认全部 54 份已选权重仍与训练核验记录一致。
    for job in plan["runs"]:
        weight = (ccsae_folder(output, job) / "stage2_best.pt" if job["model"] == "ccsae_v1"
                  else output / job["id"] / "best.pt")
        assert fingerprint(weight) == frozen_weights[job["id"]]
    for job in plan["runs"]:
        result = read_json(output / job["id"] / "result.json")
        assert result["status"] == "trained" and not result["test_evaluated"]
        metrics, values = predictions(output, job, "test")
        np.savez_compressed(output / job["id"] / "test_predictions.npz", **values)
        result.update(status="complete", test_evaluated=True, test=metrics)
        write_json(output / job["id"] / "result.json", result)
    full = []
    for run in read_json(BASELINE_FULL / "summary.json")["runs"]:
        full.append({**run, "case": "skl" if run["circuit"] == "SKLCircuit" else "abl", "k": "full",
                     "origin": "historical verified server full-Train results"})
    destination = output / "full_train_ccsae"
    destination.mkdir(exist_ok=False)
    for run in read_json(CCSAE_FULL / "summary.json")["runs"]:
        configure(run["seed"], 4, "cuda")
        splits = load_fixed(DATA, REFERENCE, run["case"])
        source = CCSAE_FULL / run["case"] / f"seed_{run['seed']}" / "stage2_best.pt"
        model, _ = checkpoint_model(source, "cuda")
        metrics, values = predict(model, splits.test, 64)
        name = f"{run['case']}_ccsae_v1_full_seed{run['seed']}"
        np.savez_compressed(destination / f"{name}_test_predictions.npz", **values)
        full.append({"id": name, "case": run["case"], "model": "ccsae_v1", "seed": run["seed"], "k": "full",
                     "validation": run["validation"], "test": metrics, "test_evaluated": True,
                     "parameters": run["parameters_total"], "train_seconds": run["train_seconds"],
                     "checkpoint_sha256": fingerprint(source), "checkpoint_path": str(source),
                     "origin": "existing V1 full-Train checkpoint; new Test inference, no retraining"})
    write_json(output / "full_train_results.json", full)
    write_json(output / "test_complete.json", {"status": "complete", "new_small_test_runs": 54, "new_full_ccsae_test_runs": 6})


def main():
    """一个进程执行冻结流程并保存真实状态，后台运行失败时保留全部证据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    state_path = Path(str(output) + ".pipeline.json")
    if output.exists() or state_path.exists():
        raise FileExistsError("输出已经存在，请使用新目录，禁止覆盖历史实验")
    state = {"status": "running", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
             "completed_phases": [], "smoke_only": args.smoke, "output_dir": str(output)}
    output.parent.mkdir(parents=True, exist_ok=True)

    def phase(name, action):
        """记录阶段起止；未成功完成的阶段不会被标记通过。"""
        state.update(phase=name, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        print("PHASE " + name, flush=True)
        value = action()
        state["completed_phases"].append(name)
        write_json(state_path, state)
        return value

    try:
        configure(42, 4, "cuda")
        plan = phase("prepare", lambda: setup(output, args.smoke))
        phase("train", lambda: train_all(output, plan))
        from verify_fewshot import verify_training, verify_final
        phase("verify_training", lambda: verify_training(output, plan))
        if not args.smoke:
            phase("test", lambda: test_all(output, plan))
            phase("verify_final", lambda: verify_final(output, plan))
            from analyze_fewshot import analyze
            phase("report", lambda: analyze(output, plan))
        state.update(status="complete", phase="complete", exit_code=0, updated_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        state.update(status="failed", error=repr(error), exit_code=1, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        raise
    write_json(state_path, state)
    print("FEWSHOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
