"""固定基线实验：先验证流程，训练全部模型，最后统一评估测试集。

命令分为 smoke/train/test。正式训练不创建测试 DataLoader；验证 Macro-F1
选择最优 checkpoint，冻结的 12 次训练全部完成后才允许 test 阶段运行。
训练轮次、损失、归一化、网络宽度和种子固定，不由测试成绩反向修改。
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import update_bn

from .data import fingerprint, make_loader, prepare, write_json
from .metrics import classification_metrics
from .models import make_model


CONFIG = {"batch_size": 128, "max_epochs": 100, "patience": 15,
          "learning_rate": 1e-3, "weight_decay": 1e-4, "threads": 4,
          "scheduler_factor": 0.5, "scheduler_patience": 5, "workers": 2,
          "bn_statistics": "train_only_recalibration_before_each_validation; fixed_row_order"}
SEEDS = (42, 43, 44)
PROJECT = Path(__file__).resolve().parent.parent


def set_seed(seed: int, threads: int, device: str = "cpu") -> None:
    """固定随机状态；CUDA 使用与服务器 CCSAE 相同的确定性 FP32 设置。

    GPU 不可用时直接报错，不自动换回 CPU。不启用 AMP 或 TF32；固定种子
    保证的是当前环境的重复性，不能保证不同设备之间逐位一致。
    """
    if device == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if not torch.cuda.is_available():
            raise RuntimeError("明确请求 GPU，但当前环境无法使用 CUDA")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def source_path(plan: dict, name: str) -> Path:
    """定位实际执行的源码与 uv 配置，支持复用相邻 CCSAE 的锁定环境。

    旧 CPU 计划未记录环境项目目录时，依赖配置仍从基线项目读取。
    """
    if name in ("uv.lock", "pyproject.toml", ".python-version"):
        return Path(plan["environment"].get("project_dir", str(PROJECT))) / name
    if name in ("verify_reproducibility.py", "run_server_gpu.py"):
        return PROJECT / name
    return PROJECT / "baselines" / name


def environment_info(device: str) -> dict:
    """记录实际解释器、数值库、GPU 和精度设置，供三个模型对照检查。"""
    info = {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
            "python_executable": sys.executable, "project_dir": str(Path(sys.prefix).parent),
            "device": device, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if device == "cuda":
        info.update(cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    gpu_name=torch.cuda.get_device_name(0),
                    compute_capability=list(torch.cuda.get_device_capability(0)),
                    matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision,
                    cudnn_fp32_precision=torch.backends.cudnn.fp32_precision,
                    cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"])
    return info


def check_fixed_inputs(prepared: Path, meta: dict) -> dict:
    """逐元素核对新缓存与原基线，并确认服务器 CCSAE 用同一份输入。

    旧 metadata 含 Windows 路径，因此只对照数值统计、类别权重和 CSV 指纹。
    六个集合的波形或标签有任何变化都停止，防止迁移环境改变预处理。
    """
    reference = PROJECT / "artifacts" / "baseline_v2" / "prepared"
    original_meta = json.loads((reference / "metadata.json").read_text(encoding="utf-8"))
    ccsae = json.loads((PROJECT.parent / "Conference-Paper" / "outputs" / "server_gpu_v1" /
                        "summary.json").read_text(encoding="utf-8"))
    expected = ccsae["runs"][0]["protocol"]["prepared_file_sha256"]
    for name, digest in expected.items():
        if fingerprint(reference / name) != digest:
            raise ValueError("原基线缓存与服务器 CCSAE 冻结输入不同")
    if meta["circuits"] != original_meta["circuits"] or meta["file_sha256"] != original_meta["file_sha256"]:
        raise ValueError("标准化、类别权重或 CSV 与原正式基线不同")
    checks = []
    for circuit in meta["circuits"]:
        for split in ("train", "val", "test"):
            name = f"{circuit}_{split}.npz"
            with np.load(prepared / name, allow_pickle=False) as actual, np.load(reference / name, allow_pickle=False) as old:
                if not all(np.array_equal(actual[key], old[key]) for key in ("x", "y")):
                    raise ValueError(f"{name}：波形或标签与原正式输入不同")
            checks.append(name)
    return {"status": "passed", "arrays_equal": checks,
            "same_inputs_as_local_baselines_and_server_ccsae": True,
            "same_train_normalization_and_class_weights": True}


def evaluate(model, loader, classes: int) -> tuple[dict, dict]:
    """不更新参数，返回完整预测及指标；评估损失使用未加权交叉熵。"""
    model.eval()
    outputs = {key: [] for key in ("row_index", "label", "prediction", "probability")}
    loss_sum = 0.0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for x, y, index in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise ValueError("评估分数含 NaN/Inf")
            loss_sum += float(nn.functional.cross_entropy(logits, y, reduction="sum"))
            values = (index, y, logits.argmax(1), logits.softmax(1))
            for key, value in zip(outputs, values):
                outputs[key].append(value.cpu().numpy())
    arrays = {key: np.concatenate(value) for key, value in outputs.items()}
    metrics = classification_metrics(arrays["label"], arrays["prediction"], classes)
    metrics["loss_unweighted"] = loss_sum / len(arrays["label"])
    return metrics, arrays


def train_one(plan: dict, prepared: Path, output: Path, job: dict, meta: dict) -> dict:
    """训练一个冻结配置，记录逐轮日志；最优验证分数相同则保留更早轮次。"""
    config = plan["training"]
    circuit, kind, seed = job["circuit"], job["model"], job["seed"]
    device = plan["environment"]["device"]
    set_seed(seed, config["threads"], device)
    folder = output / job["id"]
    folder.mkdir(exist_ok=False)
    classes = meta["circuits"][circuit]["classes"]
    model = make_model(kind, classes).to(device)
    train_loader = make_loader(prepared, circuit, "train", config["batch_size"], seed)
    calibration_loader = make_loader(prepared, circuit, "train", config["batch_size"], seed, shuffle=False)
    val_loader = make_loader(prepared, circuit, "val", config["batch_size"], seed)
    weights = torch.tensor(meta["circuits"][circuit]["class_weights"], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config["scheduler_factor"], patience=config["scheduler_patience"])
    write_json(folder / "experiment.json", {**job, "training": config, "classes": classes,
               "parameters": sum(p.numel() for p in model.parameters()),
               "prepared_metadata_sha256": fingerprint(prepared / "metadata.json"),
               "plan_sha256": fingerprint(output / "plan.json"), "device": device})
    fields = ["epoch", "train_loss_weighted", "val_loss_unweighted", "val_macro_f1",
              "val_accuracy", "learning_rate", "seconds"]
    best, best_epoch, stale, history = -1.0, 0, 0, []
    start = time.perf_counter()
    print(f"START {job['id']}", flush=True)
    with (folder / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config["max_epochs"] + 1):
            epoch_start = time.perf_counter()
            model.train()
            total_loss, total_weight = 0.0, 0.0
            current_lr = optimizer.param_groups[0]["lr"]
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                numerator = criterion(model(x), y)
                denominator = weights[y].sum()
                loss = numerator / denominator
                if not torch.isfinite(loss):
                    raise ValueError(f"{job['id']}/{epoch}：训练损失非有限数")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError(f"{job['id']}/{epoch}：训练梯度非有限数")
                optimizer.step()
                total_loss += float(numerator.detach())
                total_weight += float(denominator)
            # 参数在一个 epoch 内不断变化，深层 BN 的滑动统计可能滞后。
            # 使用官方函数，在固定参数下遍历训练输入重估统计量；不读取验证标签，
            # 不计算梯度、不更新任何可训练参数。两种网络执行完全相同的步骤。
            update_bn(calibration_loader, model, device=device)
            val_metrics, _ = evaluate(model, val_loader, classes)
            score = val_metrics["macro_f1"]
            row = {"epoch": epoch, "train_loss_weighted": total_loss / total_weight,
                   "val_loss_unweighted": val_metrics["loss_unweighted"], "val_macro_f1": score,
                   "val_accuracy": val_metrics["accuracy"], "learning_rate": current_lr,
                   "seconds": time.perf_counter() - epoch_start}
            writer.writerow(row)
            handle.flush()
            history.append(row)
            if score > best:
                best, best_epoch, stale = score, epoch, 0
                torch.save({"state": model.state_dict(), "best_epoch": epoch,
                            "best_val_macro_f1": score, "bn_statistics": config["bn_statistics"]}, folder / "best.pt")
            else:
                stale += 1
            scheduler.step(score)
            # 每个进程只写自己的进度文件，避免并行任务覆盖对方的记录。
            write_json(folder / "progress.json", {**job, **row, "best_epoch": best_epoch,
                       "best_val_macro_f1": best, "phase": "train"})
            if epoch == 1 or epoch % 10 == 0 or stale >= config["patience"]:
                print(f"{job['id']} epoch={epoch} val_F1={score:.4f} best={best:.4f} epoch_s={row['seconds']:.2f}", flush=True)
            if stale >= config["patience"]:
                break
    # 模型选择完成后重载磁盘权重；训练阶段仍只验证验证集。
    restored = make_model(kind, classes).to(device)
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    restored.load_state_dict(checkpoint["state"])
    validation, prediction = evaluate(restored, val_loader, classes)
    if validation["macro_f1"] != best:
        raise ValueError("重载 checkpoint 后验证指标不同")
    np.savez_compressed(folder / "val_predictions.npz", **prediction)
    result = {"status": "trained", **job, "parameters": sum(p.numel() for p in restored.parameters()),
              "best_epoch": best_epoch, "epochs_run": len(history), "validation": validation,
              "train_seconds": time.perf_counter() - start,
              "checkpoint_sha256": fingerprint(folder / "best.pt"), "test_evaluated": False}
    write_json(folder / "result.json", result)
    print(f"TRAINED {job['id']} best_epoch={best_epoch} val_F1={best:.4f}", flush=True)
    return result


def train_all(data_dir: Path, output: Path, device: str = "cpu") -> None:
    """在首次训练前保存完整计划与源码快照，拒绝覆盖已存在的实验目录。"""
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / "prepared"
    meta = prepare(data_dir, prepared)
    set_seed(42, CONFIG["threads"], device)
    if device == "cuda":
        write_json(output / "input_equivalence.json", check_fixed_inputs(prepared, meta))
    environment = environment_info(device)
    source = output / "source_snapshot"
    source.mkdir()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copy2(path, source / path.name)
    # 保存实际 GPU 环境的配置，避免把未使用的旧 CPU 锁文件当作本次依赖。
    environment_project = Path(environment["project_dir"])
    for name in ("uv.lock", "pyproject.toml", ".python-version"):
        shutil.copy2(environment_project / name, source / name)
    for name in ("verify_reproducibility.py", "run_server_gpu.py"):
        shutil.copy2(PROJECT / name, source / name)
    jobs = [{"id": f"{circuit}_{model}_seed{seed}", "circuit": circuit, "model": model, "seed": seed}
            for circuit in ("SKLCircuit", "ABLCircuit") for model in ("cnn1d", "resnet1d") for seed in SEEDS]
    config = {**CONFIG, "workers": 1 if device == "cuda" else CONFIG["workers"]}
    plan = {"training": config, "runs": jobs, "selection": "max_validation_macro_f1; first_epoch_on_ties",
            "source_sha256": {p.name: fingerprint(p) for p in sorted(source.iterdir())},
            "prepared_sha256": {p.name: fingerprint(p) for p in sorted(prepared.iterdir())},
            "environment": environment,
            "protocol": "deduplicated_fixed_70_15_15; no_test_feedback",
            "architecture": {"cnn1d": "channels32/64/128;kernels7/5/3;maxpool2;GAP;head64;dropout0.2",
                             "resnet1d": "ResNet18_basic_blocks1D;stages2/2/2/2;channels32/64/128/256;GAP"}}
    write_json(output / "plan.json", plan)
    # 各进程独立设置种子、构造模型及优化器；只共享只读数据和冻结计划。
    # GPU 每次只跑一个任务；spawn 避免初始化 CUDA 后再 fork 产生设备错误。
    with ProcessPoolExecutor(max_workers=config["workers"],
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(train_one, plan, prepared, output, job, meta): job for job in jobs}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            write_json(output / "progress.json", {"phase": "train", "completed_runs": completed,
                       "total_runs": len(jobs), "last_completed": result["id"]})
    write_json(output / "training_complete.json", {"status": "complete", "runs": len(jobs), "test_evaluated": False})


def test_all(output: Path) -> None:
    """全部训练完成后逐个重载最优权重评估测试集；禁止重复评测覆盖结果。"""
    if (output / "summary.json").exists():
        raise FileExistsError("测试汇总已经存在，拒绝覆盖")
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if not (output / "training_complete.json").exists():
        raise ValueError("尚未完成全部冻结训练，禁止评估测试集")
    prepared = output / "prepared"
    meta = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))
    for name, expected in plan["source_sha256"].items():
        if fingerprint(output / "source_snapshot" / name) != expected:
            raise ValueError("源码快照已变化")
        current = source_path(plan, name)
        if fingerprint(current) != expected:
            raise ValueError("当前执行源码或依赖配置与冻结版本不同")
    for name, expected in plan["prepared_sha256"].items():
        if fingerprint(prepared / name) != expected:
            raise ValueError("准备数据已变化")
    results = []
    for job in plan["runs"]:
        device = plan["environment"]["device"]
        set_seed(job["seed"], plan["training"]["threads"], device)
        folder = output / job["id"]
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if result["status"] != "trained" or result["test_evaluated"]:
            raise ValueError("训练未完成或该模型已经评估过测试集")
        if fingerprint(folder / "best.pt") != result["checkpoint_sha256"]:
            raise ValueError("最优权重与训练完成时的指纹不同")
        classes = meta["circuits"][job["circuit"]]["classes"]
        model = make_model(job["model"], classes).to(device)
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state"])
        loader = make_loader(prepared, job["circuit"], "test", plan["training"]["batch_size"], job["seed"])
        metrics, prediction = evaluate(model, loader, classes)
        np.savez_compressed(folder / "test_predictions.npz", **prediction)
        result.update({"status": "complete", "test": metrics, "test_evaluated": True})
        write_json(folder / "result.json", result)
        results.append(result)
        print(f"TESTED {job['id']} acc={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f}", flush=True)
    aggregates = []
    for circuit in ("SKLCircuit", "ABLCircuit"):
        for model in ("cnn1d", "resnet1d"):
            selected = [r for r in results if r["circuit"] == circuit and r["model"] == model]
            row = {"circuit": circuit, "model": model, "runs": len(selected), "parameters": selected[0]["parameters"]}
            for split, field in (("validation", "val"), ("test", "test")):
                for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                    values = [r[split][metric] for r in selected]
                    row[f"{field}_{metric}_mean"] = float(np.mean(values))
                    row[f"{field}_{metric}_std"] = float(np.std(values, ddof=1))
            row["train_seconds_total"] = float(sum(r["train_seconds"] for r in selected))
            aggregates.append(row)
    write_json(output / "summary.json", {"status": "complete", "groups": aggregates, "runs": results,
               "std_definition": "sample_std_across_3_training_seeds; fixed_data_split"})
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)


def smoke_one(circuit: str, kind: str, classes: int, prepared: Path, output: Path, device: str = "cpu") -> dict:
    """在独立进程检查真实数据前向/梯度、BN 重估和权重重载。"""
    set_seed(42, CONFIG["threads"], device)
    model = make_model(kind, classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = make_loader(prepared, circuit, "train", CONFIG["batch_size"], 42)
    begin = time.perf_counter()
    model.train()
    for batch, (x, y, _) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        if logits.shape != (len(y), classes):
            raise ValueError("网络输出形状错误")
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("短流程损失或梯度异常")
        optimizer.step()
        if batch == 1:
            break
    two_batch_seconds = time.perf_counter() - begin
    parameters = [p.detach().clone() for p in model.parameters()]
    update_bn(make_loader(prepared, circuit, "train", CONFIG["batch_size"], 42, shuffle=False), model, device=device)
    if not all(torch.equal(a, b) for a, b in zip(parameters, model.parameters())):
        raise ValueError("BN 重估改变了可训练参数")
    val_loader = make_loader(prepared, circuit, "val", CONFIG["batch_size"], 42)
    _, before = evaluate(model, val_loader, classes)
    path = output / f"{circuit}_{kind}.pt"
    torch.save(model.state_dict(), path)
    restored = make_model(kind, classes).to(device)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    _, after = evaluate(restored, val_loader, classes)
    if not np.array_equal(before["probability"], after["probability"]):
        raise ValueError("短流程权重重载后预测不同")
    return {"circuit": circuit, "model": kind,
            "parameters": sum(p.numel() for p in model.parameters()),
            "two_train_batches_seconds": two_batch_seconds, "reload_predictions_exact": True,
            "bn_recalibration_parameters_unchanged": True}


def smoke(data_dir: Path, output: Path, device: str = "cpu") -> None:
    """四种组合各训练两个批次，只评估验证集，并检查设备与进程兼容性。"""
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / "prepared"
    meta = prepare(data_dir, prepared)
    if device == "cuda":
        write_json(output / "input_equivalence.json", check_fixed_inputs(prepared, meta))
    with ProcessPoolExecutor(max_workers=1 if device == "cuda" else CONFIG["workers"],
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(smoke_one, circuit, kind, meta["circuits"][circuit]["classes"], prepared, output, device)
                   for circuit in ("SKLCircuit", "ABLCircuit") for kind in ("cnn1d", "resnet1d")]
        checks = [future.result() for future in futures]
    write_json(output / "result.json", {"status": "passed", "checks": checks, "test_evaluated": False})
    print(json.dumps(checks), flush=True)


def main():
    """运行授权的两个基线；默认目录独立于原数据及历史电力变换器项目。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train", "test"), required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT.parent / "Data_deduplicated")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    output = args.output_dir or PROJECT / "artifacts" / ("smoke_v2" if args.mode == "smoke" else "baseline_v2")
    if args.mode == "smoke":
        smoke(args.data_dir, output, args.device)
    elif args.mode == "train":
        train_all(args.data_dir, output, args.device)
    else:
        test_all(output)


if __name__ == "__main__":
    main()
