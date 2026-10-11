"""独立核查正式结果：数据指纹、选模依据、逐行预测和汇总指标。

不参与训练或超参数选择。使用逐类布尔计数和标准库 statistics 重算指标，
避免仅靠训练程序自己的汇总来证明正确；检查权重重载后的概率逐元素相同。
"""

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch
from torch.optim.swa_utils import update_bn

from .data import fingerprint, make_loader, write_json
from .metrics import classification_metrics
from .models import make_model
from .run import PROJECT, evaluate, set_seed, source_path


def read_json(path: Path) -> dict:
    """读取 UTF-8 元数据；文件缺失或解析失败直接停止核验。"""
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual, expected) -> None:
    """仅容许浮点运算顺序产生的微小差异，不能用宽松阈值掩盖指标错误。"""
    if not np.allclose(actual, expected, rtol=0, atol=1e-12):
        raise AssertionError(f"数值不一致：{actual} / {expected}")


def check_metric_fixture() -> None:
    """用可手算的三类案例检验宏平均，同时覆盖从未被预测的类别。

    首个案例五个样本中四个正确，逐类 F1 为 1、2/3、2/3，因此宏 F1=7/9；
    第二个案例模型只预测类别 0，另外两类的 F1 必须为零，不能被跳过。
    """
    metric = classification_metrics([0, 0, 1, 1, 2], [0, 0, 1, 2, 2], 3)
    close(metric["accuracy"], 4 / 5)
    close(metric["macro_f1"], 7 / 9)
    close(metric["balanced_accuracy"], 5 / 6)
    metric = classification_metrics([0, 1, 2], [0, 0, 0], 3)
    close(metric["per_class_f1"], [0.5, 0, 0])
    close(metric["macro_f1"], 1 / 6)


def verify_prediction(saved: Path, metric: dict, labels: np.ndarray, classes: int) -> None:
    """逐行对照真实标签，独立重算混淆矩阵及全部分类指标。"""
    with np.load(saved, allow_pickle=False) as archive:
        rows, y, predicted, probability = [archive[k] for k in ("row_index", "label", "prediction", "probability")]
    if not np.array_equal(rows, np.arange(len(labels))) or not np.array_equal(y, labels):
        raise AssertionError("预测记录与数据行顺序或标签不符")
    if probability.shape != (len(y), classes) or not np.isfinite(probability).all():
        raise AssertionError("概率维度错误或存在 NaN/Inf")
    if (probability < 0).any() or not np.allclose(probability.sum(1), 1, rtol=0, atol=2e-6):
        raise AssertionError("概率不是合法分布")
    if not np.array_equal(probability.argmax(1), predicted):
        raise AssertionError("预测类别与最大概率不符")
    confusion = np.zeros((classes, classes), dtype=np.int64)
    # 逐样本累加，不复用训练程序的 bincount 实现。
    for target, output in zip(y, predicted):
        confusion[target, output] += 1
    precisions, recalls, f1s = [], [], []
    for category in range(classes):
        true_positive = int(((y == category) & (predicted == category)).sum())
        actual_count = int((y == category).sum())
        predicted_count = int((predicted == category).sum())
        precisions.append(true_positive / predicted_count if predicted_count else 0)
        recalls.append(true_positive / actual_count if actual_count else 0)
        f1s.append(2 * true_positive / (actual_count + predicted_count) if actual_count + predicted_count else 0)
    close(metric["accuracy"], float((y == predicted).mean()))
    close(metric["macro_f1"], statistics.mean(f1s))
    close(metric["balanced_accuracy"], statistics.mean(recalls))
    close(metric["per_class_precision"], precisions)
    close(metric["per_class_recall"], recalls)
    close(metric["per_class_f1"], f1s)
    if not np.array_equal(metric["confusion_matrix"], confusion):
        raise AssertionError("混淆矩阵不一致")
    if not np.array_equal(metric["support"], confusion.sum(1)):
        raise AssertionError("逐类样本数不一致")


def verify(output: Path) -> dict:
    """核验完整的 12 次实验，成功后输出机器可读的 validation.json。"""
    check_metric_fixture()
    plan, summary = read_json(output / "plan.json"), read_json(output / "summary.json")
    prepared = output / "prepared"
    meta = read_json(prepared / "metadata.json")
    if summary["status"] != "complete" or len(summary["runs"]) != len(plan["runs"]):
        raise AssertionError("正式实验未全部完成")
    for name, expected in plan["source_sha256"].items():
        current = source_path(plan, name)
        if fingerprint(current) != expected or fingerprint(output / "source_snapshot" / name) != expected:
            raise AssertionError("当前源码或快照指纹不同")
    for name, expected in plan["prepared_sha256"].items():
        if fingerprint(prepared / name) != expected:
            raise AssertionError("标准化数据或元数据指纹不同")
    data_dir = Path(meta["data_dir"])
    for name, expected in meta["file_sha256"].items():
        if fingerprint(data_dir / name) != expected:
            raise AssertionError("去重数据被修改")
    original = read_json(data_dir / "summary.json")["original_file_sha256"]
    for name, expected in original.items():
        if fingerprint(data_dir.parent / "Data" / name) != expected:
            raise AssertionError("原始 CSV 被修改")
    # 重新从训练 CSV 拟合标量统计，确认缓存并未采用验证或测试数据的统计量。
    for circuit, info in meta["circuits"].items():
        raw = np.loadtxt(data_dir / f"{circuit}_INCI_train_data.csv", delimiter=",")
        mean, std = float(raw.mean()), float(raw.std())
        close(info["normalization"]["mean"], mean)
        close(info["normalization"]["std"], std)
        with np.load(prepared / f"{circuit}_train.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["x"], ((raw - mean) / std).astype(np.float32)):
                raise AssertionError("训练缓存与训练集标准化不一致")
            counts = np.bincount(archive["y"], minlength=info["classes"])
            weights = (len(archive["y"]) / (info["classes"] * counts)).astype(np.float32)
            if not np.array_equal(weights, np.asarray(info["class_weights"], dtype=np.float32)):
                raise AssertionError("类别权重不是仅由训练标签决定")
    checked = []
    for job in plan["runs"]:
        folder = output / job["id"]
        result, experiment = read_json(folder / "result.json"), read_json(folder / "experiment.json")
        if result["status"] != "complete" or not result["test_evaluated"]:
            raise AssertionError("单次实验未完成")
        if experiment["plan_sha256"] != fingerprint(output / "plan.json"):
            raise AssertionError("训练采用的计划与当前计划不同")
        if fingerprint(folder / "best.pt") != result["checkpoint_sha256"]:
            raise AssertionError("最优权重被修改")
        with (folder / "history.csv").open(encoding="utf-8", newline="") as handle:
            history = list(csv.DictReader(handle))
        scores = [float(row["val_macro_f1"]) for row in history]
        best_epoch = scores.index(max(scores)) + 1
        if result["best_epoch"] != best_epoch or result["epochs_run"] != len(history):
            raise AssertionError("最优轮次不是验证 Macro-F1 首次达到最大值的轮次")
        if len(history) < plan["training"]["max_epochs"] and len(history) - best_epoch != plan["training"]["patience"]:
            raise AssertionError("早停轮次与冻结耐心值不一致")
        for row in history:
            if not all(math.isfinite(float(value)) for value in row.values()):
                raise AssertionError("训练日志存在非有限数")
        close(result["validation"]["macro_f1"], max(scores))
        device = plan["environment"]["device"]
        set_seed(job["seed"], plan["training"]["threads"], device)
        classes = meta["circuits"][job["circuit"]]["classes"]
        model = make_model(job["model"], classes).to(device)
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        if checkpoint["best_epoch"] != best_epoch:
            raise AssertionError("权重内记录的最优轮次不一致")
        model.load_state_dict(checkpoint["state"])
        if sum(p.numel() for p in model.parameters()) != result["parameters"]:
            raise AssertionError("参数量记录不一致")
        # 同一固定训练行序重新估计 BN，应精确恢复权重中保存的统计量。
        # 验证/测试均不参与这次估计，额外检查保证所谓校准没有改变模型参数。
        calibrated = copy.deepcopy(model)
        update_bn(make_loader(prepared, job["circuit"], "train", plan["training"]["batch_size"], job["seed"],
                              shuffle=False), calibrated, device=device)
        if not all(torch.equal(a, b) for a, b in zip(model.parameters(), calibrated.parameters())):
            raise AssertionError("BN 重估改变模型参数")
        if not all(torch.equal(a, b) for a, b in zip(model.buffers(), calibrated.buffers())):
            raise AssertionError("保存的 BN 统计量不能仅由固定训练集重建")
        for split, field in (("val", "validation"), ("test", "test")):
            with np.load(prepared / f"{job['circuit']}_{split}.npz", allow_pickle=False) as archive:
                labels = archive["y"].copy()
            saved = folder / f"{split}_predictions.npz"
            verify_prediction(saved, result[field], labels, classes)
            loader = make_loader(prepared, job["circuit"], split, plan["training"]["batch_size"], job["seed"])
            restored_metrics, prediction = evaluate(model, loader, classes)
            with np.load(saved, allow_pickle=False) as archive:
                if not np.array_equal(archive["probability"], prediction["probability"]):
                    raise AssertionError("重载权重的预测概率与保存结果不同")
            close(result[field]["loss_unweighted"], restored_metrics["loss_unweighted"])
        matching = [r for r in summary["runs"] if r["id"] == job["id"]]
        if matching != [result]:
            raise AssertionError("汇总中的逐次结果不一致")
        checked.append({"id": job["id"], "best_epoch": best_epoch, "reload_predictions_exact": True})
    for group in summary["groups"]:
        selected = [r for r in summary["runs"] if r["circuit"] == group["circuit"] and r["model"] == group["model"]]
        if len(selected) != 3 or group["runs"] != 3:
            raise AssertionError("汇总遗漏训练种子")
        for split, prefix in (("validation", "val"), ("test", "test")):
            for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                values = [r[split][metric] for r in selected]
                close(group[f"{prefix}_{metric}_mean"], statistics.mean(values))
                close(group[f"{prefix}_{metric}_std"], statistics.stdev(values))
    report = {"status": "passed", "runs_checked": checked, "metric_fixture": "passed",
              "original_and_deduplicated_csv_unchanged": True, "source_and_cache_hashes_match": True,
              "train_only_normalization_and_class_weights": True, "independent_metrics_and_aggregates": True}
    report["bn_statistics_rebuilt_from_training_exactly"] = True
    if plan["environment"]["device"] == "cuda":
        from .run import check_fixed_inputs
        check_fixed_inputs(prepared, meta)
        report["same_inputs_as_local_baselines_and_server_ccsae"] = True
    write_json(output / "validation.json", report)
    return report


def main():
    """metric-check 可在训练前运行；正式核验需要所有实验完成。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-check", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "artifacts" / "baseline_v2")
    args = parser.parse_args()
    if args.metric_check:
        check_metric_fixture()
        print("手算指标案例通过", flush=True)
    else:
        result = verify(args.output_dir)
        print(f"核验通过：{len(result['runs_checked'])} 次实验，全部权重重载预测一致", flush=True)


if __name__ == "__main__":
    main()
