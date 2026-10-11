"""独立核验固定划分 CCSAE 实验，避免仅依赖训练程序自己的汇总。

使用逐类布尔计数和标准库 statistics 重算指标，检查两阶段选模规则、
解码器冻结、来源指纹及保存权重重载的逐行概率。核验不用于修改模型配置。
"""

import argparse
from pathlib import Path
import statistics

import numpy as np
import torch

from .fixed_data import fingerprint, load_fixed, read_json, write_json
from .fixed_experiment import PROJECT, check_frozen, checkpoint_model, configure, predict


def close(actual, expected) -> None:
    """只允许浮点加和顺序带来的微小差异，整数计数另做严格相等核验。"""
    if not np.allclose(actual, expected, rtol=0, atol=1e-12):
        raise AssertionError(f"指标不一致：{actual} / {expected}")


def verify_predictions(path: Path, labels: np.ndarray, metric: dict, classes: int) -> None:
    """从逐行预测独立构造混淆矩阵和逐类指标，不复用 sklearn 指标函数。"""
    with np.load(path, allow_pickle=False) as archive:
        y, pred, probability, rows = [archive[k].copy() for k in ("label", "prediction", "probability", "row_index")]
    if not np.array_equal(y, labels) or not np.array_equal(rows, np.arange(len(y))):
        raise AssertionError("预测记录与固定数据的标签/行顺序不同")
    if probability.shape != (len(y), classes) or not np.isfinite(probability).all():
        raise AssertionError("概率形状错误或含 NaN/Inf")
    if (probability < 0).any() or not np.allclose(probability.sum(1), 1, rtol=0, atol=2e-6):
        raise AssertionError("概率分布不合法")
    if not np.array_equal(pred, probability.argmax(1)):
        raise AssertionError("类别与最大概率不符")
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for target, output in zip(y, pred):
        confusion[target, output] += 1
    precisions, recalls, f1s, support = [], [], [], []
    for category in range(classes):
        tp = int(((y == category) & (pred == category)).sum())
        actual_count, predicted_count = int((y == category).sum()), int((pred == category).sum())
        support.append(actual_count)
        precisions.append(tp / predicted_count if predicted_count else 0)
        recalls.append(tp / actual_count if actual_count else 0)
        f1s.append(2 * tp / (actual_count + predicted_count) if actual_count + predicted_count else 0)
    close(metric["accuracy"], float((y == pred).mean()))
    close(metric["macro_f1"], statistics.mean(f1s))
    close(metric["balanced_accuracy"], statistics.mean(recalls))
    close(metric["per_class_precision"], precisions)
    close(metric["per_class_recall"], recalls)
    close(metric["per_class_f1"], f1s)
    if not np.array_equal(metric["support"], support) or not np.array_equal(metric["confusion_matrix"], confusion):
        raise AssertionError("混淆矩阵或类别样本数不符")


def verify(output: Path) -> dict:
    """要求全部正式结果已生成，再独立核对六次训练及两个汇总组。"""
    plan, summary = read_json(output / "plan.json"), read_json(output / "summary.json")
    if plan["smoke_only"] or summary["status"] != "complete_validation" or len(summary["runs"]) != 6:
        raise ValueError("尚未完成六次正式实验")
    check_frozen(plan, output)
    splits = {case: load_fixed(Path(plan["data_dir"]), Path(plan["baseline_dir"]), case) for case in ("skl", "abl")}
    checked = []
    for job in plan["runs"]:
        configure(job["seed"], plan["training"]["threads"], plan["training"]["device"])
        folder = output / job["case"] / f"seed_{job['seed']}"
        result = read_json(folder / "summary.json")
        if result["status"] != "complete_validation" or result["test_evaluated"] or "test" in result:
            raise AssertionError("单次运行没有按验证集范围完成")
        if result["experiment_plan_sha256"] != fingerprint(output / "plan.json"):
            raise AssertionError("训练使用的计划与当前计划不同")
        history = read_json(folder / "history.json")
        first, second = [r for r in history if r["stage"] == 1], [r for r in history if r["stage"] == 2]
        if len(first) != plan["training"]["feature_epochs"] or len(second) != plan["training"]["classification_epochs"]:
            raise AssertionError("两阶段没有完成冻结的全部轮数")
        for stage in (first, second):
            if [r["epoch"] for r in stage] != list(range(1, len(stage) + 1)):
                raise AssertionError("逐轮日志不连续")
        if not all(np.isfinite(value) for row in history for value in row.values()):
            raise AssertionError("训练或验证日志存在非有限值")
        expected_first = min(first, key=lambda r: r["loss"])["epoch"]
        expected_second = max(second, key=lambda r: r["macro_f1"])["epoch"]
        for filename, field in (("stage1_best.pt", "stage1_sha256"), ("stage2_best.pt", "stage2_sha256")):
            if fingerprint(folder / filename) != result[field]:
                raise AssertionError("权重指纹不同")
        stage1 = torch.load(folder / "stage1_best.pt", map_location="cpu", weights_only=True)
        model, stage2 = checkpoint_model(folder / "stage2_best.pt", plan["training"]["device"])
        if stage1["epoch"] != expected_first or stage2["epoch"] != expected_second:
            raise AssertionError("权重未按验证指标选择最优轮次")
        close(result["validation"]["macro_f1"], max(r["macro_f1"] for r in second))
        # 分类阶段不使用解码器；其参数应与特征阶段选出的权重逐位相同。
        for name, tensor in stage1["state_dict"].items():
            if name.startswith("decoder.") and not torch.equal(tensor, stage2["state_dict"][name]):
                raise AssertionError("分类阶段更新了解码器")
        total = sum(p.numel() for p in model.parameters())
        inference = total - sum(p.numel() for p in model.decoder.parameters())
        if result["parameters_total"] != total or result["parameters_inference"] != inference:
            raise AssertionError("参数量记录不同")
        for split, field in (("val", "validation"),):
            dataset = getattr(splits[job["case"]], split)
            saved = folder / f"{split}_predictions.npz"
            verify_predictions(saved, dataset.tensors[1].numpy(), result[field], model.cfg.classes)
            metric, prediction = predict(model, dataset, plan["training"]["batch_size"])
            with np.load(saved, allow_pickle=False) as archive:
                if not np.array_equal(archive["probability"], prediction["probability"]):
                    raise AssertionError("重载权重后的概率不同")
            close(result[field]["loss_unweighted"], metric["loss_unweighted"])
        if [r for r in summary["runs"] if r["id"] == job["id"]] != [result]:
            raise AssertionError("汇总中的逐次结果不同")
        checked.append({"id": job["id"], "stage1_best_epoch": expected_first,
                        "stage2_best_epoch": expected_second, "reload_predictions_exact": True})
    for group in summary["groups"]:
        selected = [r for r in summary["runs"] if r["case"] == group["case"]]
        if len(selected) != 3 or group["runs"] != 3:
            raise AssertionError("汇总缺少训练种子")
        for split, prefix in (("validation", "val"),):
            for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                values = [r[split][metric] for r in selected]
                close(group[f"{prefix}_{metric}_mean"], statistics.mean(values))
                close(group[f"{prefix}_{metric}_std"], statistics.stdev(values))
    if any(output.rglob("test_predictions.npz")) or summary["test_evaluated"]:
        raise AssertionError("本轮不允许产生新的测试预测")
    # V1 必须记录两个损失分量；允许 FP32 中先合成总损失产生的舍入误差。
    coefficient = plan["training"]["scl_weights"]
    for job in plan["runs"]:
        history = read_json(output / job["case"] / f"seed_{job['seed']}" / "history.json")
        for row in [r for r in history if r["stage"] == 1]:
            for total, recon, scl in (("train_loss", "train_reconstruction_loss", "train_scl_loss"),
                                      ("loss", "reconstruction_loss", "scl_loss")):
                expected = row[recon] + coefficient[job["case"]] * row[scl]
                if not np.isclose(row[total], expected, rtol=2e-6, atol=1e-7):
                    raise AssertionError("V1 损失分量与总损失不符")
    original_hashes = read_json(Path(plan["data_dir"]) / "summary.json")["original_file_sha256"]
    for name, expected in original_hashes.items():
        if fingerprint(Path(plan["data_dir"]).parent / "Data" / name) != expected:
            raise AssertionError("原始数据发生变化")
    report = {"status": "passed", "runs": checked, "independent_metrics_and_aggregates": True,
              "decoder_frozen_in_stage2": True, "source_and_data_hashes_match": True,
              "original_csv_unchanged": True, "same_inputs_as_cnn_and_resnet": True,
              "test_evaluated": False, "v1_loss_components_verified": True}
    write_json(output / "validation.json", report)
    return report


def main():
    """验证汇总生成后调用；核验失败不会修改或覆盖训练权重。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs" / "fixed_v1")
    args = parser.parse_args()
    report = verify(args.output_dir)
    print(f"核验通过：{len(report['runs'])} 次实验，指标与重载预测全部一致", flush=True)


if __name__ == "__main__":
    main()
