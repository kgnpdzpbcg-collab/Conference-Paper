"""复核服务器少样本实验的文本归档；不训练、不推理、不改写原始证据。

检查服务器文件指纹、54组已存混淆矩阵和18组少样本三种子统计。
权重重载/数据追溯使用原训练阶段的核验记录，本脚本不声称重新执行这些步骤。
"""
from pathlib import Path
import hashlib
import json
import math
import statistics

BASE = Path(__file__).resolve().parent / "results/server_fewshot_v1"


def read(path):
    """读取 UTF-8 JSON；缺失或格式错误直接报错，防止跳过失败记录。"""
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(matrix):
    """由混淆矩阵独立复算 Accuracy、Macro-F1 和按类召回均值。"""
    classes = len(matrix)
    assert all(len(row) == classes for row in matrix)
    actual = [sum(row) for row in matrix]
    selected = [sum(matrix[i][j] for i in range(classes)) for j in range(classes)]
    tp = [matrix[i][i] for i in range(classes)]
    assert all(n > 0 for n in actual)
    f1 = [2*tp[i]/(actual[i]+selected[i]) for i in range(classes)]
    return {"accuracy":sum(tp)/sum(actual), "macro_f1":statistics.mean(f1),
            "balanced_accuracy":statistics.mean(tp[i]/actual[i] for i in range(classes))}


def main():
    """先检查原件一致性，再复算指标和三种子均值/样本标准差。"""
    manifest = read(BASE / "SERVER_SOURCE_MANIFEST.json")
    for name,record in manifest["files"].items():
        p = BASE/name
        assert p.stat().st_size == record["bytes"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == record["sha256"]
    assert read(BASE/"training_validation.json")["status"] == "passed"
    assert read(BASE/"validation.json")["status"] == "passed"
    summary = read(BASE/"summary.json")
    runs = [r for r in summary["runs"] if r["k"] in (10,20,40)]
    expected = {(c,m,k,s) for c in ("skl","abl") for m in ("cnn1d","resnet1d","ccsae_v1")
                for k in (10,20,40) for s in (42,43,44)}
    assert len(runs) == 54
    assert {(r["case"],r["model"],r["k"],r["seed"]) for r in runs} == expected
    for r in runs:
        saved = read(BASE/r["id"]/"result.json")
        assert saved["test_evaluated"] is True
        for split in ("validation","test"):
            assert saved[split] == r[split]
            matrix = saved[split]["confusion_matrix"]
            assert len(matrix) == (13 if r["case"] == "skl" else 11)
            assert sum(map(sum,matrix)) == (570 if r["case"] == "skl" else 465)
            for name,value in metrics(matrix).items():
                assert math.isclose(value,saved[split][name],rel_tol=0,abs_tol=1e-12)
    groups = [g for g in summary["groups"] if g["k"] in (10,20,40)]
    assert len(groups) == 18
    for g in groups:
        selected = [r for r in runs if (r["case"],r["model"],r["k"]) == (g["case"],g["model"],g["k"])]
        assert {r["seed"] for r in selected} == {42,43,44}
        for split in ("validation","test"):
            for name in ("accuracy","macro_f1","balanced_accuracy"):
                values = [r[split][name] for r in selected]
                assert math.isclose(statistics.mean(values),g[f"{split}_{name}_mean"],rel_tol=0,abs_tol=1e-12)
                assert math.isclose(statistics.stdev(values),g[f"{split}_{name}_std"],rel_tol=0,abs_tol=1e-12)
    print(json.dumps({"status":"passed","server_files_byte_exact":len(manifest["files"]),
        "fewshot_runs":54,"fewshot_groups":18,"all_saved_val_and_test_confusion_metrics_verified":True,
        "three_seed_statistics_independently_verified":True,"original_training_and_test_verification_reused":True,
        "no_training_or_model_inference_in_this_publication":True},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
