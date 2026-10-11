"""仅用Python标准库复核服务器归档、预测指标和统计；不加载模型或训练。

输入：完整非权重归档的解压目录及tar.gz；输出：LOCAL_VERIFICATION.json。
权重未传到本地，其状态由服务器独立重载验证和WEIGHTS_MANIFEST证明。
"""
from pathlib import Path
from itertools import product
from collections import Counter
import argparse, array, ast, csv, hashlib, json, math, statistics, struct, sys, zipfile


def close(left, right):
    """容许浮点求和次序的微小差异，不能容许实际分数偏移。"""
    assert math.isclose(left, right, rel_tol=0, abs_tol=1e-10), (left, right)


def npy_doubles(data):
    """读取已归档的20000个bootstrap均值；无需安装或导入NumPy。"""
    assert data[:6] == b"\x93NUMPY"
    version = data[6]
    assert version in (1, 2)
    length_bytes = 2 if version == 1 else 4
    length = struct.unpack("<H" if version == 1 else "<I", data[8:8+length_bytes])[0]
    start = 8 + length_bytes
    header = ast.literal_eval(data[start:start+length].decode("latin1"))
    assert header == {"descr": "<f8", "fortran_order": False, "shape": (20000,)}
    values = array.array("d")
    values.frombytes(data[start+length:])
    if sys.byteorder != "little":
        values.byteswap()
    assert len(values) == 20000
    return sorted(values)


def quantile(sorted_values, probability):
    """与服务器percentile默认线性插值规则一致。"""
    position = (len(sorted_values)-1) * probability
    low = int(position)
    fraction = position-low
    return sorted_values[low]*(1-fraction) + sorted_values[min(low+1, len(sorted_values)-1)]*fraction


def main(base, archive):
    """逐文件校验后，从逐条预测重新计算90份指标、9组汇总和6组比较。"""
    def read(name):
        return json.loads((base/name).read_text(encoding="utf-8"))
    repo = base.parents[2]
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive_sha == "4955d8a98d7c2376db3e1f7d127fb62cb12c79a6b4a847706d0ea4f29b199bf7"
    manifest = read("TRANSFER_MANIFEST.json")
    for name, item in manifest["files"].items():
        data = (repo/name).read_bytes()
        assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"]
    evaluation = read("TEST_EVALUATION.json")
    rows = evaluation["results"]
    assert len(rows) == 90
    actual = {}
    truth_reference = None
    for row in rows:
        k, seed, condition = row["k"], row["seed"], row["condition"]
        prediction = read(f"target/k{k}_seed{seed}/{condition}/test_predictions.json")
        truth, predicted = prediction["truth"], prediction["predicted"]
        if truth_reference is None:
            truth_reference = truth
        assert truth == truth_reference and len(truth) == len(predicted) == 465
        assert Counter(truth) == {0:15, **{i:45 for i in range(1,11)}}
        cm = [[0]*11 for _ in range(11)]
        for label, guess in zip(truth, predicted):
            assert 0 <= label < 11 and 0 <= guess < 11
            cm[label][guess] += 1
        assert cm == row["confusion_matrix"]
        row_sum = [sum(line) for line in cm]
        col_sum = [sum(cm[i][j] for i in range(11)) for j in range(11)]
        f1 = [2*cm[i][i]/(row_sum[i]+col_sum[i]) for i in range(11)]
        for left, right in zip(f1, row["per_class_f1"]):
            close(left, right)
        metrics = {"accuracy":sum(cm[i][i] for i in range(11))/465,
                   "macro_f1":statistics.mean(f1),
                   "balanced_accuracy":statistics.mean(cm[i][i]/row_sum[i] for i in range(11))}
        for name, value in metrics.items():
            close(value, row["metrics"][name])
        actual[k,seed,condition] = row
    assert len(actual) == 90
    with (base/"PER_SEED_RESULTS.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 90
    for row in csv_rows:
        original = actual[int(row["k"]),int(row["seed"]),row["condition"]]
        assert row["checkpoint_sha256"] == original["checkpoint_sha256"]
        for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
            close(float(row["test_"+metric]), original["metrics"][metric])
    summary = read("SUMMARY.json")
    for item in summary["aggregates"]:
        values = [actual[item["k"],seed,item["condition"]]["metrics"]["macro_f1"]*100 for seed in range(42,52)]
        close(statistics.mean(values), item["test_macro_f1_mean_percent"])
        close(statistics.stdev(values), item["test_macro_f1_sample_sd_percent"])
    p_values = []
    with zipfile.ZipFile(base/"bootstrap_means.npz") as bootstrap:
        for item in summary["contrasts"]:
            deltas = [(actual[item["k"],seed,item["condition"]]["metrics"]["macro_f1"]-
                       actual[item["k"],seed,"target_only"]["metrics"]["macro_f1"])*100 for seed in range(42,52)]
            close(statistics.mean(deltas), item["paired_delta_mean_pp"])
            close(statistics.stdev(deltas), item["paired_delta_sample_sd_pp"])
            assert sum(delta>0 for delta in deltas) == item["wins"]
            for seed, delta in zip(range(42,52), deltas):
                close(delta, item["per_seed_delta_pp"][str(seed)])
            # 全部1024种配对符号组合，不把Test样本当成独立训练重复。
            observed = abs(statistics.mean(deltas))
            count = sum(abs(statistics.mean(d*s for d,s in zip(deltas, signs))) >= observed-1e-12
                        for signs in product((-1,1), repeat=10))
            assert count == item["null_tail_count"]
            close(count/1024, item["p_raw_two_sided"])
            p_values.append(count/1024)
            values = npy_doubles(bootstrap.read(f"k{item['k']}_{item['comparison']}.npy"))
            for probability, saved in zip((.025,.975), item["paired_bootstrap_ci95_pp"]):
                close(quantile(values, probability), saved)
    # 同一六项检验族采用Holm，不单独用原始p值宣称提升或下降。
    order = sorted(range(6), key=lambda i:p_values[i])
    maximum = 0
    for rank, index in enumerate(order):
        maximum = max(maximum, min(1, (6-rank)*p_values[index]))
        close(maximum, summary["contrasts"][index]["p_holm_two_sided"])
    result = {"status":"passed", "archive_sha256":archive_sha, "server_manifest_files_verified":len(manifest["files"]),
              "prediction_metrics_verified":90, "aggregate_mean_sample_sd_verified":9, "paired_contrasts_verified":6,
              "exact_sign_flip_and_holm_verified":True, "bootstrap_ci_from_saved_20000_arrays_verified":True,
              "local_model_inference_or_training":False, "local_packages_installed":False,
              "weights_not_transferred":"server independently reloaded and verified all best weights"}
    (base/"LOCAL_VERIFICATION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    main(args.base.resolve(), args.archive.resolve())
