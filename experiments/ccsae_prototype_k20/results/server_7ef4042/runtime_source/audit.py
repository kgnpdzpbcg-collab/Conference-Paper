"""训练前独立检查数据、上游源码与旧实验指纹，只读原数据和旧结果。"""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from ccsae.data import load_raw
from ccsae.fewshot_data import make_fewshot_split
from common import dump, file_sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
OUT = ROOT / "server_experiment/provenance"


def wave_hash(row):
    """按新分支实际采用的 float32 波形字节比较，避免文本格式差异影响去重判断。"""
    return hashlib.sha256(np.ascontiguousarray(row, dtype=np.float32).tobytes()).hexdigest()


def fingerprint(items):
    return hashlib.sha256("\n".join(sorted(items)).encode("ascii")).hexdigest()


def existing_records():
    """记录原去重数据和所有已完成实验输出；正式实验结束后再次检查，防止覆盖旧结果。"""
    groups = ["Data_deduplicated", "analog-circuit-fewshot/outputs/server_fewshot_v1",
              "Conference-Paper-ccsae-v1/outputs/server_v1_signed_v1",
              "analog-circuit-baselines/artifacts/server_gpu_v1",
              "analog-circuit-crossdomain/outputs/binary_direct/server_v1",
              "analog-circuit-crossdomain/outputs/multiclass_transfer/server_v1"]
    return {group: [{"path": str(p.relative_to(BASE)), "bytes": p.stat().st_size,
                     "sha256": file_sha(p)} for p in sorted((BASE / group).rglob("*")) if p.is_file()]
            for group in groups}


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert commit == "7ef4042fe18275f2f366c06c5aa19aa817c45258"
    # 所有受版本管理的文件都与固定提交一致，仅 pyproject.toml 的服务器依赖设置例外。
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    upstream = {}
    for name in tracked:
        original = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        if name != "pyproject.toml":
            assert (ROOT / name).read_bytes() == original, f"上游文件意外变更：{name}"
        upstream[name] = hashlib.sha256(original).hexdigest()
    dump(OUT / "upstream_sources.json", {"commit": commit, "files": upstream,
         "code_unmodified": True, "environment_only_change": "pyproject.toml/uv.lock/.python-version"})
    assert file_sha(BASE / "Data.zip") == "11623819dcbb9287ff7ffea4d763d0a6fe3dd77d74c61bfe581bd8ea792b4e49"
    case_reports = {}
    for case, prefix, expected_rows in [("skl", "SKLCircuit", 3800), ("abl", "ABLCircuit", 3100)]:
        tr_x, tr_y, te_x, te_y = load_raw(BASE / "Data.zip", case)
        all_x = np.concatenate([tr_x, te_x])
        all_y = np.concatenate([tr_y, te_y])
        # 独立重建首条保留顺序及三个池，检查代码声明的指纹和历史划分的实际交集。
        keys = [wave_hash(x) for x in all_x]
        keep = {}
        for i, key in enumerate(keys):
            if key in keep:
                assert all_y[i] == all_y[keep[key]], "重复波形标签冲突"
            else:
                keep[key] = i
        assert len(keep) == expected_rows
        unique_keys = list(keep)
        labels = np.array([all_y[keep[key]] for key in unique_keys])
        pools = {name: set() for name in ("train", "val", "test")}
        rng = np.random.default_rng(20261009)
        for c in np.unique(labels):
            ids = np.flatnonzero(labels == c)
            rng.shuffle(ids)
            ntr, nval = round(len(ids) * .7), round(len(ids) * .15)
            for name, chosen in [("train", ids[:ntr]), ("val", ids[ntr:ntr+nval]), ("test", ids[ntr+nval:])]:
                pools[name].update(unique_keys[i] for i in chosen)
        assert not pools["train"] & pools["val"]
        assert not pools["train"] & pools["test"]
        assert not pools["val"] & pools["test"]
        old = {}
        for name in pools:
            x = np.loadtxt(BASE / "Data_deduplicated" / f"{prefix}_INCI_{name}_data.csv", delimiter=",", dtype=np.float32)
            y = np.loadtxt(BASE / "Data_deduplicated" / f"{prefix}_INCI_{name}_label.csv", delimiter=",", dtype=np.float32).argmax(1)
            old[name] = {wave_hash(row) for row in x}
            assert len(old[name]) == len(x)
            assert all(all_y[keep[wave_hash(row)]] == target for row, target in zip(x, y))
        assert set.union(*old.values()) == set(keep), "两协议的唯一波形集合不一致"
        selections = {}
        fixed_val = None
        for seed in (42, 43, 44):
            split = make_fewshot_split(BASE / "Data.zip", case, k=20, sample_seed=seed)
            a = split.audit
            assert a["unique_rows"] == expected_rows
            assert split.train_y.bincount().tolist() == [20] * a["classes"]
            assert split.val_y.bincount().tolist() == [5] * a["classes"]
            for name in pools:
                assert a[f"{name}_pool_sha256"] == fingerprint(pools[name])
            train_keys = sum(a["selected_train_hashes_by_class"].values(), [])
            val_keys = sum(a["selected_val_hashes_by_class"].values(), [])
            assert set(train_keys) <= pools["train"] and set(val_keys) <= pools["val"]
            assert not set(train_keys) & set(val_keys)
            if fixed_val is None:
                fixed_val = val_keys
            assert fixed_val == val_keys, "三个种子的验证样本改变"
            raw = np.array([all_x[keep[key]] for key in train_keys], dtype=np.float64)
            assert raw.mean() == a["train_scaler_mean"] and raw.std() == a["train_scaler_std"]
            normalized = ((raw - raw.mean()) / raw.std()).astype(np.float32)
            assert np.array_equal(normalized, split.train_x.numpy())
            selections[str(seed)] = a
        case_reports[case] = {"unique_rows": len(keep), "pool_sizes": {n: len(v) for n,v in pools.items()},
            "old_pool_sizes": {n: len(v) for n,v in old.items()},
            "old_new_intersections": {new: {prior: len(pools[new] & old[prior]) for prior in old} for new in pools},
            "old_split_identical": all(pools[name] == old[name] for name in pools), "selections": selections}
    dump(OUT / "data_audit.json", {"status": "passed", "test_inference": False, "cases": case_reports})
    dump(OUT / "existing_artifacts_before.json", existing_records())
    print(json.dumps({case: {k:v for k,v in report.items() if k != "selections"} for case,report in case_reports.items()}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
