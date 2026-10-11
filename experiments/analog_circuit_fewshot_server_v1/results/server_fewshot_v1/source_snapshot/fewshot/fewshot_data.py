"""在既有 Train 中分层抽取 K-shot，Val/Test 的原始行和标签保持不变。

每个电路、抽样种子共用一组类内随机排列，10/20/40 为其嵌套前缀。
归一化只拟合抽中的训练波形；未抽中的 Train 不进入任何训练或 BN 重估。
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ccsae.data import DataSplits
from ccsae.fixed_data import fingerprint, load_fixed, read_json, write_json


CASES = {"skl": "SKLCircuit", "abl": "ABLCircuit"}
SHOTS = (10, 20, 40)
SEEDS = (42, 43, 44)


def select_indices(labels: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    """每类独立打乱后取前 K 条，最后按原 Train 行序排列，便于一致的 BN 重估。"""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("类别编号应连续且从零开始")
    shuffled = [rng.permutation(np.flatnonzero(labels == c)) for c in classes]
    if min(map(len, shuffled)) < max(SHOTS):
        raise ValueError("原训练集中某类不足 40 条，不能执行既定方案")
    return {k: np.sort(np.concatenate([rows[:k] for rows in shuffled])) for k in SHOTS}


def normalize(train: np.ndarray, splits: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """用所选训练集全部采样值计算一个均值/标准差，保留样本间幅值差异。"""
    mean, std = float(train.mean()), float(train.std())
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError("少样本训练集归一化统计非法")
    arrays = {s: ((x - mean) / std).astype(np.float32) for s, x in splits.items()}
    if not all(np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("归一化后出现非有限值")
    return arrays, {"mean": mean, "std": std, "fitted_split": "selected_train_only",
                    "scope": "scalar_over_all_selected_training_values", "sample_count": len(train)}


def cache_folder(output: Path, case: str, seed: int, k: int) -> Path:
    """同一目录供三个模型共用，避免各模型重新随机抽样。"""
    return output / "prepared" / case / f"seed_{seed}" / f"k_{k}"


def prepare(data: Path, reference: Path, output: Path) -> dict:
    """检查冻结数据，再生成 18 份少样本缓存和逐行来源清单。

    原全训练缓存只用于核对既有数据，少样本标准化始终重新拟合所选 Train。
    原始 CSV 不写入；Val/Test 可以读取以检查完整性，但此处不产生模型预测。
    """
    metadata = {}
    for case, circuit in CASES.items():
        fixed = load_fixed(data, reference, case)
        raw, labels = {}, {}
        info = fixed.report
        for split in ("train", "val", "test"):
            raw[split] = np.loadtxt(data / f"{circuit}_INCI_{split}_data.csv", delimiter=",", ndmin=2)
            labels[split] = getattr(fixed, split).tensors[1].numpy()
            stats = info["normalization"]
            old = ((raw[split] - stats["mean"]) / stats["std"]).astype(np.float32)
            if not np.array_equal(old, getattr(fixed, split).tensors[0].numpy()):
                raise AssertionError("原始行顺序与此前完整训练缓存不一致")
        for seed in SEEDS:
            indices = select_indices(labels["train"], seed)
            for k in SHOTS:
                folder = cache_folder(output, case, seed, k)
                folder.mkdir(parents=True, exist_ok=False)
                rows = indices[k]
                values = {"train": raw["train"][rows], "val": raw["val"], "test": raw["test"]}
                arrays, stats = normalize(values["train"], values)
                ys = {"train": labels["train"][rows], "val": labels["val"], "test": labels["test"]}
                # 检查转换为 float32 后也没有新的重复波形，防止数值转换破坏隔离。
                hashes = [[hashlib.sha256(row.tobytes()).digest() for row in arrays[s]] for s in arrays]
                flattened = [h for group in hashes for h in group]
                if len(set(flattened)) != len(flattened):
                    raise AssertionError("少样本归一化后出现集合内或跨集合重复")
                for split in arrays:
                    np.savez_compressed(folder / f"{circuit}_{split}.npz", x=arrays[split], y=ys[split],
                                        original_row_index=rows if split == "train" else np.arange(len(ys[split])))
                with (folder / "train_selection.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["source_split", "source_row_1based", "class_id", "waveform_sha256"])
                    for row in rows:
                        waveform = raw["train"][row].copy()
                        waveform[waveform == 0] = 0
                        writer.writerow(["train", int(row) + 1, int(labels["train"][row]),
                                         hashlib.sha256(waveform.tobytes()).hexdigest()])
                classes = info["classes"]
                counts = np.bincount(ys["train"], minlength=classes).tolist()
                if counts != [k] * classes:
                    raise AssertionError("抽样后不是每类恰好 K 条")
                circuit_meta = {"classes": classes, "normalization": stats, "train_class_counts": counts,
                                "class_weights": [1.0] * classes,
                                "split_counts": {s: len(ys[s]) for s in ys}, "raw_and_float32_duplicates": 0}
                meta = {"status": "passed", "case": case, "k": k, "sampling_seed": seed,
                        "original_split_seed": info["split_seed"], "selected_train_indices": rows.tolist(),
                        "cross_split_duplicate_groups": {"train_val": 0, "train_test": 0, "val_test": 0},
                        "selection_rule": "numpy_default_rng(seed), class permutation, nested prefix, source row order",
                        "circuits": {circuit: circuit_meta}, "original_protocol": info,
                        "file_sha256": {p.name: fingerprint(p) for p in sorted(folder.iterdir())}}
                write_json(folder / "metadata.json", meta)
                metadata[f"{case}_seed{seed}_k{k}"] = meta
    return metadata


def load_splits(folder: Path, case: str) -> DataSplits:
    """将同一缓存转换为 CCSAE 的 [N,1000] 数据；基线自行加单通道维度。"""
    circuit = CASES[case]
    meta = read_json(folder / "metadata.json")
    datasets = {}
    for split in ("train", "val", "test"):
        with np.load(folder / f"{circuit}_{split}.npz", allow_pickle=False) as archive:
            datasets[split] = TensorDataset(torch.from_numpy(archive["x"].copy()), torch.from_numpy(archive["y"].copy()))
    return DataSplits(datasets["train"], datasets["val"], datasets["test"], meta)
