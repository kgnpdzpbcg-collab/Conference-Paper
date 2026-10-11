"""加载冻结的去重划分，检查指纹并只用训练集拟合全局幅值标准化。

保持每条波形的相对幅值差异，不逐样本归一化、不裁剪、不做数据增强。
原始 CSV 只读，缓存是为重复训练生成的 float32 张量，不改变数据划分。
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def fingerprint(path: Path) -> str:
    """文件字节指纹，用于将实验绑定到具体的数据、配置和源码版本。"""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    """按 UTF-8 保存可检查的实验元数据。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(data_dir: Path, output_dir: Path) -> dict:
    """核验全部文件和逐样本清单，保存归一化统计及三个集合的缓存。

    float64 原记录及归一化后的 float32 记录都检查跨集合精确重复。
    测试集仅参与数据完整性检查，训练阶段不会访问其指标或用于选模。
    """
    gate = json.loads((data_dir / "validation.json").read_text(encoding="utf-8"))
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    if gate["status"] != "passed":
        raise ValueError("去重数据核验未通过")
    hashes = {name: fingerprint(data_dir / name) for name in gate["output_file_sha256"]}
    if hashes != gate["output_file_sha256"]:
        raise ValueError("去重 CSV 与冻结指纹不同，禁止训练")
    output_dir.mkdir(parents=True, exist_ok=False)
    circuits = {}
    for circuit in summary["circuits"]:
        name = circuit["circuit"]
        with (data_dir / f"{name}_manifest.csv").open(encoding="utf-8", newline="") as handle:
            manifest = list(csv.DictReader(handle))
        arrays, labels = {}, {}
        for split in ("train", "val", "test"):
            x = np.loadtxt(data_dir / f"{name}_INCI_{split}_data.csv", delimiter=",", ndmin=2)
            hot = np.loadtxt(data_dir / f"{name}_INCI_{split}_label.csv", delimiter=",", ndmin=2)
            if x.shape != (circuit["split_counts"][split], 1000) or hot.shape != (len(x), len(circuit["per_class"])):
                raise ValueError(f"{name}/{split}：维度与去重协议不同")
            if not np.isfinite(x).all() or not (np.isin(hot, [0, 1]).all() and (hot.sum(1) == 1).all()):
                raise ValueError(f"{name}/{split}：非法波形或标签")
            y = hot.argmax(1).astype(np.int64)
            entries = sorted((r for r in manifest if r["split"] == split), key=lambda r: int(r["row_1based"]))
            if len(entries) != len(x):
                raise ValueError("样本清单长度不同")
            x[x == 0] = 0
            for index, entry in enumerate(entries):
                if (int(entry["row_1based"]) != index + 1 or int(entry["class_id"]) != y[index]
                        or hashlib.sha256(x[index].tobytes()).hexdigest() != entry["waveform_sha256"]):
                    raise ValueError("原始波形与逐样本清单不符")
            arrays[split], labels[split] = x, y
        raw = np.concatenate(list(arrays.values()))
        if len(np.unique(raw, axis=0)) != len(raw):
            raise ValueError("仍存在原始波形重复")
        mean, std = float(arrays["train"].mean()), float(arrays["train"].std())
        if std <= 0:
            raise ValueError("训练波形方差为零")
        prepared = {s: ((x - mean) / std).astype(np.float32) for s, x in arrays.items()}
        converted = np.concatenate(list(prepared.values()))
        if len(np.unique(converted, axis=0)) != len(converted):
            raise ValueError("标准化后 float32 出现新的精确重复")
        classes = hot.shape[1]
        counts = np.bincount(labels["train"], minlength=classes)
        if (counts == 0).any():
            raise ValueError("训练集存在空类别，不能计算逆频数权重")
        # 逆频数权重仅由训练标签决定，减轻去重后 C0 样本较少的影响。
        weights = (len(labels["train"]) / (classes * counts)).astype(np.float32)
        for split in prepared:
            np.savez_compressed(output_dir / f"{name}_{split}.npz", x=prepared[split], y=labels[split])
        circuits[name] = {"classes": classes, "normalization": {"mean": mean, "std": std,
                          "fitted_split": "train", "scope": "scalar_over_all_training_values"},
                          "class_weights": weights.tolist(), "train_class_counts": counts.tolist(),
                          "split_counts": circuit["split_counts"], "raw_and_float32_duplicates": 0,
                          "manifest_sha256": hashes[f"{name}_manifest.csv"]}
    result = {"status": "passed", "data_dir": str(data_dir.resolve()), "file_sha256": hashes,
              "circuits": circuits}
    write_json(output_dir / "metadata.json", result)
    return result


def make_loader(folder: Path, circuit: str, split: str, batch_size: int, seed: int,
                shuffle: bool | None = None) -> DataLoader:
    """读取 [N,1,1000] 输入，默认仅训练打乱；BN 重估显式使用训练集固定行序。"""
    with np.load(folder / f"{circuit}_{split}.npz", allow_pickle=False) as data:
        x, y = torch.from_numpy(data["x"][:, None, :].copy()), torch.from_numpy(data["y"].copy())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(x, y, torch.arange(len(y))), batch_size=batch_size,
                      shuffle=split == "train" if shuffle is None else shuffle,
                      generator=generator, num_workers=0)
