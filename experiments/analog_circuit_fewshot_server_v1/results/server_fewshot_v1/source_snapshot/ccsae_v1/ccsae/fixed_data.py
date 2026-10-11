"""读取已经完成审计的固定去重划分，不重新抽样、不重新拟合归一化。

输入为 Data_deduplicated 和 CNN/ResNet 的 baseline_v2 目录。先核验 CSV、
标准化缓存和来源清单指纹，再返回 CCSAE 要求的 [N,1000] TensorDataset。
这样训练种子仅影响模型与批次顺序，不影响三种网络共用的数据划分。
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .data import DataSplits, _row_hashes


def fingerprint(path: Path) -> str:
    """对文件字节计算 SHA256，将训练绑定到已经验证的数据版本。"""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    """按 UTF-8 读取元数据；缺失或解析失败直接报错，不更换数据源。"""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    """保存实验配置和记录，供逐行核验与复跑使用。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_fixed(root: Path, baseline: Path, case: str) -> DataSplits:
    """核验两份已有审计结果，加载与基线逐元素相同的输入及标签。

    测试集仅在这里做结构和重复检查；训练函数不会用测试指标选模。
    不调用原 grouped 划分器，因为它会再次随机切分并保留组内重复行。
    """
    if read_json(root / "validation.json")["status"] != "passed":
        raise ValueError("去重数据审计未通过")
    if read_json(baseline / "validation.json")["status"] != "passed":
        raise ValueError("基线数据及实验核验未通过")
    plan = read_json(baseline / "plan.json")
    prepared = baseline / "prepared"
    for name, expected in plan["prepared_sha256"].items():
        if fingerprint(prepared / name) != expected:
            raise ValueError(f"基线缓存或元数据指纹变化：{name}")
    meta = read_json(prepared / "metadata.json")
    for name, expected in meta["file_sha256"].items():
        if fingerprint(root / name) != expected:
            raise ValueError(f"去重数据或来源清单指纹变化：{name}")
    circuit = {"skl": "SKLCircuit", "abl": "ABLCircuit"}[case]
    info = meta["circuits"][circuit]
    classes = info["classes"]
    datasets, hashes, counts = {}, {}, {}
    for split in ("train", "val", "test"):
        with np.load(prepared / f"{circuit}_{split}.npz", allow_pickle=False) as archive:
            x, y = archive["x"].copy(), archive["y"].copy()
        if x.shape != (info["split_counts"][split], 1000) or x.dtype != np.float32 or y.dtype != np.int64:
            raise ValueError("基线缓存形状或数据类型错误")
        if y.shape != (len(x),) or not np.isfinite(x).all() or (y < 0).any() or (y >= classes).any():
            raise ValueError("非法输入或标签")
        counts[split] = np.bincount(y, minlength=classes).tolist()
        if min(counts[split]) == 0:
            raise ValueError("固定划分存在空类别")
        row_hashes = _row_hashes(x)
        hashes[split] = set(row_hashes)
        if len(hashes[split]) != len(x):
            raise ValueError("固定划分内部仍有精确重复")
        datasets[split] = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    overlap = {f"{a}_{b}": len(hashes[a] & hashes[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlap.values()):
        raise ValueError("固定划分存在跨集合重复")
    report = {"case": case, "mode": "fixed_deduplicated_baseline", "classes": classes,
              "split_seed": read_json(root / "summary.json")["seed"],
              "split_sizes": info["split_counts"], "class_counts": counts,
              "cross_split_duplicate_groups": overlap, "normalization": info["normalization"],
              "baseline_plan_sha256": fingerprint(baseline / "plan.json"),
              "prepared_file_sha256": plan["prepared_sha256"], "csv_file_sha256": meta["file_sha256"],
              "protocol_limitation": "Exact deduplication does not prove independent physical acquisitions."}
    return DataSplits(datasets["train"], datasets["val"], datasets["test"], report)
