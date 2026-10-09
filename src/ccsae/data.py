"""DataPort CSV loader and leakage-aware splitting for the two analog circuits."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import hashlib
import io
import json
import zipfile
import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass
class DataSplits:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    report: dict


def _csv(root: Path, filename: str) -> np.ndarray:
    if root.is_file() and root.suffix.lower() == ".zip":
        with zipfile.ZipFile(root) as archive:
            return np.loadtxt(io.BytesIO(archive.read(filename)), delimiter=",", dtype=np.float32)
    return np.loadtxt(root / filename, delimiter=",", dtype=np.float32)


def load_raw(root: str | Path, case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prefix = {"skl": "SKLCircuit", "abl": "ABLCircuit"}.get(case)
    if prefix is None:
        raise ValueError("case must be skl or abl")
    root = Path(root)
    loaded = {}
    for split in ("train", "test"):
        x = _csv(root, f"{prefix}_INCI_{split}_data.csv")
        onehot = _csv(root, f"{prefix}_INCI_{split}_label.csv")
        if x.ndim != 2 or x.shape[1] != 1000 or onehot.ndim != 2 or x.shape[0] != onehot.shape[0]:
            raise ValueError("unexpected data or one-hot shape")
        if not np.isfinite(x).all() or not np.isfinite(onehot).all() or not np.allclose(onehot.sum(axis=1), 1):
            raise ValueError("non-finite or invalid one-hot labels")
        loaded[split] = (x, onehot.argmax(axis=1).astype(np.int64))
    return loaded["train"][0], loaded["train"][1], loaded["test"][0], loaded["test"][1]


def _row_hashes(x: np.ndarray) -> list[str]:
    return [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in x]


def split_data(root: str | Path, case: str, seed: int = 42,
               mode: str = "grouped", allow_official_overlap: bool = False) -> DataSplits:
    """Grouped: union official data, group exact waveforms, stratify groups 70/15/15.

    Official: old 70/30 split with 15% of official train reserved for val.
    Official protocol is blocked on exact test-train overlap unless explicitly overridden.
    """
    xtr, ytr, xte, yte = load_raw(root, case)
    train_hashes = set(_row_hashes(xtr))
    overlap = sum(h in train_hashes for h in _row_hashes(xte))
    if mode == "official" and overlap and not allow_official_overlap:
        raise ValueError(f"official test split has {overlap} exact train/test duplicates; "
                         "use --split grouped or explicitly --allow-official-overlap")
    rng = np.random.default_rng(seed)
    if mode == "grouped":
        x = np.concatenate([xtr, xte], axis=0)
        y = np.concatenate([ytr, yte], axis=0)
        groups = defaultdict(list)
        for i, h in enumerate(_row_hashes(x)):
            groups[h].append(i)
        # Identical features bearing conflicting labels make the task ambiguous.
        conflicts = sum(len({int(y[j]) for j in members}) > 1 for members in groups.values())
        if conflicts:
            raise ValueError(f"{conflicts} identical waveforms have conflicting labels")
        by_label = defaultdict(list)
        for key, members in groups.items():
            by_label[int(y[members[0]])].append(members)
        idx = {"train": [], "val": [], "test": []}
        for label, group_list in sorted(by_label.items()):
            rng.shuffle(group_list)
            n = len(group_list)
            if n < 3:
                raise ValueError(f"too few distinct groups for label {label}")
            ntrain = max(1, round(n * 0.70))
            nval = max(1, round(n * 0.15))
            if ntrain + nval >= n:
                ntrain = n - nval - 1
            for g in group_list[:ntrain]:
                idx["train"].extend(g)
            for g in group_list[ntrain:ntrain + nval]:
                idx["val"].extend(g)
            for g in group_list[ntrain + nval:]:
                idx["test"].extend(g)
    elif mode == "official":
        x = np.concatenate([xtr, xte], axis=0)
        y = np.concatenate([ytr, yte], axis=0)
        idx = {"train": [], "val": [], "test": list(range(len(xtr), len(x)))}
        for label in np.unique(ytr):
            inds = np.where(ytr == label)[0]
            rng.shuffle(inds)
            nval = max(1, round(len(inds) * 0.15))
            idx["val"].extend(inds[:nval].tolist())
            idx["train"].extend(inds[nval:].tolist())
    else:
        raise ValueError("mode must be grouped or official")
    for inds in idx.values():
        rng.shuffle(inds)
    train_values = x[idx["train"]]
    # Dataset-wide scalar stats, fit exclusively on train; avoids per-example amplitude removal.
    mean, std = float(train_values.mean(dtype=np.float64)), float(train_values.std(dtype=np.float64))
    std = max(std, 1e-8)
    datasets = {}
    for spl in ("train", "val", "test"):
        ids = idx[spl]
        feats = np.array((x[ids] - mean) / std, dtype=np.float32)
        datasets[spl] = TensorDataset(torch.from_numpy(feats), torch.from_numpy(y[ids].copy()))
    counts = {name: {str(int(k)): int(v) for k, v in zip(*np.unique(y[ids], return_counts=True))}
              for name, ids in idx.items()}
    hashes = _row_hashes(x)
    cross_group_overlap = {f"{a}_{b}": len({hashes[i] for i in idx[a]} & {hashes[j] for j in idx[b]})
                           for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    report = {"case": case, "mode": mode, "seed": seed, "original_test_overlap": overlap,
              "split_sizes": {k: len(v) for k, v in idx.items()}, "class_counts": counts,
              "cross_split_duplicate_groups": cross_group_overlap, "mean": mean, "std": std,
              "protocol_limitation": "Exact-value grouping is not proof of independent physical acquisitions."}
    return DataSplits(datasets["train"], datasets["val"], datasets["test"], report)