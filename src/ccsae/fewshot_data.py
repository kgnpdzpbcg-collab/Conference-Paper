"""Leakage-aware, limited-label splits for SKL/ABL.

Independent of the V1 grouped loader: V1 groups identical rows but retains their
multiplicity. This module actually retains one row per unique float32 waveform.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch

from .data import load_raw


@dataclass
class FewShotSplit:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    audit: dict


def _hashes(x: np.ndarray) -> list[str]:
    return [sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in x]


def _fingerprint(hashes: list[str]) -> str:
    return sha256("\n".join(sorted(hashes)).encode("ascii")).hexdigest()


def make_fewshot_split(
    source: str | Path,
    case: str,
    k: int = 20,
    val_per_class: int = 5,
    split_seed: int = 20261009,
    sample_seed: int = 42,
    val_seed: int = 20261009,
) -> FewShotSplit:
    """Deduplicate before stratifying; subsample ONLY train pool; K-only normalization.

    Sampling uses first-occurrence ordering and numpy.default_rng. For a given
    sample_seed, K=10 indices are nested inside K=20 and K=40.
    """
    if k < 5:
        raise ValueError("k must be at least 5 for the default 3-support/2-query episodes")
    if val_per_class < 1:
        raise ValueError("val_per_class must be >= 1")

    tr_x, tr_y, te_x, te_y = load_raw(source, case)
    x_all = np.concatenate((tr_x, te_x), axis=0).astype(np.float32, copy=False)
    y_all = np.concatenate((tr_y, te_y), axis=0).astype(np.int64, copy=False)
    hashes_all = _hashes(x_all)

    # Preserve first occurrence deterministically; reject contradictory labels.
    seen: dict[str, int] = {}
    unique_ids: list[int] = []
    for idx, h in enumerate(hashes_all):
        if h in seen:
            if y_all[seen[h]] != y_all[idx]:
                raise ValueError(f"same waveform has conflicting labels: {h}")
        else:
            seen[h] = idx
            unique_ids.append(idx)

    unique_idx = np.asarray(unique_ids, dtype=np.int64)
    x_unique, y_unique = x_all[unique_idx], y_all[unique_idx]
    hashes_unique = [hashes_all[i] for i in unique_idx]
    classes = np.unique(y_unique)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("expected 0..C-1 class labels")

    rng = np.random.default_rng(split_seed)
    pool: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for c in classes:
        ids = np.flatnonzero(y_unique == c)
        rng.shuffle(ids)
        n = len(ids)
        ntrain = round(n * 0.70)
        nval = round(n * 0.15)
        if ntrain < k or nval < val_per_class or n - ntrain - nval < 1:
            raise ValueError(f"insufficient unique records for class {c}: {n}")
        pool["train"].extend(ids[:ntrain].tolist())
        pool["val"].extend(ids[ntrain:ntrain + nval].tolist())
        pool["test"].extend(ids[ntrain + nval:].tolist())

    def choose_per_class(pool_ids: list[int], size: int, seed: int) -> np.ndarray:
        crng = np.random.default_rng(seed)
        result: list[int] = []
        ids_array = np.asarray(pool_ids, dtype=np.int64)
        for c in classes:
            available = ids_array[y_unique[ids_array] == c].copy()
            crng.shuffle(available)
            if len(available) < size:
                raise ValueError(f"class {c}: only {len(available)} records")
            result.extend(available[:size].tolist())
        return np.asarray(result, dtype=np.int64)

    few_ids = choose_per_class(pool["train"], k, sample_seed)
    val_ids = choose_per_class(pool["val"], val_per_class, val_seed)

    # IMPORTANT: fit scaler to ONLY K labelled training records per class.
    raw_train = x_unique[few_ids].astype(np.float64)
    mean, std = float(raw_train.mean()), float(raw_train.std())
    std = max(std, 1e-8)
    to_tensor = lambda ids: torch.from_numpy(
        ((x_unique[ids].astype(np.float64) - mean) / std).astype(np.float32)
    ).contiguous()

    all_sets = {name: {hashes_unique[j] for j in indices} for name, indices in pool.items()}
    if (all_sets["train"] & all_sets["val"] or all_sets["train"] & all_sets["test"]
            or all_sets["val"] & all_sets["test"]):
        raise AssertionError("data leakage: duplicate waveform across pools")

    audit = {
        "case": case, "classes": int(len(classes)), "k": int(k),
        "val_per_class": int(val_per_class),
        "split_seed": int(split_seed), "sample_seed": int(sample_seed),
        "val_seed": int(val_seed),
        "original_rows": int(len(x_all)), "unique_rows": int(len(unique_ids)),
        "duplicate_rows_removed": int(len(x_all) - len(unique_ids)),
        "pool_sizes": {name: len(ids) for name, ids in pool.items()},
        "selected_train_size": int(len(few_ids)), "selected_val_size": int(len(val_ids)),
        "train_pool_sha256": _fingerprint([hashes_unique[j] for j in pool["train"]]),
        "val_pool_sha256": _fingerprint([hashes_unique[j] for j in pool["val"]]),
        "test_pool_sha256": _fingerprint([hashes_unique[j] for j in pool["test"]]),
        "selected_train_hashes_by_class": {
            str(int(c)): [hashes_unique[i] for i in few_ids if y_unique[i] == c]
            for c in classes
        },
        "selected_val_hashes_by_class": {
            str(int(c)): [hashes_unique[i] for i in val_ids if y_unique[i] == c]
            for c in classes
        },
        "train_scaler_mean": mean, "train_scaler_std": std,
        "known_limitation": (
            "Exact-waveform deduplication is not independent physical acquisition. "
            "Regenerated partition may differ from historical server baseline manifest."
        ),
    }
    return FewShotSplit(
        train_x=to_tensor(few_ids), train_y=torch.from_numpy(y_unique[few_ids].copy()),
        val_x=to_tensor(val_ids), val_y=torch.from_numpy(y_unique[val_ids].copy()),
        audit=audit,
    )
