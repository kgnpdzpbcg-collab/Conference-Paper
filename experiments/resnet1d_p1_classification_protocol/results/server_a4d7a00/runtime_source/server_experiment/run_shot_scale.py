"""冻结 P1，只改变每类 Train 数 K；K20 使用旧结果，补跑 K10/40。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
import numpy as np
from common import configure, dump, file_sha, state_sha
import torch
import ccsae
import ccsae.fewshot_train as ft

ROOT = Path(__file__).resolve().parents[1]


def worker_init():
    """子进程独立初始化 GPU；明确从本轮 src 导入，不修改旧环境中的 editable 包。"""
    configure()
    assert Path(ccsae.__file__).resolve() == ROOT / "src/ccsae/__init__.py"


def run_pair(case, k, seed, output):
    """一个进程负责一组共享 Stage 1 的 FC/Proto，确保配对内部不被拆开。"""
    out = Path(output)
    job = out / case / f"k{k}_sample{seed}_train{seed}_distnone"
    job.mkdir(parents=True, exist_ok=False)
    args = argparse.Namespace(data=str(ROOT.parent / "Data.zip"), case=case,
        k=k, val_per_class=5, split_seed=20261009, val_seed=20261009,
        sample_seed=seed, seed=seed, mode="both", stage1_epochs=10,
        stage1_batch_size=64, epochs=40, steps_per_epoch=10,
        support=3, query=2, metric="euclidean", temperature=1.0, distance_normalization="none",
        lr=None, scl_weight=None, device="cuda", output=str(out), tiny=False)
    trace = {"case": case, "k": k, "seed": seed, "distance_normalization": "none", "methods": {}, "stage1_last_sha256": None}
    active = None
    streams = {}
    original_train = ft.train_one
    original_sample = ft.sample_episode

    def observed_sample(*sampling_args, **sampling_kwargs):
        # 仅记录上游实际返回的 Support/Query 索引，不改变样本流、随机数或梯度计算。
        s, q = original_sample(*sampling_args, **sampling_kwargs)
        streams[active][0].append(s.numpy().copy())
        streams[active][1].append(q.numpy().copy())
        return s, q

    def observed_train(method, initial_state, split, cfg, train_args, device, save_dir):
        nonlocal active
        active = method
        streams[method] = [[], []]
        before = state_sha(initial_state)
        started = time.perf_counter()
        trace["methods"][method] = {"initial_state_sha256": before,
            "embedding_dim": cfg.feature_dim * 2, "classes": cfg.classes}
        dump(job / "progress.json", {"state": "running", "stage": method, "seed": seed, "k": k, "case": case})
        result = original_train(method, initial_state, split, cfg, train_args, device, save_dir)
        assert state_sha(initial_state) == before, "训练修改了共享初始化"
        trace["methods"][method]["wall_seconds"] = time.perf_counter() - started
        return result

    ft.sample_episode, ft.train_one = observed_sample, observed_train
    started = time.perf_counter()
    try:
        with (job / "train.log").open("w", encoding="utf-8", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            report = ft.run(args)
            torch.cuda.synchronize()
        for part in (0,1):
            assert np.array_equal(np.stack(streams["fc"][part]), np.stack(streams["proto"][part])), "配对 Episode 流不一致"
        assert trace["methods"]["fc"]["initial_state_sha256"] == trace["methods"]["proto"]["initial_state_sha256"]
        shared = torch.load(job / "shared_stage1_last.pt", map_location="cpu", weights_only=True)
        assert state_sha(shared["state_dict"]) == trace["methods"]["fc"]["initial_state_sha256"]
        trace["stage1_last_sha256"] = file_sha(job / "shared_stage1_last.pt")
        trace["paired_initialization_identical"] = True
        trace["paired_episodes_identical"] = True
        trace["updates_per_method"] = args.epochs * args.steps_per_epoch
        trace["total_wall_seconds"] = time.perf_counter() - started
        episode_arrays = {f"{m}_{name}": np.stack(streams[m][part]) for m in streams
                          for part,name in enumerate(("support", "query"))}
        np.savez_compressed(job / "episodes.npz", **episode_arrays)
        for name, array in episode_arrays.items():
            trace.setdefault("episode_stream_sha256", {})[name] = hashlib.sha256(array.tobytes()).hexdigest()
        dump(job / "pair_verification.json", trace)
        dump(job / "progress.json", {"state": "complete", "case": case, "k": k, "seed": seed})
        return {"case": case, "k": k, "seed": seed, "seconds": trace["total_wall_seconds"],
                "metrics": {m: report["models"][m]["val"] for m in ("fc", "proto")}}
    except BaseException:
        dump(job / "progress.json", {"state": "failed", "case": case, "k": k, "seed": seed, "error": traceback.format_exc()})
        raise
    finally:
        ft.sample_episode, ft.train_one = original_sample, original_train


class Resources:
    """每秒读取 GPU 与系统 CPU，用真实采样记录并发窗口的资源占用。"""
    def __init__(self):
        self.stop = threading.Event()
        self.rows = []
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        previous = None
        while not self.stop.is_set():
            counts = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
            idle, total = counts[3] + counts[4], sum(counts[:8])
            cpu = None if previous is None else 100 * (1 - (idle-previous[0]) / max(total-previous[1],1))
            previous = idle,total
            query = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip()
            gpu, memory = map(float, query.split(","))
            self.rows.append({"elapsed_seconds": time.perf_counter()-self.started,
                "gpu_percent": gpu, "gpu_memory_mib": memory, "cpu_percent": cpu})
            self.stop.wait(1)

    def start(self):
        self.started = time.perf_counter()
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join()
        return {"samples": self.rows,
            "mean_gpu_percent": float(np.mean([r["gpu_percent"] for r in self.rows])),
            "peak_gpu_memory_mib": max(r["gpu_memory_mib"] for r in self.rows),
            "sampling_window": "整个配对训练批次，包含子进程初始化和数据加载/保存"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,43,44], choices=(42,43,44))
    parser.add_argument("--ks", type=int, nargs="+", default=[10,40], choices=(10,40))
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    seeds = tuple(args.seeds)
    assert tuple(sorted(args.ks)) == (10,40) and tuple(sorted(seeds)) == (42,43,44)
    assert args.workers == 4, "使用已验证的四进程资源配置"
    assert json.loads((out.parent / "provenance/preflight.json").read_text())["status"] == "passed"
    tasks = [(case,k,seed) for k in args.ks for seed in seeds for case in ("skl","abl")]
    # 先启动较慢的 ABL 配对，让后续短任务补齐空闲进程，不改变每组训练内容。
    tasks.sort(key=lambda item: item[0] != "abl")
    started = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    resources = Resources()
    resources.start()
    results = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=worker_init) as pool:
            futures = [pool.submit(run_pair, case, k, seed, str(out)) for case,k,seed in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                dump(out / "progress.json", {"completed_pairs": len(results), "total_pairs":len(tasks), "results": results})
                print(f"完成 {len(results)}/{len(tasks)}: {result['case']} K={result['k']} seed={result['seed']} {result['seconds']:.2f}s", flush=True)
    except BaseException:
        dump(out / "failure.json", {"error": traceback.format_exc(), "completed": results})
        raise
    finally:
        usage = resources.finish()
        dump(out / "resource_usage.json", usage)
    dump(out / "timing.json", {"started_utc": start_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter()-started, "workers": args.workers, "cpu_threads_per_worker":4,
        "smoke": False, "pairs":results, "test_evaluated":False})

if __name__ == "__main__":
    main()
