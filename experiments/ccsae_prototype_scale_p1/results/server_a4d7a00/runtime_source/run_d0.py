"""按最新 MD 执行六组 D0；只读旧 Stage1 权重，输出与 P0 隔离。

输入为旧正式输出目录及原始 Data.zip；每组重采样 32 次 3-shot Support。
四进程独立 CUDA 诊断，复用现有 Python 环境和 P0 确定性 FP32 设置。
诊断前后校验旧结果、数据和已有环境元数据，拒绝覆盖本轮输出目录。
"""
from __future__ import annotations
import contextlib
import importlib.metadata
import json
import multiprocessing as mp
import subprocess
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from common import configure, file_sha, dump

ROOT=Path(__file__).resolve().parent.parent
WORK=ROOT.parent
OLD=WORK/'Conference-Paper-ccsae-prototype-k20/outputs/server_proto_k20_7ef4042'
OUT=ROOT/'outputs/proto_diagnostics/server_a4d7a00'


def fingerprints():
    """保护全部旧 P0 输出和冻结数据，同时记录环境配置文件。"""
    paths=[WORK/'Data.zip']
    for folder in (OLD,WORK/'Data',WORK/'Data_deduplicated'):
        paths.extend(sorted(p for p in folder.rglob('*') if p.is_file()))
    for name in ('pyproject.toml','uv.lock','.python-version'):
        p=WORK/'analog-circuit-crossdomain'/name
        if p.exists():
            paths.append(p)
    return {str(p.relative_to(WORK)):file_sha(p) for p in paths}


def initializer():
    """固定实际导入路径和数值设置，避免旧 editable 安装抢占新源码。"""
    configure()
    import ccsae.proto_diagnostics as module
    assert Path(module.__file__).resolve()==ROOT/'src/ccsae/proto_diagnostics.py'


def worker(case,seed):
    """单组 D0 使用真实 checkpoint；不执行 optimizer.step 或 Test 推理。"""
    import ccsae.proto_diagnostics as diagnostic
    start=time.monotonic()
    path=OUT/case/f'k20_seed{seed}_stage1.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    args=Namespace(data=str(WORK/'Data.zip'),run_dir=str(OLD/case/f'k20_sample{seed}_train{seed}'),
                   device='cuda',seed=seed,repeats=32,output=str(path))
    with path.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log):
        result=diagnostic.run(args)
    # 上游诊断函数恢复模型状态，旧磁盘权重只能读取；最终再核验全部文件。
    return {'case':case,'seed':seed,'seconds':time.monotonic()-start,'result':result}


def main():
    """先冻结输入与版本，再执行六组；失败保留日志，不自动重跑或覆盖。"""
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='a4d7a008d7e7597a864624f63f4185d91fada144'
    assert not OUT.exists(),'D0 输出目录已存在，拒绝覆盖'
    OUT.mkdir(parents=True)
    before=fingerprints()
    dump(OUT/'input_before.json',before)
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    dump(OUT/'runtime.json',{'python_environment':str(WORK/'analog-circuit-crossdomain/.venv'),
                           'packages':packages,'source_commit':'a4d7a008d7e7597a864624f63f4185d91fada144',
                           'diagnosed_checkpoint':'P0 shared_stage1_last.pt','workers':4,
                           'new_environment_created':False,'new_packages_installed':False})
    start=time.monotonic()
    jobs=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initializer) as pool:
        futures=[pool.submit(worker,case,seed) for case in ('skl','abl') for seed in (42,43,44)]
        for future in as_completed(futures):
            row=future.result()
            jobs.append(row)
            print('D0 done',row['case'],row['seed'],row['seconds'],flush=True)
    jobs.sort(key=lambda r:(r['case'],r['seed']))
    after=fingerprints()
    assert before==after,'旧结果、冻结数据或环境配置发生变化'
    assert packages=={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    dump(OUT/'input_after.json',after)
    summary={'stage':'D0','jobs':jobs,'wall_seconds':time.monotonic()-start,
             'old_results_and_data_unchanged':True,'existing_environment_unchanged':True,
             'test_evaluated':False,'checkpoint_kind':'Stage1 last; not Proto best'}
    dump(OUT/'SUMMARY.json',summary)
    print('D0 finished',summary['wall_seconds'],flush=True)

if __name__=='__main__':
    main()
