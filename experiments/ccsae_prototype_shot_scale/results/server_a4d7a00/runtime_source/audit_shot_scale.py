"""规模验证的只读预检：冻结源码/环境/旧结果，独立复算划分和归一化。

输出 provenance/，不修改模型代码、旧权重或原始数据。所有断言通过后才允许训练。
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import ccsae
from ccsae.data import load_raw
from ccsae.fewshot_data import make_fewshot_split
from common import dump, file_sha

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
BASE = ROOT / 'outputs/proto_shot_scale/server_a4d7a00'
P1 = ROOT / 'outputs/proto_scale_p1/server_a4d7a00'
D0 = ROOT / 'outputs/proto_diagnostics/server_a4d7a00'
COMMIT = 'a4d7a008d7e7597a864624f63f4185d91fada144'


def old_job(case, seed):
    """定位已有、已完成核验的 K20 P1 配对；后续始终只读访问。"""
    phase = 'seed42_precheck' if seed == 42 else 'seeds43_44_followup'
    return P1 / phase / case / f'k20_sample{seed}_train{seed}_distnone'


def digest_rows(values):
    """按真实 float32 波形字节识别样本，与模型输入精度一致。"""
    return [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in values]


def select(pool, labels, classes, size, seed):
    """独立重放逐类池打乱后取前缀，不调用上游内部采样函数。"""
    rng = np.random.default_rng(seed)
    result = []
    for c in range(classes):
        candidates = np.asarray([i for i in pool if labels[i] == c], dtype=np.int64)
        rng.shuffle(candidates)
        assert len(candidates) >= size
        result.extend(candidates[:size].tolist())
    return np.asarray(result)


def main():
    """先检查版本和旧结果，再审计全部 18 个 K/电路/种子组合。"""
    provenance = BASE / 'provenance'
    provenance.mkdir(parents=True, exist_ok=False)
    (provenance / 'splits').mkdir()
    assert Path(ccsae.__file__).resolve() == ROOT / 'src/ccsae/__init__.py'
    assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip() == COMMIT
    previous = json.loads((P1 / 'COMPLETION_VERIFICATION.json').read_text())
    assert previous['status'] == 'passed' and previous['p1_paired_runs'] == 6
    assert previous['independent_checkpoint_reload_predictions_and_metrics_verified'] is True
    names = subprocess.check_output(['git','ls-files','src','tests','pyproject.toml'], cwd=ROOT, text=True).splitlines()
    sources = {n: file_sha(ROOT/n) for n in names}
    for name, sha in sources.items():
        blob = subprocess.check_output(['git','show',f'{COMMIT}:{name}'], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == sha
    helpers = {p.relative_to(ROOT).as_posix(): file_sha(p) for p in sorted((ROOT/'server_experiment').glob('*.py'))}
    dump(provenance / 'source_before.json', {'source_commit':COMMIT,'tracked_sources':sources,'server_helpers':helpers})
    protected = json.loads((D0/'input_before.json').read_text())
    # 在原 170 个旧结果/数据/环境文件基础上，保护全部已归档 D0、P1 与旧辅助脚本。
    for folder in (D0,P1):
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                protected[p.relative_to(WORK).as_posix()] = file_sha(p)
    previous_helpers = json.loads((P1/'SOURCE_MANIFEST.json').read_text())['server_helpers']
    for name, sha in previous_helpers.items():
        assert file_sha(ROOT/name) == sha
        protected[(ROOT/name).relative_to(WORK).as_posix()] = sha
    assert all(file_sha(WORK/name) == sha for name,sha in protected.items())
    dump(provenance / 'input_before.json', protected)
    packages = {d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    assert packages == json.loads((D0/'runtime.json').read_text())['packages']
    runtime = {'python':sys.version,'executable':sys.executable,'packages':packages,
               'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__,
               'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
               'environment_reused':True,'installed_or_downloaded_packages':False}
    dump(provenance / 'runtime_before.json', runtime)
    assert file_sha(WORK/'Data.zip') == '11623819dcbb9287ff7ffea4d763d0a6fe3dd77d74c61bfe581bd8ea792b4e49'
    audits = []
    for case in ('skl','abl'):
        tx,ty,ex,ey = load_raw(WORK/'Data.zip',case)
        raw = np.concatenate((tx,ex)).astype(np.float32)
        labels = np.concatenate((ty,ey)).astype(np.int64)
        hashes = digest_rows(raw)
        seen, ids = {}, []
        for i,h in enumerate(hashes):
            if h in seen:
                assert labels[seen[h]] == labels[i]
            else:
                seen[h] = i
                ids.append(i)
        raw,labels = raw[ids],labels[ids]
        hashes = [hashes[i] for i in ids]
        classes = 13 if case == 'skl' else 11
        assert len(raw) == (3800 if case == 'skl' else 3100)
        pools = {n:[] for n in ('train','val','test')}
        rng = np.random.default_rng(20261009)
        for c in range(classes):
            available = np.flatnonzero(labels == c)
            rng.shuffle(available)
            n = len(available)
            nt,nv = round(n*.70),round(n*.15)
            pools['train'].extend(available[:nt].tolist())
            pools['val'].extend(available[nt:nt+nv].tolist())
            pools['test'].extend(available[nt+nv:].tolist())
        sets = {n:{hashes[i] for i in v} for n,v in pools.items()}
        assert not (sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
        fingerprints = {n:hashlib.sha256('\n'.join(sorted(v)).encode('ascii')).hexdigest() for n,v in sets.items()}
        val_ids = select(pools['val'],labels,classes,5,20261009)
        fixed_val = {str(c):[hashes[i] for i in val_ids if labels[i]==c] for c in range(classes)}
        for seed in (42,43,44):
            k20_audit = json.loads((old_job(case,seed)/'split_audit.json').read_text())
            selection = {}
            for k in (10,20,40):
                split = make_fewshot_split(WORK/'Data.zip',case,k=k,sample_seed=seed)
                a = split.audit
                train_ids = select(pools['train'],labels,classes,k,seed)
                train_hashes = {str(c):[hashes[i] for i in train_ids if labels[i]==c] for c in range(classes)}
                assert a['selected_train_hashes_by_class'] == train_hashes
                assert a['selected_val_hashes_by_class'] == fixed_val == k20_audit['selected_val_hashes_by_class']
                for n,sha in fingerprints.items():
                    assert a[f'{n}_pool_sha256'] == sha == k20_audit[f'{n}_pool_sha256']
                assert a['selected_train_size'] == classes*k and a['selected_val_size'] == classes*5
                assert np.bincount(split.train_y.numpy()).tolist() == [k]*classes
                assert np.bincount(split.val_y.numpy()).tolist() == [5]*classes
                assert np.array_equal(split.train_y.numpy(),labels[train_ids])
                assert np.array_equal(split.val_y.numpy(),labels[val_ids])
                selected = raw[train_ids].astype(np.float64)
                mean,std = float(selected.mean()),max(float(selected.std()),1e-8)
                assert a['train_scaler_mean'] == mean and a['train_scaler_std'] == std
                assert np.array_equal(split.train_x.numpy(),((selected-mean)/std).astype(np.float32))
                assert np.array_equal(split.val_x.numpy(),((raw[val_ids].astype(np.float64)-mean)/std).astype(np.float32))
                if k == 20:
                    assert a == k20_audit, '已有 K20 数据必须精确一致'
                selection[k] = train_hashes
                dump(provenance/'splits'/f'{case}_k{k}_seed{seed}.json', a)
                audits.append({'case':case,'k':k,'seed':seed,'train_size':classes*k,'val_size':classes*5,
                               'stage1_updates':10*((classes*k+63)//64),'stage2_updates_per_method':400,
                               'train_scaler_mean':mean,'train_scaler_std':std})
            for c in range(classes):
                name=str(c)
                assert selection[10][name] == selection[20][name][:10] == selection[40][name][:10]
                assert selection[20][name] == selection[40][name][:20]
    dump(provenance/'preflight.json', {'status':'passed','source_commit':COMMIT,'combinations':18,
         'new_pairs':12,'reused_k20_pairs':6,'nested_train_prefixes_exact':True,'fixed_raw_val_exact':True,
         'same_pools_as_k20':True,'independent_dedup_split_sampling_and_scaler_verified':True,
         'protected_file_count':len(protected),'test_evaluated':False,'jobs':audits})
    print(f'预检通过：18 个划分独立复算一致，嵌套 Train、固定 Val、K20 复用均核验；保护 {len(protected)} 个旧文件。')

if __name__ == '__main__':
    main()
