"""ResNet分类协议对照的只读预检：核对18个既有P1划分和真实Episode流。

不修改原数据、旧训练文件或环境；新证据只写入独立输出目录。
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'server_experiment'))
from common import dump, file_sha
import torch
import ccsae
from ccsae.data import load_raw
from ccsae.fewshot_data import make_fewshot_split
from ccsae.prototype import sample_episode

WORK = ROOT.parent
BASE = ROOT/'outputs/resnet_shot_scale/server_a4d7a00'
P1 = ROOT/'outputs/proto_shot_scale/server_a4d7a00'
COMMIT = 'a4d7a008d7e7597a864624f63f4185d91fada144'
MODEL_SHA = 'b9c96fa1d78ca0c4a1e7a7626a4f5a6a5e29456faa091ff5db4dcd47048e5263'


def read(path):
    """读取已有JSON；缺失文件立即报错，不跳过任何K或种子。"""
    return json.loads(path.read_text())


def reference_job(case, k, seed):
    """定位本轮对齐的CCSAE-FC/P1，而非历史完整Val/Test实验。"""
    if k == 20:
        phase = 'seed42_precheck' if seed == 42 else 'seeds43_44_followup'
        return ROOT/'outputs/proto_scale_p1/server_a4d7a00'/phase/case/f'k20_sample{seed}_train{seed}_distnone'
    return P1/'new_k10_k40'/case/f'k{k}_sample{seed}_train{seed}_distnone'


def main():
    """冻结旧文件后，核对输入、18个标准化划分和400步实际采样。"""
    provenance = BASE/'provenance'
    provenance.mkdir(parents=True,exist_ok=False)
    (provenance/'splits').mkdir()
    assert Path(ccsae.__file__).resolve() == ROOT/'src/ccsae/__init__.py'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == COMMIT
    assert read(P1/'COMPLETION_VERIFICATION.json')['status'] == 'passed'
    sources = read(P1/'SOURCE_MANIFEST.json')['tracked_sources']
    assert all(file_sha(ROOT/n) == h for n,h in sources.items())
    assert file_sha(Path(__file__).parent/'reference_models.py') == MODEL_SHA
    assert file_sha(WORK/'analog-circuit-baselines/baselines/models.py') == MODEL_SHA
    protected = read(P1/'provenance/input_before.json')
    # 旧P0/D0/K20的保护清单继续有效，并额外保护上轮规模验证的所有文件。
    for folder in (P1,WORK/'analog-circuit-fewshot/outputs/server_fewshot_v1'):
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                protected[p.relative_to(WORK).as_posix()] = file_sha(p)
    for name in read(P1/'SOURCE_MANIFEST.json')['server_helpers']:
        protected[(ROOT/name).relative_to(WORK).as_posix()] = file_sha(ROOT/name)
    protected['analog-circuit-baselines/baselines/models.py'] = MODEL_SHA
    assert all(file_sha(WORK/n) == h for n,h in protected.items())
    dump(provenance/'input_before.json',protected)
    packages = {d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    assert packages == read(P1/'provenance/runtime_before.json')['packages']
    dump(provenance/'runtime_before.json',{'executable':sys.executable,'python':sys.version,
         'packages':packages,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
         'environment_reused':True,'installed_or_downloaded_packages':False})
    jobs=[]
    for case in ('skl','abl'):
        tx,ty,ex,ey = load_raw(WORK/'Data.zip',case)
        raw=np.concatenate((tx,ex)).astype(np.float32)
        lookup={hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest():row for row in raw}
        classes=13 if case=='skl' else 11
        for k in (10,20,40):
            for seed in (42,43,44):
                split=make_fewshot_split(WORK/'Data.zip',case,k=k,sample_seed=seed)
                a=split.audit
                assert a==read(P1/'provenance/splits'/f'{case}_k{k}_seed{seed}.json')
                assert a==read(reference_job(case,k,seed)/'split_audit.json')
                # 从波形指纹回查原始float32行，独立检查当前K的Train-only标准化。
                train=np.stack([lookup[h] for c in range(classes) for h in a['selected_train_hashes_by_class'][str(c)]]).astype(np.float64)
                val=np.stack([lookup[h] for c in range(classes) for h in a['selected_val_hashes_by_class'][str(c)]]).astype(np.float64)
                mean,std=float(train.mean()),max(float(train.std()),1e-8)
                assert a['train_scaler_mean']==mean and a['train_scaler_std']==std
                assert np.array_equal(split.train_x.numpy(),((train-mean)/std).astype(np.float32))
                assert np.array_equal(split.val_x.numpy(),((val-mean)/std).astype(np.float32))
                assert np.bincount(split.train_y.numpy()).tolist()==[k]*classes
                assert np.bincount(split.val_y.numpy()).tolist()==[5]*classes
                old=np.load(reference_job(case,k,seed)/'episodes.npz')
                rng=np.random.default_rng(seed+901)
                for step in range(400):
                    s,q=sample_episode(split.train_y,classes,3,2,rng)
                    assert np.array_equal(s.numpy(),old['fc_support'][step])
                    assert np.array_equal(q.numpy(),old['fc_query'][step])
                    assert np.array_equal(old['fc_support'][step],old['proto_support'][step])
                    assert np.array_equal(old['fc_query'][step],old['proto_query'][step])
                dump(provenance/'splits'/f'{case}_k{k}_seed{seed}.json',a)
                jobs.append({'case':case,'k':k,'seed':seed,'train_size':len(train),'val_size':len(val),
                     'reference_relative_to_work':reference_job(case,k,seed).relative_to(WORK).as_posix(),
                     'reference_episode_file_sha256':file_sha(reference_job(case,k,seed)/'episodes.npz')})
    dump(provenance/'preflight.json',{'status':'passed','source_commit':COMMIT,'resnet_model_sha256':MODEL_SHA,
         'combinations':18,'all_split_audits_exact':True,'all_reference_episode_indices_exact':True,
         'independent_train_only_scaler_verified':True,'protected_file_count':len(protected),
         'stage1_epochs':0,'stage2_updates':400,'test_evaluated':False,'jobs':jobs})
    print(f'预检通过：18个划分和实际400步Episode均与P1一致；保护{len(protected)}个旧文件；原ResNet模型指纹一致。')

if __name__=='__main__':
    main()
