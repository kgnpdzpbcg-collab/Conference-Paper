"""固定Test只读评估：冻结54份Val最佳权重，禁止训练、重新选权重和调参。

preflight先核对数据、权重、旧证据；run并发评估；verify独立重载；report汇总。
所有新文件写入fixed_test，旧训练目录、数据、环境均保持不变。
"""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse, csv, hashlib, importlib.metadata, json, multiprocessing as mp
import subprocess, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT.parent
BASE=ROOT/'outputs/fixed_test/server_a4d7a00'
RES=ROOT/'outputs/resnet_shot_scale/server_a4d7a00'
P1=ROOT/'outputs/proto_shot_scale/server_a4d7a00'
COMMIT='a4d7a008d7e7597a864624f63f4185d91fada144'
DATA_SHA='11623819dcbb9287ff7ffea4d763d0a6fe3dd77d74c61bfe581bd8ea792b4e49'
MODEL_SHA='b9c96fa1d78ca0c4a1e7a7626a4f5a6a5e29456faa091ff5db4dcd47048e5263'
sys.path.insert(0,str(ROOT/'server_experiment'))
sys.path.insert(0,str(ROOT/'server_experiment/resnet_shot_scale'))
# common在torch初始化前设定CUDA确定性变量；保留服务器原环境。
from common import configure, dump, file_sha, state_sha
from verify_shot_scale import manual_metrics
from reference_models import ResNet1D
import torch, ccsae
from ccsae.data import load_raw
from ccsae.fewshot_data import make_fewshot_split
from ccsae.model import CCSAE, CCSAEConfig
from ccsae.prototype import build_prototypes, prototype_logits, macro_metrics
METHODS=('fc','proto','resnet1d')
NAMES={'fc':'CCSAE-FC','proto':'CCSAE-Proto (P1)','resnet1d':'ResNet1D'}


def read(path):
    """已有证据缺失立即报错，不跳过K、种子或模型。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reference(case,k,seed):
    """只定位当前P1权重，不使用历史完整Val/Test模型。"""
    label=f'k{k}_sample{seed}_train{seed}_distnone'
    if k==20:
        phase='seed42_precheck' if seed==42 else 'seeds43_44_followup'
        return ROOT/'outputs/proto_scale_p1/server_a4d7a00'/phase/case/label
    return P1/'new_k10_k40'/case/label


@lru_cache(maxsize=2)
def pool(case):
    """独立重建冻结数据：原Train后原Test、float32指纹、首出现去重。

    按类顺序使用同一default_rng(20261009)，保留70/15/15切片顺序。
    不按指纹排序；不使用历史Data_deduplicated的另一套Test。
    返回原始波形、标签、指纹、三个池索引及池指纹。
    """
    tx,ty,ex,ey=load_raw(WORK/'Data.zip',case)
    raw=np.concatenate((tx,ex)).astype(np.float32)
    labels=np.concatenate((ty,ey)).astype(np.int64)
    seen,ids,hashes={},[],[]
    for i,row in enumerate(raw):
        h=hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        if h in seen:
            assert labels[seen[h]]==labels[i], '重复波形标签冲突'
        else:
            seen[h]=i; ids.append(i); hashes.append(h)
    x,y=raw[ids],labels[ids]
    classes=13 if case=='skl' else 11
    assert np.array_equal(np.unique(y),np.arange(classes))
    pools={n:[] for n in ('train','val','test')}
    rng=np.random.default_rng(20261009)
    for c in range(classes):
        candidate=np.flatnonzero(y==c); rng.shuffle(candidate)
        nt,nv=round(len(candidate)*.7),round(len(candidate)*.15)
        pools['train'].extend(candidate[:nt].tolist())
        pools['val'].extend(candidate[nt:nt+nv].tolist())
        pools['test'].extend(candidate[nt+nv:].tolist())
    sets={n:{hashes[i] for i in v} for n,v in pools.items()}
    assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
    fp={n:hashlib.sha256('\n'.join(sorted(s)).encode('ascii')).hexdigest() for n,s in sets.items()}
    assert len(x)==(3800 if case=='skl' else 3100)
    expected=[30]+[45]*12 if case=='skl' else [15]+[45]*10
    assert np.bincount(y[pools['test']],minlength=classes).tolist()==expected
    return x,y,hashes,pools,fp


def tensors(case,audit):
    """由保存指纹恢复有序Train/Val；Test统一使用保存的Train统计量。

    从原始Train复算mean/std要求逐位相等；Test从不参与拟合。
    三方法共享返回的张量与标签，批大小及顺序保持一致。
    """
    x,y,hashes,pools,fp=pool(case)
    lookup={h:i for i,h in enumerate(hashes)}
    classes=audit['classes']; selected={}
    for n in ('train','val'):
        hs=[h for c in range(classes) for h in audit[f'selected_{n}_hashes_by_class'][str(c)]]
        selected[n]=[lookup[h] for h in hs]
        assert set(selected[n])<=set(pools[n])
        size=audit['k'] if n=='train' else 5
        assert np.bincount(y[selected[n]],minlength=classes).tolist()==[size]*classes
    selected['test']=pools['test']
    for n in pools:
        assert fp[n]==audit[f'{n}_pool_sha256'] and len(pools[n])==audit['pool_sizes'][n]
    train=x[selected['train']].astype(np.float64)
    mean,std=float(train.mean()),max(float(train.std()),1e-8)
    assert mean==audit['train_scaler_mean'] and std==audit['train_scaler_std']
    return {n:(torch.from_numpy(((x[idx].astype(np.float64)-mean)/std).astype(np.float32)).contiguous(),
               torch.from_numpy(y[idx].copy())) for n,idx in selected.items()}


def preflight():
    """在首次Test推理前，冻结权重选择、输入、源码、旧输出和包版本。"""
    assert not BASE.exists(), '禁止覆盖已存在的Test评估'
    assert Path(ccsae.__file__).resolve()==ROOT/'src/ccsae/__init__.py'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==COMMIT
    assert file_sha(WORK/'Data.zip')==DATA_SHA
    assert file_sha(ROOT/'server_experiment/resnet_shot_scale/reference_models.py')==MODEL_SHA
    source=read(P1/'SOURCE_MANIFEST.json')['tracked_sources']
    assert all(file_sha(ROOT/n)==h for n,h in source.items())
    protected=read(RES/'provenance/input_before.json')
    for folder in (RES,ROOT/'server_experiment/resnet_shot_scale'):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                protected[path.relative_to(WORK).as_posix()]=file_sha(path)
    assert all(file_sha(WORK/n)==h for n,h in protected.items())
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    assert packages==read(RES/'provenance/runtime_before.json')['packages']
    prior={}
    for k20,path in ((True,P1/'REUSED_K20_VERIFICATION.json'),(False,P1/'new_k10_k40/verification.json')):
        evidence=read(path); assert evidence['status']=='passed'
        for r in evidence['results']:
            for m,v in r['methods'].items():
                prior[(r['case'],20 if k20 else r['k'],r['seed'],m)]=v['checkpoint_sha256']
    for r in read(RES/'VERIFICATION.json')['results']:
        prior[(r['case'],r['k'],r['seed'],'resnet1d')]=r['best_checkpoint_sha256']
    assert len(prior)==54
    manifests,checkpoints=[],[]
    for case in ('skl','abl'):
        x,y,hs,pools,fp=pool(case)
        manifests.append({'case':case,'pool_sha256':fp,'pool_sizes':{n:len(v) for n,v in pools.items()},
            'test_hashes_ordered':[hs[i] for i in pools['test']], 'test_labels_ordered':y[pools['test']].tolist(),
            'test_class_counts':np.bincount(y[pools['test']]).tolist()})
        for k in (10,20,40):
            for seed in (42,43,44):
                ref=reference(case,k,seed)
                res=RES/'runs'/case/f'k{k}_sample{seed}_train{seed}'
                a=read(ref/'split_audit.json')
                assert a==read(res/'split_audit.json')==make_fewshot_split(WORK/'Data.zip',case,k=k,sample_seed=seed).audit
                tensors(case,a)
                cfg=read(ref/'config.json')
                assert cfg['distance_normalization']=='none' and cfg['temperature']==1 and cfg['metric']=='euclidean'
                assert cfg['stage1_epochs']==10 and not cfg['tiny']
                for m in METHODS:
                    job=res if m=='resnet1d' else ref
                    path=job/('best.pt' if m=='resnet1d' else f'{m}_best.pt')
                    sha=file_sha(path); assert sha==prior[(case,k,seed,m)]
                    cp=torch.load(path,map_location='cpu',weights_only=True)
                    history=read(job/('history.json' if m=='resnet1d' else f'{m}_history.json'))
                    best=max(history,key=lambda r:r['macro_f1'])
                    summary=read(job/'summary.json'); assert summary['test_evaluated'] is False
                    r=summary if m=='resnet1d' else summary['models'][m]
                    assert len(history)==40 and cp['epoch']==best['epoch']==r['best_epoch']
                    assert r['val']['macro_f1']==best['macro_f1']
                    if m!='resnet1d':
                        assert cp['case']==case and cp['k']==k and cp['method']==m
                        assert cp['metric']=='euclidean' and cp['temperature']==1 and cp['distance_normalization']=='none'
                    assert all(torch.isfinite(t).all() for t in cp['state_dict'].values())
                    checkpoints.append({'case':case,'k':k,'seed':seed,'method':m,
                        'checkpoint':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha,
                        'state_sha256':state_sha(cp['state_dict']),'best_epoch':cp['epoch'],
                        'selection':'highest Val Macro-F1; earliest epoch on exact tie',
                        'val_macro_f1':best['macro_f1'],'split_audit_sha256':file_sha(job/'split_audit.json'),
                        'train_scaler_mean':a['train_scaler_mean'],'train_scaler_std':a['train_scaler_std']})
    BASE.mkdir(parents=True); (BASE/'provenance').mkdir()
    dump(BASE/'provenance/input_before.json',protected)
    dump(BASE/'provenance/runtime_before.json',{'packages':packages,'executable':sys.executable,
        'python':sys.version,'torch':torch.__version__,'environment_reused':True,'packages_installed':False})
    dump(BASE/'TEST_MANIFEST.json',{'data_sha256':DATA_SHA,'protocol':'P1 float32 first-occurrence dedup; 70/15/15; split_seed=20261009',
        'test_use':'complete existing fixed test pool; no resampling','cases':manifests,
        'limitation':'Current model Train/Test are disjoint, but dataset and some waveforms were used historically; not globally pristine blind test.'})
    dump(BASE/'CHECKPOINT_MANIFEST.json',{'frozen_before_test_inference':True,'count':54,'checkpoints':checkpoints})
    dump(BASE/'SOURCE_MANIFEST.json',{'source_commit':COMMIT,'tracked_sources':source,
        'helpers':{p.relative_to(ROOT).as_posix():file_sha(p) for p in (Path(__file__),ROOT/'server_experiment/common.py',ROOT/'server_experiment/verify_shot_scale.py',ROOT/'server_experiment/resnet_shot_scale/reference_models.py')}})
    dump(BASE/'PREFLIGHT.json',{'status':'passed','checkpoint_count':54,'split_groups':18,
        'fixed_test_pools_verified':True,'train_val_test_disjoint':True,'train_only_scalers_verified':True,
        'previous_checkpoint_hashes_exact':True,'protected_file_count':len(protected),
        'test_inference_started':False,'utc':datetime.now(timezone.utc).isoformat()})
    print(f'预检通过：54份Val最佳权重、18组Train统计量和两个Test池已冻结；保护{len(protected)}份旧文件。',flush=True)


def worker_init():
    """每进程四CPU线程、确定性FP32；显式使用当前源码。"""
    configure()
    assert Path(ccsae.__file__).resolve()==ROOT/'src/ccsae/__init__.py'


@torch.inference_mode()
def infer(model,method,query,train,labels,classes,independent=False):
    """eval批大小64，无梯度和BN重新估计。Proto仅用Train类均值。

    正式路径使用forward；核验路径分开encode/features和classifier。
    原型距离固定为FP32 cdist平方，温度1，不除嵌入维度。
    """
    assert not any(m.training for m in model.modules())
    if method=='resnet1d':
        scores=torch.cat([model.classifier(model.features(c.cuda().unsqueeze(1))) if independent
                          else model(c.cuda().unsqueeze(1)) for c in query.split(64)])
        return scores,None,None
    if method=='fc':
        scores=torch.cat([model.classifier(model.feature_vector(model.encode(c.cuda()))) if independent
                          else model(c.cuda())['logits'] for c in query.split(64)])
        return scores,None,None
    def features(values):
        return torch.cat([model.feature_vector(model.encode(c.cuda())) if independent
                          else model(c.cuda())['features'] for c in values.split(64)])
    qz,tz=features(query),features(train)
    centers=(torch.stack([tz[labels.cuda()==c].mean(0) for c in range(classes)]) if independent
             else build_prototypes(tz,labels.cuda(),classes))
    scores=prototype_logits(qz,centers,metric='euclidean',temperature=1.,distance_normalization='none')
    return scores,qz,centers


def evaluate_group(case,k,seed):
    """三种方法共享该组数据；先复核原Val预测，再执行固定Test推理。"""
    started=time.perf_counter()
    out=BASE/'runs'/case/f'k{k}_seed{seed}'; out.mkdir(parents=True,exist_ok=False)
    audit=read(reference(case,k,seed)/'split_audit.json'); data=tensors(case,audit)
    classes=audit['classes']; results=[]
    records=[r for r in read(BASE/'CHECKPOINT_MANIFEST.json')['checkpoints'] if (r['case'],r['k'],r['seed'])==(case,k,seed)]
    for r in records:
        method=r['method']; path=ROOT/r['checkpoint']
        assert file_sha(path)==r['checkpoint_sha256']
        cp=torch.load(path,map_location='cpu',weights_only=True)
        model=(ResNet1D(classes) if method=='resnet1d' else CCSAE(CCSAEConfig.for_case(case))).cuda().eval()
        model.load_state_dict(cp['state_dict'],strict=True)
        assert state_sha(model.state_dict())==r['state_sha256']
        val,_,_=infer(model,method,data['val'][0],*data['train'],classes)
        saved=read(path.parent/('val_predictions.json' if method=='resnet1d' else f'{method}_val_predictions.json'))
        assert saved['truth']==data['val'][1].tolist() and saved['predicted']==val.argmax(1).cpu().tolist(), 'Val复核失败，停止该组Test'
        scores,qz,centers=infer(model,method,data['test'][0],*data['train'],classes)
        assert torch.isfinite(scores).all()
        truth,pred=data['test'][1].numpy(),scores.argmax(1).cpu().numpy()
        metrics,cm,f1=manual_metrics(truth,pred,classes)
        assert state_sha(model.state_dict())==r['state_sha256'], '推理改变权重或BN'
        arrays={'truth':truth,'predicted':pred,'scores':scores.cpu().numpy()}
        if method=='proto':
            arrays.update(query_features=qz.cpu().numpy(),centers=centers.cpu().numpy())
        np.savez_compressed(out/f'{method}_scores.npz',**arrays)
        row={**r,'model_name':NAMES[method],'test_evaluated':True,'test_size':len(truth),
            'test_pool_sha256':audit['test_pool_sha256'],'metrics':metrics,
            'confusion_matrix':cm.tolist(),'per_class_f1':f1.tolist(),
            'saved_val_predictions_exact':True,'model_state_unchanged':True}
        dump(out/f'{method}_predictions.json',{'truth':truth.tolist(),'predicted':pred.tolist()})
        dump(out/f'{method}_result.json',row); results.append(row)
        del model
    torch.cuda.synchronize()
    dump(out/'group.json',{'case':case,'k':k,'seed':seed,'status':'completed',
        'methods':METHODS,'seconds':time.perf_counter()-started,'test_evaluated':True})
    return results


def run():
    """四进程评估18组，共54份模型；无优化器、backward或train调用。"""
    assert read(BASE/'PREFLIGHT.json')['status']=='passed'
    started=time.perf_counter(); utc=datetime.now(timezone.utc).isoformat(); results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=worker_init) as p:
        futures=[p.submit(evaluate_group,c,k,s) for c in ('skl','abl') for k in (10,20,40) for s in (42,43,44)]
        for f in as_completed(futures):
            rows=f.result(); results.extend(rows)
            print(f"完成{len(results)}/54模型：{rows[0]['case']} K={rows[0]['k']} seed={rows[0]['seed']}",flush=True)
    assert len(results)==54
    dump(BASE/'EVALUATION.json',{'status':'completed','models':54,'test_evaluated':True,
        'no_training':True,'no_checkpoint_reselection':True,'workers':4,'threads_per_worker':4,'batch_size':64,
        'utc_start':utc,'utc_end':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-started,
        'results':sorted(results,key=lambda r:(r['case'],r['k'],r['seed'],r['method']))})


@torch.inference_mode()
def verify():
    """重载全部最佳权重独立重建输出，复算混淆矩阵和sklearn指标。

    Proto另检查直接平方距离/FP64距离的舍入边界；正式预测不替换。
    """
    configure(); checks=[]
    for row in read(BASE/'EVALUATION.json')['results']:
        case,k,seed,method=(row[n] for n in ('case','k','seed','method'))
        audit=read(reference(case,k,seed)/'split_audit.json'); data=tensors(case,audit); classes=audit['classes']
        path=ROOT/row['checkpoint']; assert file_sha(path)==row['checkpoint_sha256']
        cp=torch.load(path,map_location='cpu',weights_only=True)
        model=(ResNet1D(classes) if method=='resnet1d' else CCSAE(CCSAEConfig.for_case(case))).cuda().eval()
        model.load_state_dict(cp['state_dict'])
        scores,qz,centers=infer(model,method,data['test'][0],*data['train'],classes,independent=True)
        truth,pred=data['test'][1].numpy(),scores.argmax(1).cpu().numpy()
        out=BASE/'runs'/case/f'k{k}_seed{seed}'
        with np.load(out/f'{method}_scores.npz') as saved:
            assert np.array_equal(truth,saved['truth']) and np.array_equal(pred,saved['predicted'])
            assert np.array_equal(scores.cpu().numpy(),saved['scores'])
            if method=='proto':
                assert np.array_equal(qz.cpu().numpy(),saved['query_features']) and np.array_equal(centers.cpu().numpy(),saved['centers'])
        metrics,cm,f1=manual_metrics(truth,pred,classes); other=macro_metrics(truth.tolist(),pred.tolist(),classes)
        for n,v in metrics.items():
            assert np.isclose(v,row['metrics'][n],rtol=0,atol=1e-12) and np.isclose(v,other[n],rtol=0,atol=1e-12)
        assert cm.tolist()==row['confusion_matrix'] and f1.tolist()==row['per_class_f1']
        diagnostic=None
        if method=='proto':
            direct=-((qz[:,None,:]-centers[None,:,:]).square().sum(-1))
            err=(scores-direct).abs().max(1).values
            top=scores.topk(2,dim=1).values; margin=top[:,0]-top[:,1]
            mismatch=scores.argmax(1)!=direct.argmax(1)
            assert bool((margin[mismatch]<=2*err[mismatch]+1e-12).all())
            stable=-((qz.double()[:,None,:]-centers.double()[None,:,:]).square().sum(-1))
            eps=torch.finfo(torch.float32).eps; width=qz.shape[1]; gamma=width*eps/(1-width*eps)
            norm=qz.double().square().sum(1)[:,None]+centers.double().square().sum(1)[None,:]
            budget=8*gamma*norm+8*eps*stable.abs()+1e-12
            assert bool(((scores.double()-stable).abs()<=budget).all()) and bool(((direct.double()-stable).abs()<=budget).all())
            err64=(scores.double()-stable).abs().max(1).values; mismatch64=scores.argmax(1)!=stable.argmax(1)
            assert bool((margin[mismatch64].double()<=2*err64[mismatch64]+1e-12).all())
            diagnostic={'direct_fp32_prediction_differences':int(mismatch.sum()),
                'direct_fp64_distance_prediction_differences':int(mismatch64.sum()),
                'max_absolute_fp64_score_error':float(err64.max()),'near_tie_roundoff_bounds_passed':True,
                'diagnostic_only_formal_predictions_unchanged':True}
        assert state_sha(model.state_dict())==row['state_sha256']
        checks.append({'case':case,'k':k,'seed':seed,'method':method,'status':'passed',
            'checkpoint_sha256':file_sha(path),'independent_reload_scores_exact':True,
            'predictions_exact':True,'manual_and_sklearn_metrics_equal':True,
            'state_and_bn_buffers_unchanged':True,'distance_numerics':diagnostic})
        del model
    protected=read(BASE/'provenance/input_before.json')
    assert all(file_sha(WORK/n)==h for n,h in protected.items())
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    assert packages==read(BASE/'provenance/runtime_before.json')['packages']
    sources=read(BASE/'SOURCE_MANIFEST.json')
    assert all(file_sha(ROOT/n)==h for part in ('tracked_sources','helpers') for n,h in sources[part].items())
    assert file_sha(WORK/'Data.zip')==DATA_SHA
    dump(BASE/'VERIFICATION.json',{'status':'passed','models':54,'test_evaluated':True,'checks':checks,
        'protected_file_count':len(protected),'all_previous_files_unchanged':True,
        'data_source_and_environment_unchanged':True,'installed_or_downloaded_packages':False})
    print('核验通过：54模型独立重载scores/预测/指标完全一致；权重、BN、数据、旧文件和包版本不变。',flush=True)


def report():
    """按固定3种子汇总Test Macro-F1，样本标准差ddof=1；保存配对差值。"""
    assert read(BASE/'VERIFICATION.json')['status']=='passed'
    e=read(BASE/'EVALUATION.json'); rows=e['results']; aggregates=[]; paired=[]
    for case in ('skl','abl'):
        for k in (10,20,40):
            values={}
            for m in METHODS:
                selected=sorted([r for r in rows if (r['case'],r['k'],r['method'])==(case,k,m)],key=lambda r:r['seed'])
                assert [r['seed'] for r in selected]==[42,43,44]
                v=np.array([r['metrics']['macro_f1'] for r in selected])*100; values[m]=v
                aggregates.append({'case':case,'k':k,'method':m,'model_name':NAMES[m],
                    'test_macro_f1_mean_percent':float(v.mean()),'test_macro_f1_sample_sd_percent':float(v.std(ddof=1))})
            for a,b in (('proto','fc'),('proto','resnet1d'),('resnet1d','fc')):
                d=values[a]-values[b]
                paired.append({'case':case,'k':k,'comparison':f'{a}_minus_{b}',
                    'mean_percentage_points':float(d.mean()),'sample_sd_percentage_points':float(d.std(ddof=1)),
                    'wins':int((d>0).sum()),'per_seed':dict(zip(('42','43','44'),d.tolist()))})
    dump(BASE/'SUMMARY.json',{'test_evaluated':True,'models':54,'seeds':[42,43,44],
        'std_definition':'sample SD, ddof=1','aggregates':aggregates,'paired_deltas':paired})
    with (BASE/'PER_SEED_RESULTS.csv').open('w',encoding='utf-8',newline='') as stream:
        w=csv.writer(stream)
        w.writerow(['case','k','seed','method','best_epoch','test_size','test_macro_f1','test_accuracy','test_balanced_accuracy','val_macro_f1','checkpoint_sha256','test_pool_sha256'])
        for r in rows:
            w.writerow([r[n] for n in ('case','k','seed','method','best_epoch','test_size')]+[r['metrics'][n] for n in ('macro_f1','accuracy','balanced_accuracy')]+[r['val_macro_f1'],r['checkpoint_sha256'],r['test_pool_sha256']])
    lines=['# 当前固定Test最终评估：CCSAE-Proto、CCSAE-FC、ResNet1D','',
        '仅加载此前由Val Macro-F1选出的最佳权重，完成两电路、K=10/20/40、种子42/43/44的54次固定Test评估。未训练、调参、Test选权重或模型重选。','',
        '## 最终Test Macro-F1','', '单位：%，3种子的均值 ± 样本标准差（ddof=1）。Test固定，种子改变Train抽样和训练。','',
        '| 电路 | 模型 | K=10 | K=20 | K=40 |','|---|---|---:|---:|---:|']
    for c in ('skl','abl'):
        for m in ('proto','fc','resnet1d'):
            cells=[]
            for k in (10,20,40):
                r=next(r for r in aggregates if (r['case'],r['k'],r['method'])==(c,k,m))
                cells.append(f"{r['test_macro_f1_mean_percent']:.2f} ± {r['test_macro_f1_sample_sd_percent']:.2f}")
            lines.append(f'| {c.upper()} | {NAMES[m]} | '+' | '.join(cells)+' |')
    lines+=['','## 冻结协议与边界','',
        '- 当前P1：原始Train/Test合并后，按float32整行指纹首出现去重，再按类70%/15%/15%划分，split_seed=20261009。不是历史Data_deduplicated测试集。',
        '- 使用完整固定Test：SKL 570条（标签0为30条，其余12类各45条）；ABL 465条（标签0为15条，其余10类各45条），未抽成5条/类。',
        '- 每组使用已保存K条/类Train与5条/类Val。标准化统计量只从该组Train拟合并冻结；三模型使用同一原始Test及同一组Train统计量，Test不参与拟合。',
        '- 最佳模型由40轮Stage2的最高Val Macro-F1选择，同分保留最早轮次；推理前54份权重指纹均与上轮独立核验记录相同。',
        '- Proto固定P1欧氏平方距离、温度1、不除嵌入维度；类原型仅由全部已标注K条/类Train的eval特征均值构建，不用Test拟合。',
        '- eval、批大小64、确定性FP32、关闭TF32，无增强或BN重新估计。沿用服务器原uv环境，4进程×4CPU线程执行推理。',
        '- CCSAE-FC/Proto有10轮重构+SupCon的Stage1，ResNet1D随机初始化、Stage1为0；分类阶段对齐相同Episode和400次SGD更新。这是已选定基线协议，不应作单独骨干差异的因果解释。',
        '- 当前模型Train/Val/Test波形指纹互斥，但数据集和部分波形已用于历史实验，不能称全新外部盲测；也缺少独立采集批次/电路板标识，整行去重不能证明物理采集独立性。','',
        '## 配对差值','', '同电路、K、种子配对；单位：百分点，均值 ± 样本标准差。','',
        '| 电路 | K | Proto − FC | Proto − ResNet1D |','|---|---:|---:|---:|']
    for c in ('skl','abl'):
        for k in (10,20,40):
            cells=[]
            for name in ('proto_minus_fc','proto_minus_resnet1d'):
                r=next(r for r in paired if (r['case'],r['k'],r['comparison'])==(c,k,name))
                cells.append(f"{r['mean_percentage_points']:+.2f} ± {r['sample_sd_percentage_points']:.2f}（胜{r['wins']}/3）")
            lines.append(f'| {c.upper()} | {k} | '+' | '.join(cells)+' |')
    lines+=['','## 验证与证据','',f'- 模型源码固定于`{COMMIT}`；仅增加独立Test脚本与输出。',
        '- 54份权重均复现原Val预测；独立重载54模型，用encode/features和classifier分开重建Test输出，逐项scores与预测完全一致。',
        '- 完整类别混淆矩阵复算Macro-F1，与sklearn一致；保存逐条预测、每类F1、混淆矩阵。权重、BN缓冲、旧文件、数据和包版本不变。',
        '- Proto另用广播平方距离与同一FP32特征的FP64距离检查舍入误差；正式指标仍用冻结FP32 cdist预测，诊断不替换预测或重选模型。',
        f"- 54模型首次Test推理窗口{e['wall_seconds']:.2f}秒（含进程启动、数据读取和Val复核；不含独立核验、报告及同步）。",
        '- 证据：TEST_MANIFEST、CHECKPOINT_MANIFEST、PREFLIGHT、EVALUATION、VERIFICATION、SUMMARY、PER_SEED_RESULTS及runs逐模型记录。',
        '- 旧训练报告test_evaluated:false保留当时状态，新报告为true。本地完整归档另有scores/特征/原型NPZ；GitHub仅上传文本证据及脚本，不上传波形、权重或环境。','']
    (BASE/'SERVER_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[:16]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','run','verify','report'))
    args=parser.parse_args()
    {'preflight':preflight,'run':run,'verify':verify,'report':report}[args.phase]()