"""ResNet1D第一轮迁移：Target-only / Frozen / Fine-tuning三个条件。

只在新目录训练，原数据、源码、旧实验和uv环境只读。源域规模固定为全部SKL
Train；目标域只使用ABL每类K条及固定5条Val。先完成全部Val核验，再开放Test。
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse, hashlib, importlib.metadata, json, multiprocessing as mp
import subprocess, sys, time, traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT.parent
BASE=ROOT/'outputs/resnet_transfer_v1/server_a4d7a00'
FIXED=ROOT/'outputs/fixed_test/server_a4d7a00'
COMMIT='a4d7a008d7e7597a864624f63f4185d91fada144'
MODEL_SHA='b9c96fa1d78ca0c4a1e7a7626a4f5a6a5e29456faa091ff5db4dcd47048e5263'
SEEDS=tuple(range(42,52)); SHOTS=(10,20,40)
CONDITIONS=('target_only','frozen_transfer','finetuning_transfer')
NAMES={'target_only':'Target-only','frozen_transfer':'Frozen Transfer','finetuning_transfer':'Fine-tuning Transfer'}
sys.path.insert(0,str(ROOT/'server_experiment'))
sys.path.insert(0,str(ROOT/'server_experiment/resnet_shot_scale'))
sys.path.insert(0,str(ROOT/'server_experiment/fixed_test'))
from common import configure, dump, file_sha, state_sha
from run_shot_scale import Resources
from reference_models import ResNet1D
from evaluate import pool, tensors, read
import torch, ccsae
from torch import nn
from ccsae.fewshot_data import make_fewshot_split
from ccsae.fewshot_train import set_seed
from ccsae.prototype import sample_episode, macro_metrics


def feature_state(state):
    """仅提取骨干参数和BN缓冲；源13类分类头始终不迁移到ABL。"""
    return {n:t for n,t in state.items() if n.startswith('features.')}


def cpu_state(model):
    """复制全部参数/缓冲，避免继续训练改变已保存的状态。"""
    return {n:t.detach().cpu().clone() for n,t in model.state_dict().items()}


def source_data():
    """源域全部2660条Train拟合标量归一化；Val沿用固定65条。

    当前P1完整池由已冻结的首出现float32规则恢复，不调用历史CSV划分。
    SKL Test没有进入训练、选模或网络推理。
    """
    x,y,hs,pools,fp=pool('skl')
    a=make_fewshot_split(WORK/'Data.zip','skl',k=40,sample_seed=42).audit
    lookup={h:i for i,h in enumerate(hs)}
    ids={ 'train':pools['train'], 'val':[lookup[h] for c in range(13) for h in a['selected_val_hashes_by_class'][str(c)]]}
    raw=x[ids['train']].astype(np.float64)
    mean,std=float(raw.mean()),max(float(raw.std()),1e-8)
    data={n:(torch.from_numpy(((x[v].astype(np.float64)-mean)/std).astype(np.float32)).contiguous(),
             torch.from_numpy(y[v].copy())) for n,v in ids.items()}
    audit={'case':'skl','classes':13,'train_size':2660,'val_size':65,'split_seed':20261009,
        'train_scaler_mean':mean,'train_scaler_std':std,'pool_sha256':fp,
        'train_hashes_ordered':[hs[i] for i in ids['train']], 'train_labels_ordered':y[ids['train']].tolist(),
        'val_hashes_ordered':[hs[i] for i in ids['val']], 'val_labels_ordered':y[ids['val']].tolist()}
    assert len(data['train'][1])==2660 and len(data['val'][1])==65
    assert np.bincount(data['val'][1].numpy()).tolist()==[5]*13
    return data,audit


def target_initial(seed,condition,source_state):
    """三条件使用相同新11类头；同种子不同K也复用同一初始化。

    Target-only为随机骨干；两个迁移条件只复制同一源最佳骨干。新头用独立
    固定随机种子初始化，避免源13类/目标11类形状或加载顺序影响头初值。
    """
    set_seed(seed)
    model=ResNet1D(11).cuda()
    if condition!='target_only':
        state=model.state_dict(); state.update(feature_state(source_state)); model.load_state_dict(state,strict=True)
    set_seed(seed+2000)
    model.classifier=nn.Linear(256,11).cuda()
    if condition=='frozen_transfer':
        model.features.requires_grad_(False)
        model.features.eval()
    return model


@torch.no_grad()
def predict(model,x):
    """固定batch64的eval推理；没有BN重估计、测试增强或梯度。"""
    model.eval()
    return torch.cat([model(chunk.cuda().unsqueeze(1)) for chunk in x.split(64)])


def initialize():
    """保持原环境的确定性FP32、禁TF32，每进程4个CPU线程。"""
    configure()
    assert Path(ccsae.__file__).resolve()==ROOT/'src/ccsae/__init__.py'


def preflight():
    """看新结果前冻结输入、30组目标划分、源码、旧文件与统计计划。"""
    assert not BASE.exists(), '输出已存在，禁止覆盖或偷偷重跑'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==COMMIT
    assert file_sha(ROOT/'server_experiment/resnet_shot_scale/reference_models.py')==MODEL_SHA
    old=read(FIXED/'SOURCE_MANIFEST.json')
    assert all(file_sha(ROOT/n)==h for n,h in old['tracked_sources'].items())
    assert read(FIXED/'COMPLETION_VERIFICATION.json')['status']=='passed'
    protected=read(FIXED/'provenance/input_before.json')
    for folder in (FIXED,ROOT/'server_experiment/fixed_test'):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                protected[p.relative_to(WORK).as_posix()]=file_sha(p)
    assert all(file_sha(WORK/n)==h for n,h in protected.items())
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    assert packages==read(FIXED/'provenance/runtime_before.json')['packages']
    source,sa=source_data()
    target_tests={c['case']:c for c in read(FIXED/'TEST_MANIFEST.json')['cases']}
    assert sa['pool_sha256']==target_tests['skl']['pool_sha256']
    xs,ys,hs,sp,sfp=pool('skl'); xa,ya,ha,ap,afp=pool('abl')
    assert not set(hs)&set(ha), '发现源域与目标域完全重复波形，需先处理泄漏'
    jobs=[]; audits={}
    for seed in SEEDS:
        previous=None
        for k in SHOTS:
            split=make_fewshot_split(WORK/'Data.zip','abl',k=k,sample_seed=seed)
            a=split.audit; assert a['test_pool_sha256']==target_tests['abl']['pool_sha256']['test']
            if previous:
                for c,hashes in previous['selected_train_hashes_by_class'].items():
                    assert hashes==a['selected_train_hashes_by_class'][c][:previous['k']]
            previous=a
            if seed in (42,43,44):
                assert a==read(ROOT/f'outputs/resnet_shot_scale/server_a4d7a00/runs/abl/k{k}_sample{seed}_train{seed}/split_audit.json')
            assert a['selected_val_hashes_by_class']==make_fewshot_split(WORK/'Data.zip','abl',k=20,sample_seed=42).audit['selected_val_hashes_by_class']
            # 沿用上一阶段的独立指纹回查和Train-only统计量核验，不做任何Test推理。
            tensors('abl',a)
            audits[f'abl_k{k}_seed{seed}.json']=a
            jobs.append({'k':k,'seed':seed,'train_size':11*k,'val_size':55,'test_size':465})
    BASE.mkdir(parents=True);(BASE/'provenance/splits').mkdir(parents=True)
    for n,a in audits.items():dump(BASE/'provenance/splits'/n,a)
    dump(BASE/'provenance/source_split.json',sa)
    dump(BASE/'provenance/input_before.json',protected)
    dump(BASE/'provenance/runtime_before.json',{'packages':packages,'executable':sys.executable,
        'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'environment_reused':True,'installed_or_downloaded_packages':False})
    protocol={'model':'original ResNet1D','source_commit':COMMIT,'model_sha256':MODEL_SHA,
        'seeds':SEEDS,'k':SHOTS,'conditions':CONDITIONS,'source_train':2660,'source_val':65,
        'source_classes':13,'target_classes':11,'target_val':55,'target_test':465,
        'source_optimizer':{'name':'SGD','lr':.0022,'momentum':.9,'weight_decay':0.,'grad_clip':5.},
        'target_optimizer':{'name':'SGD','lr':.0102,'momentum':.9,'weight_decay':0.,'grad_clip':5.},
        'updates':400,'validation_cycles':40,'steps_per_cycle':10,'support_per_class':3,'query_per_class':2,
        'stage1_epochs':0,'scheduler':None,'early_stopping':False,'bn_recalibration':False,
        'frozen_backbone_parameters_and_bn_buffers':True,'frozen_training_mode':'features.eval; classifier.train',
        'fine_tuning':'all parameters and training BN updated; fresh optimizer',
        'target_only':'random backbone and same new classifier initialization',
        'target_scaler':'selected ABL K-shot Train only, shared across conditions',
        'source_scaler':'full SKL Train only','head_seed_offset':2000,'episode_seed_offset':901,
        'selection':'maximum Val Macro-F1, earliest epoch on exact tie','batch_size_eval':64,
        'workers':4,'threads_per_worker':4,'test_gate':'all 10 source and 90 target training checks must pass',
        'statistics':{'primary_setting':'K20 fine_tuning vs target_only; reported within family of 6',
            'comparisons':2,'k_settings':3,'family_size':6,'test':'two-sided exact paired sign-flip permutation of mean difference',
            'sign_flip_configurations':1024,'multiple_comparison':'Holm FWER .05',
            'ci':'paired seed bootstrap 95% percentile CI, 20000 resamples','bootstrap_seed':20261011,
            'effect_unit':'Macro-F1 percentage points','independent_unit':'paired training/sampling seed; fixed shared Test',
            'significance_interpretation':'positive gain with Holm p<.05; exploratory, conditional on fixed Test and ten seeds'},
        'limitations':['Test was evaluated in previous research; not pristine external blind testing',
            'No physical acquisition groups available; exact dedup does not establish board/batch independence',
            'Frozen vs Fine differs in both weights and BN adaptation; not a parameter-only BN ablation',
            '10 seeds are prespecified first-round repeats, not a power guarantee',
            'Same target labels per run: 11*K Train plus 55 Val; source labels and compute are additional resources']}
    dump(BASE/'PROTOCOL.json',protocol)
    paths=[ROOT/'server_experiment/common.py',ROOT/'server_experiment/run_shot_scale.py',ROOT/'server_experiment/resnet_shot_scale/reference_models.py',ROOT/'server_experiment/fixed_test/evaluate.py']
    paths += sorted(Path(__file__).parent.glob('*.py'))
    dump(BASE/'SOURCE_MANIFEST.json',{'source_commit':COMMIT,'tracked_sources':old['tracked_sources'],
        'server_helpers':{p.relative_to(ROOT).as_posix():file_sha(p) for p in paths}})
    dump(BASE/'PREFLIGHT.json',{'status':'passed','source_models':10,'target_models':90,'target_groups':30,
        'protected_files':len(protected),'cross_circuit_exact_waveform_overlap':0,
        'current_pools_and_fixed_val_test_verified':True,'nested_train_prefixes_verified':True,
        'test_evaluated':False,'jobs':jobs,'utc':datetime.now(timezone.utc).isoformat()})
    print(f'预检通过：10个源模型、30组三条件目标配对；保护{len(protected)}份旧文件。',flush=True)


def train_model(model,data,config,out):
    """统一400次CE分类更新，记录实际Episode、BN计数和最优Val轮次。

    Frozen仅训练新头：骨干不接收梯度且BN始终eval，不通过train()悄悄适配。
    所有条件都用相同输入顺序Support在前、Query在后，Val不参加梯度。
    """
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'config.json',config)
    initial=cpu_state(model); torch.save({'state_dict':initial,'config':config},out/'initial.pt')
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.SGD(params,lr=config['lr'],momentum=.9,weight_decay=0.)
    rng=np.random.default_rng(config['seed']+901)
    x,y=data['train']; vx,vy=data['val']; classes=config['classes']
    streams={'support':[],'query':[]}; history=[]; best=-1.; epoch_best=None; updates=0
    started=time.perf_counter(); mode_ok=True
    with (out/'train.log').open('w',encoding='utf-8',buffering=1) as log:
        for epoch in range(1,41):
            model.train()
            if config['condition']=='frozen_transfer':model.features.eval()
            loss_sum=0.
            for step in range(10):
                s,q=sample_episode(y,classes,3,2,rng); ids=torch.cat((s,q))
                streams['support'].append(s.numpy().copy()); streams['query'].append(q.numpy().copy())
                inputs=x[ids].cuda().unsqueeze(1)
                if config['condition']=='frozen_transfer':
                    mode_ok=mode_ok and not any(m.training for m in model.features.modules())
                    with torch.no_grad():features=model.features(inputs)
                    logits=model.classifier(features)
                else:logits=model(inputs)
                loss=nn.functional.cross_entropy(logits,y[ids].cuda())
                if not torch.isfinite(loss):raise FloatingPointError('训练损失非有限值')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True)
                optimizer.step();updates+=1;loss_sum+=float(loss.detach())
            scores=predict(model,vx);metrics=macro_metrics(vy.tolist(),scores.argmax(1).cpu().tolist(),classes)
            row={'epoch':epoch,'train_loss':loss_sum/10,**metrics};history.append(row)
            if metrics['macro_f1']>best:
                best=metrics['macro_f1'];epoch_best=epoch
                torch.save({'state_dict':cpu_state(model),'epoch':epoch,'config':config},out/'best.pt')
            log.write(f"epoch={epoch} updates={updates} loss={row['train_loss']:.6f} val_F1={metrics['macro_f1']:.6f} best={best:.6f}\n")
            dump(out/'progress.json',{'state':'running','epoch':epoch,'updates':updates,'test_evaluated':False})
    last=cpu_state(model);torch.save({'state_dict':last,'epoch':40,'config':config},out/'last.pt')
    dump(out/'history.json',history)
    arrays={n:np.stack(v) for n,v in streams.items()};np.savez_compressed(out/'episodes.npz',**arrays)
    cp=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(cp['state_dict'])
    scores=predict(model,vx);pred=scores.argmax(1).cpu().tolist()
    metrics=macro_metrics(vy.tolist(),pred,classes)
    np.savez_compressed(out/'val_scores.npz',truth=vy.numpy(),predicted=np.array(pred),scores=scores.cpu().numpy())
    dump(out/'val_predictions.json',{'truth':vy.tolist(),'predicted':pred})
    frozen=config['condition']=='frozen_transfer'
    if frozen:
        assert mode_ok and state_sha(feature_state(initial))==state_sha(feature_state(last))==state_sha(feature_state(cp['state_dict']))
    trace={'actual_updates':updates,'trainable_parameters':sum(p.numel() for p in params),
        'initial_state_sha256':state_sha(initial),'last_state_sha256':state_sha(last),
        'initial_features_sha256':state_sha(feature_state(initial)),'last_features_sha256':state_sha(feature_state(last)),
        'initial_head_sha256':state_sha({n:t for n,t in initial.items() if n.startswith('classifier.')}),
        'episode_sha256':{n:hashlib.sha256(v.tobytes()).hexdigest() for n,v in arrays.items()},
        'train_row_coverage':len(set(arrays['support'].flatten())|set(arrays['query'].flatten())),
        'frozen_parameters_and_bn_verified':True if frozen else None,'frozen_feature_eval_every_step':mode_ok if frozen else None,
        'initial_bn_counters':{n:int(t) for n,t in initial.items() if n.endswith('num_batches_tracked')},
        'last_bn_counters':{n:int(t) for n,t in last.items() if n.endswith('num_batches_tracked')}}
    assert updates==400
    dump(out/'training_trace.json',trace)
    torch.cuda.synchronize()
    summary={'stage':config['stage'],'condition':config['condition'],'k':config.get('k'),'seed':config['seed'],
        'best_epoch':epoch_best,'val':metrics,'parameters':sum(p.numel() for p in model.parameters()),
        'train_size':len(y),'val_size':len(vy),'checkpoint_sha256':file_sha(out/'best.pt'),
        'wall_seconds':time.perf_counter()-started,'test_evaluated':False}
    dump(out/'summary.json',summary);dump(out/'progress.json',{'state':'complete','updates':updates,'test_evaluated':False})
    return summary


def source_job(seed):
    """10次独立SKL预训练；相同种子的最佳源骨干供所有K与两个迁移条件共享。"""
    set_seed(seed);data,audit=source_data();model=ResNet1D(13).cuda()
    cfg={'stage':'source','condition':'source_pretrain','seed':seed,'classes':13,'lr':.0022,
        'stage1_epochs':0,'updates':400,'optimizer':'SGD','momentum':.9,'weight_decay':0.,'grad_clip':5.,
        'source_split_audit_sha256':file_sha(BASE/'provenance/source_split.json'),'source_commit':COMMIT}
    return train_model(model,data,cfg,BASE/'source'/f'seed{seed}')


def target_group(k,seed):
    """顺序执行配对三条件，保证同一源权重、头初值和实际Episode流。"""
    split=make_fewshot_split(WORK/'Data.zip','abl',k=k,sample_seed=seed)
    assert split.audit==read(BASE/'provenance/splits'/f'abl_k{k}_seed{seed}.json')
    data={'train':(split.train_x,split.train_y),'val':(split.val_x,split.val_y)}
    src=BASE/'source'/f'seed{seed}/best.pt'
    state=torch.load(src,map_location='cpu',weights_only=True)['state_dict'];results=[]
    for condition in CONDITIONS:
        model=target_initial(seed,condition,state)
        cfg={'stage':'target','condition':condition,'k':k,'seed':seed,'classes':11,'lr':.0102,
            'stage1_epochs':0,'updates':400,'optimizer':'SGD','momentum':.9,'weight_decay':0.,'grad_clip':5.,
            'split_audit_sha256':file_sha(BASE/'provenance/splits'/f'abl_k{k}_seed{seed}.json'),
            'source_checkpoint':src.relative_to(ROOT).as_posix(),'source_checkpoint_sha256':file_sha(src),
            'new_head_seed':seed+2000,'source_commit':COMMIT,'bn_frozen':condition=='frozen_transfer'}
        results.append(train_model(model,data,cfg,BASE/'target'/f'k{k}_seed{seed}'/condition))
        del model
    return results


def run_stage(stage):
    """四进程并发，记录进度/真实资源和失败；失败不自动重试。"""
    assert read(BASE/'PREFLIGHT.json')['status']=='passed'
    if stage=='target':assert read(BASE/'SOURCE_TRAINING.json')['models']==10
    folder=BASE/stage;folder.mkdir(exist_ok=False)
    tasks=[(seed,) for seed in SEEDS] if stage=='source' else [(k,seed) for k in SHOTS for seed in SEEDS]
    function=source_job if stage=='source' else target_group
    started=time.perf_counter();utc=datetime.now(timezone.utc).isoformat()
    resources=Resources();resources.start();results=[];completed=0
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize) as p:
            futures=[p.submit(function,*t) for t in tasks]
            for f in as_completed(futures):
                value=f.result();results.extend(value if stage=='target' else [value]);completed+=1
                dump(BASE/f'{stage}_progress.json',{'completed_groups':completed,'total_groups':len(tasks),'models':len(results)})
                print(f'{stage} 完成{completed}/{len(tasks)}组，累计{len(results)}模型。',flush=True)
    except BaseException:
        dump(BASE/f'{stage}_failure.json',{'error':traceback.format_exc(),'completed_models':len(results)})
        raise
    finally:dump(BASE/f'{stage}_resource_usage.json',resources.finish())
    dump(BASE/f'{stage.upper()}_TRAINING.json',{'status':'completed','models':len(results),'test_evaluated':False,
        'started_utc':utc,'ended_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-started,
        'workers':4,'threads_per_worker':4,'results':results})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('preflight','source','target'))
    args=p.parse_args()
    preflight() if args.phase=='preflight' else run_stage(args.phase)