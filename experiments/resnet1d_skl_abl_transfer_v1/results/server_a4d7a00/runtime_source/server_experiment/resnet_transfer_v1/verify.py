"""迁移实验的独立核验与Test门控：所有训练核验通过才使用ABL Test。

独立重放采样，检查新头/源骨干/冻结BN及实际400次更新。模型在常规上下文
创建，仅推理函数使用no_grad，避免inference tensors改变计算路径。
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import experiment as e
from experiment import ROOT, BASE, SEEDS, SHOTS, CONDITIONS, read, dump, file_sha, state_sha
from verify_shot_scale import manual_metrics
import torch


def job_path(stage,k,seed,condition):
    """源与目标完全分离；目标条件不覆盖彼此或历史输出。"""
    return BASE/'source'/f'seed{seed}' if stage=='source' else BASE/'target'/f'k{k}_seed{seed}'/condition


def get_data(stage,k,seed):
    """核验重新从只读原始数据恢复输入，禁止使用训练输出作为数据真值。"""
    if stage=='source':
        data,audit=e.source_data();assert audit==read(BASE/'provenance/source_split.json')
        return data
    split=e.make_fewshot_split(e.WORK/'Data.zip','abl',k=k,sample_seed=seed)
    assert split.audit==read(BASE/'provenance/splits'/f'abl_k{k}_seed{seed}.json')
    return {'train':(split.train_x,split.train_y),'val':(split.val_x,split.val_y)}


@torch.no_grad()
def independent_scores(model,values):
    """拆开features与classifier，独立重建固定batch64的推理。"""
    model.eval()
    return torch.cat([model.classifier(model.features(c.cuda().unsqueeze(1))) for c in values.split(64)])


def verify_job(stage,k,seed,condition):
    """逐模型检查初始化、训练更新、Val选模与真实预测，不只读取summary。"""
    job=job_path(stage,k,seed,condition);cfg=read(job/'config.json')
    classes=13 if stage=='source' else 11
    assert cfg['seed']==seed and cfg['condition']==condition and cfg['stage']==stage
    assert cfg['stage1_epochs']==0 and cfg['updates']==400
    assert cfg['lr']==(.0022 if stage=='source' else .0102)
    assert cfg['optimizer']=='SGD' and cfg['momentum']==.9 and cfg['weight_decay']==0. and cfg['grad_clip']==5.
    data=get_data(stage,k,seed); x,y=data['train']; labels=y.numpy()
    initial=torch.load(job/'initial.pt',map_location='cpu',weights_only=True)
    final=torch.load(job/'last.pt',map_location='cpu',weights_only=True)
    best=torch.load(job/'best.pt',map_location='cpu',weights_only=True)
    assert initial['config']==final['config']==best['config']==cfg and final['epoch']==40
    trace=read(job/'training_trace.json')
    assert trace['actual_updates']==400
    assert state_sha(initial['state_dict'])==trace['initial_state_sha256']
    assert state_sha(final['state_dict'])==trace['last_state_sha256']
    assert state_sha(e.feature_state(initial['state_dict']))==trace['initial_features_sha256']
    assert state_sha(e.feature_state(final['state_dict']))==trace['last_features_sha256']
    assert state_sha({n:t for n,t in initial['state_dict'].items() if n.startswith('classifier.')})==trace['initial_head_sha256']
    for cp in (initial,final,best):assert all(torch.isfinite(t).all() for t in cp['state_dict'].values())
    if stage=='source':
        e.set_seed(seed);model=e.ResNet1D(13).cuda();source_state=None
        assert cfg['source_split_audit_sha256']==file_sha(BASE/'provenance/source_split.json')
    else:
        assert cfg['k']==k and cfg['classes']==11
        source_path=ROOT/cfg['source_checkpoint'];assert file_sha(source_path)==cfg['source_checkpoint_sha256']
        source_state=torch.load(source_path,map_location='cpu',weights_only=True)['state_dict']
        model=e.target_initial(seed,condition,source_state)
        assert cfg['split_audit_sha256']==file_sha(BASE/'provenance/splits'/f'abl_k{k}_seed{seed}.json')
        if condition!='target_only':
            assert state_sha(e.feature_state(initial['state_dict']))==state_sha(e.feature_state(source_state))
        assert cfg['new_head_seed']==seed+2000 and cfg['bn_frozen']==(condition=='frozen_transfer')
    # 相同种子重建真实初值，验证Target-only没有暗中加载源域权重。
    assert state_sha(model.state_dict())==state_sha(initial['state_dict'])
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trace['trainable_parameters']==trainable
    frozen=condition=='frozen_transfer'
    assert trainable==(2827 if frozen else sum(p.numel() for p in model.parameters()))
    counters=[n for n in initial['state_dict'] if n.endswith('num_batches_tracked')]
    if frozen:
        assert trace['frozen_feature_eval_every_step'] and trace['frozen_parameters_and_bn_verified']
        assert not any(p.requires_grad for p in model.features.parameters())
        assert state_sha(e.feature_state(initial['state_dict']))==state_sha(e.feature_state(final['state_dict']))==state_sha(e.feature_state(best['state_dict']))
    else:
        assert all(p.requires_grad for p in model.parameters())
        names={n for n,_ in model.named_parameters() if n.startswith('features.')}
        assert any(not torch.equal(initial['state_dict'][n],final['state_dict'][n]) for n in names)
    assert any(not torch.equal(initial['state_dict'][n],final['state_dict'][n]) for n in ('classifier.weight','classifier.bias'))
    for n in counters:
        start=int(initial['state_dict'][n]);end=int(final['state_dict'][n]);b=int(best['state_dict'][n])
        assert end==start+(0 if frozen else 400)
        assert b==start+(0 if frozen else best['epoch']*10)
        assert trace['initial_bn_counters'][n]==start and trace['last_bn_counters'][n]==end
    # 不调用训练的sample_episode，独立用numpy重放每一步实际索引。
    arrays=np.load(job/'episodes.npz');rng=np.random.default_rng(seed+901)
    candidates=[np.flatnonzero(labels==c) for c in range(classes)]
    for part,size in (('support',3),('query',2)):
        assert arrays[part].shape==(400,classes*size)
        assert hashlib.sha256(arrays[part].tobytes()).hexdigest()==trace['episode_sha256'][part]
    for step in range(400):
        support,query=[],[]
        for c in range(classes):
            chosen=rng.choice(candidates[c],5,replace=False)
            support.extend(chosen[:3]);query.extend(chosen[3:])
        assert np.array_equal(support,arrays['support'][step]) and np.array_equal(query,arrays['query'][step])
        assert not set(support)&set(query)
    coverage=len(set(arrays['support'].flatten())|set(arrays['query'].flatten()))
    assert coverage==trace['train_row_coverage']
    if stage=='target' and seed in (42,43,44):
        old=np.load(ROOT/f'outputs/resnet_shot_scale/server_a4d7a00/runs/abl/k{k}_sample{seed}_train{seed}/episodes.npz')
        assert all(np.array_equal(arrays[n],old[n]) for n in ('support','query'))
    history=read(job/'history.json');assert [r['epoch'] for r in history]==list(range(1,41))
    assert all(np.isfinite(list(r.values())).all() for r in history)
    chosen=max(history,key=lambda r:r['macro_f1']);summary=read(job/'summary.json')
    assert chosen['epoch']==best['epoch']==summary['best_epoch']
    assert summary['test_evaluated'] is False and summary['checkpoint_sha256']==file_sha(job/'best.pt')
    model.load_state_dict(best['state_dict']);model.eval()
    scores=independent_scores(model,data['val'][0]);truth=data['val'][1].numpy();pred=scores.argmax(1).cpu().numpy()
    with np.load(job/'val_scores.npz') as saved:
        assert np.array_equal(truth,saved['truth']) and np.array_equal(pred,saved['predicted'])
        assert np.array_equal(scores.cpu().numpy(),saved['scores'])
    saved=read(job/'val_predictions.json');assert saved['truth']==truth.tolist() and saved['predicted']==pred.tolist()
    metrics,cm,f1=manual_metrics(truth,pred,classes)
    for n,v in metrics.items():
        assert np.isclose(v,summary['val'][n],rtol=0,atol=1e-12) and np.isclose(v,chosen[n],rtol=0,atol=1e-12)
    assert state_sha(model.state_dict())==state_sha(best['state_dict'])
    return {'stage':stage,'k':k,'seed':seed,'condition':condition,'status':'passed',
        'best_epoch':best['epoch'],'checkpoint':(job/'best.pt').relative_to(ROOT).as_posix(),
        'checkpoint_sha256':file_sha(job/'best.pt'),'best_state_sha256':state_sha(best['state_dict']),
        'initial_head_sha256':trace['initial_head_sha256'],'initial_features_sha256':trace['initial_features_sha256'],
        'episode_sha256':trace['episode_sha256'],'actual_updates':400,'train_rows_seen':coverage,
        'val_predictions_and_scores_exact':True,'val':metrics,'bn_counter_verified':True,
        'frozen_features_and_bn_unchanged':True if frozen else None,'test_evaluated':False}


def preservation():
    """原输入与包版本必须与训练前完全一致；新增输出不伪装成旧结果。"""
    protected=read(BASE/'provenance/input_before.json')
    assert all(file_sha(e.WORK/n)==h for n,h in protected.items())
    packages={d.metadata['Name']:d.version for d in e.importlib.metadata.distributions()}
    assert packages==read(BASE/'provenance/runtime_before.json')['packages']
    sources=read(BASE/'SOURCE_MANIFEST.json')
    assert all(file_sha(ROOT/n)==h for part in ('tracked_sources','server_helpers') for n,h in sources[part].items())
    return {'status':'passed','protected_files':len(protected),'old_files_unchanged':True,
        'source_data_and_environment_unchanged':True,'installed_or_downloaded_packages':False}


def verify_training():
    """100次训练全部核验后冻结90份目标最佳权重，开放一次统一Test阶段。"""
    e.initialize();results=[]
    assert read(BASE/'SOURCE_TRAINING.json')['models']==10 and read(BASE/'TARGET_TRAINING.json')['models']==90
    for seed in SEEDS:results.append(verify_job('source',None,seed,'source_pretrain'))
    for k in SHOTS:
        for seed in SEEDS:
            group=[verify_job('target',k,seed,c) for c in CONDITIONS];results.extend(group)
            assert len({r['initial_head_sha256'] for r in group})==1
            for part in ('support','query'):assert len({r['episode_sha256'][part] for r in group})==1
            assert group[1]['initial_features_sha256']==group[2]['initial_features_sha256']
    # 每个条件同种子跨K初始化不变，只有抽取的ABL训练规模改变。
    for seed in SEEDS:
        for condition in CONDITIONS:
            rows=[r for r in results if r['stage']=='target' and r['seed']==seed and r['condition']==condition]
            assert len({r['initial_head_sha256'] for r in rows})==len({r['initial_features_sha256'] for r in rows})==1
    keep=preservation()
    dump(BASE/'TRAINING_VERIFICATION.json',{'status':'passed','source_models':10,'target_models':90,
        'same_new_head_and_episode_per_group':True,'same_source_backbone_across_k':True,
        'source_and_target_best_val_selection_verified':True,'results':results,'preservation':keep,'test_evaluated':False})
    dump(BASE/'CHECKPOINT_MANIFEST.json',{'frozen_before_test_inference':True,'target_models':90,
        'source_models':10,'source_checkpoints':[r for r in results if r['stage']=='source'],
        'target_checkpoints':[r for r in results if r['stage']=='target']})
    print('训练独立核验通过：10源+90目标，400次实际更新、真实采样、新头、冻结骨干/BN和Val均一致；已冻结Test候选。',flush=True)


def test():
    """仅测试90份Val选定目标权重；完整固定ABL Test，无拟合或重选。"""
    e.initialize();assert read(BASE/'TRAINING_VERIFICATION.json')['status']=='passed'
    manifest=read(BASE/'CHECKPOINT_MANIFEST.json');results=[]
    for r in manifest['target_checkpoints']:
        k,seed,condition=r['k'],r['seed'],r['condition']
        audit=read(BASE/'provenance/splits'/f'abl_k{k}_seed{seed}.json')
        data=e.tensors('abl',audit);truth=data['test'][1].numpy()
        assert len(truth)==465
        job=job_path('target',k,seed,condition);assert not (job/'test_result.json').exists()
        path=ROOT/r['checkpoint'];assert file_sha(path)==r['checkpoint_sha256']
        cp=torch.load(path,map_location='cpu',weights_only=True);model=e.ResNet1D(11).cuda().eval()
        model.load_state_dict(cp['state_dict']);before=state_sha(model.state_dict())
        scores=e.predict(model,data['test'][0]);assert torch.isfinite(scores).all()
        pred=scores.argmax(1).cpu().numpy();metrics,cm,f1=manual_metrics(truth,pred,11)
        assert before==state_sha(model.state_dict())==r['best_state_sha256']
        np.savez_compressed(job/'test_scores.npz',truth=truth,predicted=pred,scores=scores.cpu().numpy())
        dump(job/'test_predictions.json',{'truth':truth.tolist(),'predicted':pred.tolist()})
        result={'k':k,'seed':seed,'condition':condition,'model':'ResNet1D','best_epoch':r['best_epoch'],
            'checkpoint_sha256':file_sha(path),'test_pool_sha256':audit['test_pool_sha256'],
            'test_size':465,'metrics':metrics,'confusion_matrix':cm.tolist(),'per_class_f1':f1.tolist(),
            'val':r['val'],'test_evaluated':True,'state_unchanged_after_inference':True}
        dump(job/'test_result.json',result);results.append(result);del model
    dump(BASE/'TEST_EVALUATION.json',{'status':'completed','models':90,'test_evaluated':True,
        'no_checkpoint_reselection':True,'fixed_test_size':465,'results':results})
    print('固定Test评估完成：90份Val最佳目标权重；没有源域Test推理、目标Test拟合或重选。',flush=True)


def verify_test():
    """再次重载90份权重，分解前向复现Test scores并独立复算全部指标。"""
    e.initialize();results=[]
    for row in read(BASE/'TEST_EVALUATION.json')['results']:
        k,seed,condition=row['k'],row['seed'],row['condition'];job=job_path('target',k,seed,condition)
        a=read(BASE/'provenance/splits'/f'abl_k{k}_seed{seed}.json');data=e.tensors('abl',a)
        path=job/'best.pt';assert file_sha(path)==row['checkpoint_sha256']
        cp=torch.load(path,map_location='cpu',weights_only=True);model=e.ResNet1D(11).cuda().eval();model.load_state_dict(cp['state_dict'])
        before=state_sha(model.state_dict());scores=independent_scores(model,data['test'][0])
        truth=data['test'][1].numpy();pred=scores.argmax(1).cpu().numpy()
        with np.load(job/'test_scores.npz') as saved:
            assert np.array_equal(truth,saved['truth']) and np.array_equal(pred,saved['predicted']) and np.array_equal(scores.cpu().numpy(),saved['scores'])
        metrics,cm,f1=manual_metrics(truth,pred,11);other=e.macro_metrics(truth.tolist(),pred.tolist(),11)
        assert cm.tolist()==row['confusion_matrix'] and f1.tolist()==row['per_class_f1']
        for n,v in metrics.items():
            assert np.isclose(v,row['metrics'][n],rtol=0,atol=1e-12) and np.isclose(v,other[n],rtol=0,atol=1e-12)
        assert before==state_sha(model.state_dict())
        results.append({'k':k,'seed':seed,'condition':condition,'status':'passed','scores_and_predictions_exact':True,
            'manual_and_sklearn_metrics_equal':True,'state_and_bn_unchanged':True,'checkpoint_sha256':file_sha(path)})
        del model
    dump(BASE/'TEST_VERIFICATION.json',{'status':'passed','models':90,'test_evaluated':True,'results':results,'preservation':preservation()})
    print('Test独立核验通过：90模型分数与预测逐位一致，混淆矩阵/指标一致，旧文件/数据/源码/包版本不变。',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('training','test','verify-test'))
    args=p.parse_args();{'training':verify_training,'test':test,'verify-test':verify_test}[args.phase]()