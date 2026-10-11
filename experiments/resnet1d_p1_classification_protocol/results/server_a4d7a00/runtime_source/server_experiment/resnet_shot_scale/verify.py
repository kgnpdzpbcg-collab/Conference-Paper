"""独立检查原ResNet的18次训练：输入、真实采样、400次更新和最佳Val预测。

重载初始化/最后/最佳状态，并用原模型特征层和分类层重建输出。
不延长训练、不重新选择权重、不执行Test推理。
"""
from pathlib import Path
import hashlib
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'server_experiment'))
from common import configure, dump, file_sha, state_sha
from verify_shot_scale import manual_metrics
from audit import BASE, WORK, read, reference_job, MODEL_SHA
from reference_models import ResNet1D
import torch
from ccsae.fewshot_data import make_fewshot_split
from ccsae.fewshot_train import set_seed


@torch.no_grad()
def verify_job(job):
    """逐组复算，不以保存的summary或同一指标函数作为唯一判断依据。"""
    cfg=read(job/'config.json')
    case,k,seed=cfg['case'],cfg['k'],cfg['seed']
    assert case in ('skl','abl') and k in (10,20,40) and seed in (42,43,44)
    assert cfg['stage1_epochs']==0 and cfg['initialization']=='random'
    assert cfg['stage2_epochs']==40 and cfg['steps_per_epoch']==10
    assert cfg['support']==3 and cfg['query']==2
    assert cfg['optimizer']=='SGD' and cfg['momentum']==.9 and cfg['weight_decay']==0
    assert cfg['grad_clip']==5 and cfg['lr']==(.0022 if case=='skl' else .0102)
    assert cfg['scheduler'] is None and not cfg['early_stopping'] and not cfg['bn_recalibration']
    assert cfg['model_source_sha256']==MODEL_SHA and cfg['test_evaluated'] is False
    split=make_fewshot_split(WORK/'Data.zip',case,k=k,sample_seed=seed)
    assert split.audit==read(job/'split_audit.json')==read(BASE/'provenance/splits'/f'{case}_k{k}_seed{seed}.json')
    classes=split.audit['classes']
    arrays=np.load(job/'episodes.npz')
    reference=np.load(reference_job(case,k,seed)/'episodes.npz')
    y=split.train_y.numpy()
    rng=np.random.default_rng(seed+901)
    trace=read(job/'training_verification.json')
    assert trace['actual_updates']==400
    for part,size in (('support',3),('query',2)):
        assert arrays[part].shape==(400,classes*size)
        assert np.array_equal(arrays[part],reference[f'fc_{part}'])
        assert hashlib.sha256(arrays[part].tobytes()).hexdigest()==trace['episode_stream_sha256'][part]
    # 独立重放numpy采样，不调用训练入口或上游sample_episode。
    for step in range(400):
        s,q=[],[]
        for c in range(classes):
            ids=rng.choice(np.flatnonzero(y==c),5,replace=False)
            s.extend(ids[:3]);q.extend(ids[3:])
        assert np.array_equal(s,arrays['support'][step]) and np.array_equal(q,arrays['query'][step])
        assert not set(s)&set(q)
        assert np.bincount(y[s],minlength=classes).tolist()==[3]*classes
        assert np.bincount(y[q],minlength=classes).tolist()==[2]*classes
    coverage=len(set(arrays['support'].flatten())|set(arrays['query'].flatten()))
    assert coverage==trace['train_row_coverage']
    initial=torch.load(job/'initial.pt',map_location='cpu',weights_only=True)['state_dict']
    final=torch.load(job/'last.pt',map_location='cpu',weights_only=True)
    best=torch.load(job/'best.pt',map_location='cpu',weights_only=True)
    assert final['epoch']==40
    assert state_sha(initial)==trace['initial_state_sha256']
    assert state_sha(final['state_dict'])==trace['last_state_sha256']
    # 相同种子直接重建随机初始化，验证没有加载CCSAE或历史ResNet的预训练权重。
    set_seed(seed)
    model=ResNet1D(classes).cuda()
    assert state_sha(model.state_dict())==state_sha(initial)
    parameter_names={n for n,_ in model.named_parameters()}
    assert any(not torch.equal(initial[n],final['state_dict'][n]) for n in parameter_names if n.startswith('features.'))
    assert any(not torch.equal(initial[n],final['state_dict'][n]) for n in parameter_names if n.startswith('classifier.'))
    for state in (initial,final['state_dict'],best['state_dict']):
        assert set(state)==set(initial)
        assert all(torch.isfinite(t).all() for t in state.values())
    counters=[n for n in initial if n.endswith('num_batches_tracked')]
    assert counters
    for n in counters:
        assert int(initial[n])==0 and int(final['state_dict'][n])==400
        assert int(best['state_dict'][n])==best['epoch']*10
        assert trace['last_bn_batches_tracked'][n]==400
    history=read(job/'history.json')
    assert [r['epoch'] for r in history]==list(range(1,41))
    assert all(np.isfinite(list(r.values())).all() for r in history)
    selected=max(history,key=lambda r:r['macro_f1'])
    summary=read(job/'summary.json')
    assert best['epoch']==selected['epoch']==summary['best_epoch']
    assert best['config']==cfg and final['config']==cfg
    assert summary['test_evaluated'] is False and summary['checkpoint_sha256']==file_sha(job/'best.pt')
    assert summary['parameters']==sum(p.numel() for p in model.parameters())
    old=read(WORK/f'analog-circuit-fewshot/outputs/server_fewshot_v1/{case}_resnet1d_k20_seed42/result.json')
    assert old['parameters']==summary['parameters'], '原基线结构/参数量改变'
    model.load_state_dict(best['state_dict']);model.eval()
    # 分离features和classifier重建实际推理，验证原forward和保存结果的对应关系。
    scores=torch.cat([model.classifier(model.features(chunk.cuda().unsqueeze(1))) for chunk in split.val_x.split(64)])
    assert torch.isfinite(scores).all()
    truth=split.val_y.numpy();predicted=scores.argmax(1).cpu().numpy()
    saved=read(job/'val_predictions.json')
    assert saved['truth']==truth.tolist() and saved['predicted']==predicted.tolist()
    metrics,matrix,f1=manual_metrics(truth,predicted,classes)
    for n,value in metrics.items():
        assert np.isclose(value,summary['val'][n],rtol=0,atol=1e-12)
        assert np.isclose(value,selected[n],rtol=0,atol=1e-12)
    np.savez_compressed(job/'verified_val.npz',truth=truth,predicted=predicted,scores=scores.cpu().numpy())
    return {'case':case,'k':k,'seed':seed,'status':'passed','metrics':metrics,
         'best_epoch':best['epoch'],'confusion_matrix':matrix.tolist(),'per_class_f1':f1.tolist(),
         'initial_state_sha256':state_sha(initial),'best_checkpoint_sha256':file_sha(job/'best.pt'),
         'random_initialization_exact':True,'episode_indices_exact_to_ccsae_fc':True,
         'independent_episode_replay_verified':True,'bn_counter_updates_verified':400,
         'backbone_and_classifier_updated':True,'validation_predictions_exact':True,
         'train_rows_seen':coverage,'train_size':len(y),'test_evaluated':False}


def main():
    """18次全部通过才产生总核验；同种子跨K的ResNet初始化也必须一致。"""
    configure()
    jobs=sorted((BASE/'runs').glob('*/k*_sample*_train*'))
    assert len(jobs)==18
    results=[verify_job(job) for job in jobs]
    assert {(r['case'],r['k'],r['seed']) for r in results}=={(c,k,s) for c in ('skl','abl') for k in (10,20,40) for s in (42,43,44)}
    for case in ('skl','abl'):
        for seed in (42,43,44):
            assert len({r['initial_state_sha256'] for r in results if r['case']==case and r['seed']==seed})==1
    dump(BASE/'VERIFICATION.json',{'status':'passed','runs':18,'test_evaluated':False,
         'same_seed_initial_state_equal_across_k':True,'results':results})
    print('独立核验通过：18次训练，随机初始化/400步真实采样/BN计数/最佳权重/Val预测和指标一致，无Test推理。')

if __name__=='__main__':
    main()
