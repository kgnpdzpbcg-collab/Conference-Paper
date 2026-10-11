"""原ResNet1D的分类协议对照：18个K/电路/种子组合，随机初始化、400次更新。

逐步沿用CCSAE-FC的Episode顺序、分类交叉熵、SGD和验证选模，不添加Stage1。
所有原模型源码和旧结果只读；每组训练输出到新的独立目录。
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import multiprocessing as mp
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'server_experiment'))
from common import configure, dump, file_sha, state_sha
from run_shot_scale import Resources
from audit import BASE, COMMIT, MODEL_SHA, read, reference_job
from reference_models import ResNet1D
import numpy as np
import torch
from torch import nn
from ccsae.fewshot_data import make_fewshot_split
from ccsae.fewshot_train import set_seed
from ccsae.prototype import sample_episode, macro_metrics


def initialize():
    """每进程四CPU线程、确定性FP32；复用原环境，禁止TF32和非math SDP。"""
    configure()
    assert file_sha(Path(__file__).parent/'reference_models.py') == MODEL_SHA


@torch.no_grad()
def predict(model, values):
    """验证采用上游batch64、eval模式；仅增加输入通道维度，不修改波形值。"""
    model.eval()
    return torch.cat([model(chunk.cuda().unsqueeze(1)) for chunk in values.split(64)])


def cpu_state(model):
    """复制参数与BN缓冲区，保存可独立检查的完整初始化/最佳/最后状态。"""
    return {name:tensor.detach().cpu().clone() for name,tensor in model.state_dict().items()}


def run_one(case, k, seed):
    """每组只执行预定400次分类更新；支持和查询均参与FC分类损失。"""
    set_seed(seed)
    job = BASE/'runs'/case/f'k{k}_sample{seed}_train{seed}'
    job.mkdir(parents=True,exist_ok=False)
    split=make_fewshot_split(ROOT.parent/'Data.zip',case,k=k,sample_seed=seed)
    assert split.audit == read(BASE/'provenance/splits'/f'{case}_k{k}_seed{seed}.json')
    classes=split.audit['classes']
    lr=.0022 if case=='skl' else .0102
    cfg={'model':'ResNet1D','case':case,'k':k,'sample_seed':seed,'seed':seed,
         'split_seed':20261009,'val_seed':20261009,'val_per_class':5,
         'stage1_epochs':0,'initialization':'random','stage2_epochs':40,'steps_per_epoch':10,
         'support':3,'query':2,'supervision':'cross_entropy_on_support_plus_query',
         'optimizer':'SGD','lr':lr,'momentum':.9,'weight_decay':0.,'grad_clip':5.,
         'scheduler':None,'early_stopping':False,'bn_recalibration':False,
         'validation_batch_size':64,'selection':'max_val_macro_f1_first_tie',
         'device':'cuda','test_evaluated':False,'source_commit':COMMIT,'model_source_sha256':MODEL_SHA}
    dump(job/'config.json',cfg)
    dump(job/'split_audit.json',split.audit)
    model=ResNet1D(classes).cuda()
    assert all(p.dtype==torch.float32 for p in model.parameters())
    initial=cpu_state(model)
    torch.save({'state_dict':initial,'config':cfg},job/'initial.pt')
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9)
    rng=np.random.default_rng(seed+901)
    old=np.load(reference_job(case,k,seed)/'episodes.npz')
    supports,queries=[],[]
    history=[]
    best_f1,best_epoch=-1.,None
    started=time.perf_counter()
    try:
        with (job/'train.log').open('w',encoding='utf-8',buffering=1) as log:
            for epoch in range(1,41):
                model.train()
                total=0.
                for inner in range(10):
                    s,q=sample_episode(split.train_y,classes,3,2,rng)
                    step=(epoch-1)*10+inner
                    assert np.array_equal(s.numpy(),old['fc_support'][step])
                    assert np.array_equal(q.numpy(),old['fc_query'][step])
                    supports.append(s.numpy().copy());queries.append(q.numpy().copy())
                    # 输入排列与CCSAE-FC一致：全部Support在前，全部Query在后。
                    ids=torch.cat((s,q))
                    logits=model(split.train_x[ids].cuda().unsqueeze(1))
                    loss=nn.functional.cross_entropy(logits,split.train_y[ids].cuda())
                    if not torch.isfinite(loss):
                        raise FloatingPointError('ResNet分类损失非有限值')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                    optimizer.step()
                    total+=float(loss.detach())
                scores=predict(model,split.val_x)
                metrics=macro_metrics(split.val_y.tolist(),scores.argmax(1).cpu().tolist(),classes)
                row={'epoch':epoch,'train_loss':total/10,**metrics}
                history.append(row)
                if metrics['macro_f1']>best_f1:
                    best_f1,best_epoch=metrics['macro_f1'],epoch
                    torch.save({'state_dict':cpu_state(model),'epoch':epoch,'config':cfg},job/'best.pt')
                log.write(f"epoch={epoch} train={row['train_loss']:.6f} val_F1={metrics['macro_f1']:.6f} best={best_f1:.6f}\n")
                dump(job/'progress.json',{'state':'running','epoch':epoch,'updates':epoch*10})
        final=cpu_state(model)
        torch.save({'state_dict':final,'epoch':40,'config':cfg},job/'last.pt')
        dump(job/'history.json',history)
        arrays={'support':np.stack(supports),'query':np.stack(queries)}
        np.savez_compressed(job/'episodes.npz',**arrays)
        assert all(np.array_equal(arrays[name],old[f'fc_{name}']) for name in arrays)
        model.load_state_dict(torch.load(job/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
        scores=predict(model,split.val_x)
        predicted=scores.argmax(1).cpu().tolist()
        metrics=macro_metrics(split.val_y.tolist(),predicted,classes)
        dump(job/'val_predictions.json',{'truth':split.val_y.tolist(),'predicted':predicted})
        torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        trace={'initial_state_sha256':state_sha(initial),'last_state_sha256':state_sha(final),
               'episode_stream_sha256':{n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in arrays.items()},
               'actual_updates':400,'reference_episode_indices_exact':True,
               'train_row_coverage':len(set(arrays['support'].flatten())|set(arrays['query'].flatten())),
               'last_bn_batches_tracked':{n:int(t) for n,t in final.items() if n.endswith('num_batches_tracked')},
               'wall_seconds':seconds}
        dump(job/'training_verification.json',trace)
        result={'case':case,'k':k,'seed':seed,'stage1_epochs':0,'best_epoch':best_epoch,
                'val':metrics,'train_size':len(split.train_y),'val_size':len(split.val_y),
                'parameters':sum(p.numel() for p in model.parameters()),'wall_seconds':seconds,
                'test_evaluated':False,'checkpoint_sha256':file_sha(job/'best.pt')}
        dump(job/'summary.json',result)
        dump(job/'progress.json',{'state':'complete','epoch':40,'updates':400})
        return result
    except BaseException:
        dump(job/'failure.json',{'error':traceback.format_exc()})
        raise


def main():
    """四进程并发，不重跑旧CCSAE或ResNet结果，不根据Val调参。"""
    assert read(BASE/'provenance/preflight.json')['status']=='passed'
    run_dir=BASE/'runs'
    run_dir.mkdir(exist_ok=False)
    tasks=[(case,k,seed) for case in ('skl','abl') for k in (10,20,40) for seed in (42,43,44)]
    started=time.perf_counter()
    utc=datetime.now(timezone.utc).isoformat()
    resources=Resources();resources.start()
    results=[]
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
            pending=[pool.submit(run_one,*task) for task in tasks]
            for future in as_completed(pending):
                result=future.result();results.append(result)
                dump(BASE/'progress.json',{'completed':len(results),'total':18,'results':results})
                print(f"完成{len(results)}/18: {result['case']} K={result['k']} seed={result['seed']} {result['wall_seconds']:.2f}s",flush=True)
    except BaseException:
        dump(BASE/'failure.json',{'error':traceback.format_exc(),'completed':results})
        raise
    finally:
        dump(BASE/'resource_usage.json',resources.finish())
    dump(BASE/'timing.json',{'started_utc':utc,'ended_utc':datetime.now(timezone.utc).isoformat(),
         'wall_seconds':time.perf_counter()-started,'workers':4,'cpu_threads_per_worker':4,
         'runs':results,'test_evaluated':False})

if __name__=='__main__':
    main()
