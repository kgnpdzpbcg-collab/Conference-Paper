"""独立核验抽样来源、归一化、选模、权重及逐行预测，验证失败阻止测试。

指标由布尔计数重算，训练阶段核验 Val，所有任务通过后才核验最终 Test。
"""

import csv
from pathlib import Path
import statistics

import numpy as np
import torch
from ccsae.fixed_data import fingerprint, read_json, write_json
from ccsae.fixed_verify import verify_predictions
from fewshot_data import CASES, SEEDS, SHOTS, cache_folder
from run_fewshot import DATA, REFERENCE, CCSAE_FULL, BASELINE_FULL, check_frozen, ccsae_folder, predictions
from ccsae.fixed_data import load_fixed
from ccsae.fixed_experiment import checkpoint_model, configure, predict


def verify_caches(output: Path) -> None:
    """从原始 CSV 重算抽样与统计，并核对三个模型共享的全部 18 份缓存。"""
    for case, circuit in CASES.items():
        raw = {s: np.loadtxt(DATA / f'{circuit}_INCI_{s}_data.csv', delimiter=',', ndmin=2)
               for s in ('train', 'val', 'test')}
        labels = {s: np.loadtxt(DATA / f'{circuit}_INCI_{s}_label.csv', delimiter=',', ndmin=2).argmax(1)
                  for s in raw}
        classes = len(np.unique(labels['train']))
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            per_class = [rng.permutation(np.flatnonzero(labels['train'] == c)) for c in range(classes)]
            previous = set()
            for k in SHOTS:
                folder = cache_folder(output, case, seed, k)
                meta = read_json(folder / 'metadata.json')
                rows = np.asarray(meta['selected_train_indices'], dtype=np.int64)
                expected = np.sort(np.concatenate([indices[:k] for indices in per_class]))
                assert np.array_equal(rows, expected)
                assert len(set(rows)) == classes*k and previous.issubset(set(rows))
                assert rows.min() >= 0 and rows.max() < len(raw['train'])
                assert np.bincount(labels['train'][rows], minlength=classes).tolist() == [k]*classes
                previous = set(rows)
                stats = meta['circuits'][circuit]['normalization']
                assert stats['fitted_split'] == 'selected_train_only'
                assert stats['mean'] == float(raw['train'][rows].mean())
                assert stats['std'] == float(raw['train'][rows].std())
                assert stats['sample_count'] == classes*k
                for split in raw:
                    x = raw[split][rows] if split == 'train' else raw[split]
                    y = labels[split][rows] if split == 'train' else labels[split]
                    values = ((x-stats['mean'])/stats['std']).astype(np.float32)
                    with np.load(folder / f'{circuit}_{split}.npz', allow_pickle=False) as saved:
                        assert np.array_equal(values, saved['x'])
                        assert np.array_equal(y, saved['y'])
                        assert np.array_equal(rows if split == 'train' else np.arange(len(y)), saved['original_row_index'])


def history_and_checkpoint(output: Path, plan: dict, job: dict) -> tuple[str, int]:
    """核对最优轮次与训练预算；基线额外核验 BN 只遍历了所选训练集。"""
    folder = output / job['id']
    result = read_json(folder / 'result.json')
    if job['model'] == 'ccsae_v1':
        weights = ccsae_folder(output, job)
        history = read_json(weights / 'history.json')
        first = [r for r in history if r['stage'] == 1]
        second = [r for r in history if r['stage'] == 2]
        config = plan['training']['ccsae']
        assert len(first) == config['feature_epochs'] and len(second) == config['classification_epochs']
        for records in (first, second):
            assert [r['epoch'] for r in records] == list(range(1, len(records)+1))
            assert all(np.isfinite(v) for r in records for v in r.values())
        stage1 = torch.load(weights/'stage1_best.pt', map_location='cpu', weights_only=True)
        stage2 = torch.load(weights/'stage2_best.pt', map_location='cpu', weights_only=True)
        assert stage1['epoch'] == min(first, key=lambda r:r['loss'])['epoch']
        assert stage2['epoch'] == max(second, key=lambda r:r['macro_f1'])['epoch'] == result['best_epoch']
        for name, value in stage1['state_dict'].items():
            if name.startswith('decoder.'):
                assert torch.equal(value, stage2['state_dict'][name])
        assert fingerprint(weights/'stage1_best.pt') == result['stage1_sha256']
        coefficient = config['scl_weights'][job['case']]
        for row in first:
            for total, recon, scl in (('train_loss', 'train_reconstruction_loss', 'train_scl_loss'),
                                      ('loss', 'reconstruction_loss', 'scl_loss')):
                assert np.isclose(row[total], row[recon]+coefficient*row[scl], rtol=2e-6, atol=1e-7)
        return fingerprint(weights/'stage2_best.pt'), stage2['epoch']
    with (folder/'history.csv').open(encoding='utf-8', newline='') as handle:
        rows = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(handle)]
    config = plan['training']['baseline']
    assert [r['epoch'] for r in rows] == list(range(1,len(rows)+1))
    assert all(np.isfinite(v) for row in rows for v in row.values())
    assert len(rows) == result['epochs_run'] <= config['max_epochs']
    best = max(rows,key=lambda r:r['val_macro_f1'])
    assert int(best['epoch']) == result['best_epoch']
    assert abs(best['val_macro_f1']-result['validation']['macro_f1']) < 1e-12
    if len(rows) < config['max_epochs']:
        assert len(rows)-result['best_epoch'] == config['patience']
    checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
    expected_batches = int(np.ceil((13 if job['case']=='skl' else 11)*job['k']/config['batch_size']))
    for name, value in checkpoint['state'].items():
        if name.endswith('num_batches_tracked'):
            assert int(value) == expected_batches, (job['id'],name,int(value),expected_batches)
    assert checkpoint['best_epoch'] == result['best_epoch']
    return fingerprint(folder/'best.pt'), result['best_epoch']


def verify_training(output: Path, plan: dict) -> dict:
    """训练全部完成后独立重算 Val，并冻结全部最优权重指纹供测试阶段检查。"""
    assert read_json(output/'training_complete.json')['runs'] == len(plan['runs'])
    check_frozen(output, plan)
    verify_caches(output)
    checked, weights = [], {}
    for job in plan['runs']:
        folder = output/job['id']
        result = read_json(folder/'result.json')
        assert result['status']=='trained' and not result['test_evaluated']
        digest, epoch = history_and_checkpoint(output,plan,job)
        assert digest == result['checkpoint_sha256']
        weights[job['id']] = digest
        prepared = cache_folder(output,job['case'],job['seed'],job['k'])
        with np.load(prepared/f"{job['circuit']}_val.npz",allow_pickle=False) as data:
            labels = data['y'].copy()
        classes = 13 if job['case']=='skl' else 11
        verify_predictions(folder/'val_predictions.npz',labels,result['validation'],classes)
        metric, arrays = predictions(output,job,'val')
        with np.load(folder/'val_predictions.npz',allow_pickle=False) as old:
            assert np.array_equal(old['probability'],arrays['probability'])
        assert abs(metric['loss_unweighted']-result['validation']['loss_unweighted']) < 1e-12
        checked.append({'id':job['id'],'best_epoch':epoch,'reload_validation_predictions_exact':True})
    assert not list(output.rglob('test_predictions.npz'))
    report={'status':'passed','runs':checked,'checkpoint_sha256':weights,
            'sampling_nested_and_balanced':True,'normalization_selected_train_only':True,
            'val_test_original_rows_unchanged':True,'no_new_test_predictions':True}
    write_json(output/'training_validation.json',report)
    print(f"training verified: {len(checked)} jobs",flush=True)
    return report


def verify_final(output: Path, plan: dict) -> dict:
    """独立重算 54 份少样本 Test；新增完整训练 V1 的六份 Test 也重载核验。"""
    check_frozen(output,plan)
    gate=read_json(output/'training_validation.json')
    assert gate['status']=='passed'
    checked=[]
    for job in plan['runs']:
        folder=output/job['id']
        result=read_json(folder/'result.json')
        assert result['status']=='complete' and result['test_evaluated']
        assert result['checkpoint_sha256']==gate['checkpoint_sha256'][job['id']]
        classes=13 if job['case']=='skl' else 11
        prepared=cache_folder(output,job['case'],job['seed'],job['k'])
        with np.load(prepared/f"{job['circuit']}_test.npz",allow_pickle=False) as data:
            labels=data['y'].copy()
        verify_predictions(folder/'test_predictions.npz',labels,result['test'],classes)
        metric,arrays=predictions(output,job,'test')
        with np.load(folder/'test_predictions.npz',allow_pickle=False) as old:
            assert np.array_equal(old['probability'],arrays['probability'])
        assert abs(metric['loss_unweighted']-result['test']['loss_unweighted']) < 1e-12
        current,_=history_and_checkpoint(output,plan,job)
        assert current==gate['checkpoint_sha256'][job['id']]
        checked.append(job['id'])
    full=read_json(output/'full_train_results.json')
    assert len(full)==18
    for run in full:
        if run['model']!='ccsae_v1':
            historical=next(r for r in read_json(BASELINE_FULL/'summary.json')['runs'] if r['id']==run['id'])
            assert run['validation']==historical['validation'] and run['test']==historical['test']
            continue
        configure(run['seed'],4,'cuda')
        splits=load_fixed(DATA,REFERENCE,run['case'])
        assert fingerprint(Path(run['checkpoint_path']))==run['checkpoint_sha256']
        model,_=checkpoint_model(Path(run['checkpoint_path']),'cuda')
        metrics,arrays=predict(model,splits.test,64)
        filename=output/'full_train_ccsae'/f"{run['id']}_test_predictions.npz"
        verify_predictions(filename,splits.test.tensors[1].numpy(),run['test'],model.cfg.classes)
        with np.load(filename,allow_pickle=False) as old:
            assert np.array_equal(old['probability'],arrays['probability'])
        assert abs(metrics['loss_unweighted']-run['test']['loss_unweighted']) < 1e-12
    report={'status':'passed','small_test_runs':len(checked),'full_ccsae_test_runs':6,
            'historical_full_baseline_runs_reused':12,'reload_test_predictions_exact':True,
            'independent_metrics':True,'source_data_and_checkpoint_hashes_match':True}
    write_json(output/'validation.json',report)
    print('final verified: 54 few-shot + 6 existing V1 checkpoints',flush=True)
    return report
