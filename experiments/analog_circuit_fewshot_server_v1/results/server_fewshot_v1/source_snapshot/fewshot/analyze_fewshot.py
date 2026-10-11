"""汇总服务器少样本测试结果，按预先固定的种子计算均值及样本标准差。"""

import csv
from pathlib import Path
import statistics

import numpy as np
from ccsae.fixed_data import fingerprint, read_json, write_json
from fewshot_data import CASES, SHOTS
from run_fewshot import MODELS


def analyze(output: Path, plan: dict) -> dict:
    """只汇总已通过最终核验的结果；完整训练对照明确注明复用及新测试来源。"""
    gate=read_json(output/'validation.json')
    assert gate['status']=='passed' and not plan['smoke_only']
    runs=[read_json(output/job['id']/'result.json') for job in plan['runs']]
    runs+=read_json(output/'full_train_results.json')
    assert len(runs)==72 and all(r['test_evaluated'] for r in runs)
    groups=[]
    for case in CASES:
        for model in MODELS:
            for k in (*SHOTS,'full'):
                selected=[r for r in runs if r['case']==case and r['model']==model and r['k']==k]
                assert sorted(r['seed'] for r in selected)==[42,43,44]
                classes=13 if case=='skl' else 11
                row={'case':case,'model':model,'k':k,'train_samples':classes*k if k!='full' else (2660 if case=='skl' else 2170),
                     'val_samples':570 if case=='skl' else 465,'test_samples':570 if case=='skl' else 465,'runs':3}
                for split in ('validation','test'):
                    for metric in ('accuracy','macro_f1','balanced_accuracy'):
                        values=[r[split][metric] for r in selected]
                        mean,std=float(np.mean(values)),float(np.std(values,ddof=1))
                        assert abs(mean-statistics.mean(values))<1e-12
                        assert abs(std-statistics.stdev(values))<1e-12
                        row[f'{split}_{metric}_mean']=mean
                        row[f'{split}_{metric}_std']=std
                groups.append(row)
    summary={'status':'complete','groups':groups,'runs':runs,
             'std_definition':'sample std across 3 paired sampling+training seeds; full-Train has training seed variation only',
             'fewshot_training_runs':54,'new_test_inferences':60,'historical_full_baseline_runs':12}
    write_json(output/'summary.json',summary)
    with (output/'summary.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
    fields=['id','case','model','k','seed','validation_accuracy','validation_macro_f1','test_accuracy','test_macro_f1','test_balanced_accuracy']
    with (output/'per_seed.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for run in runs:
            row={name:run[name] for name in ('id','case','model','k','seed')}
            for split,metric in (('validation','accuracy'),('validation','macro_f1'),('test','accuracy'),('test','macro_f1'),('test','balanced_accuracy')):
                row[f'{split}_{metric}']=run[split][metric]
            writer.writerow(row)
    gate.update(independent_group_aggregates=True,aggregate_groups=24)
    write_json(output/'validation.json',gate)
    names={'cnn1d':'1D-CNN','resnet1d':'ResNet1D','ccsae_v1':'CCSAE v1'}
    lines=['# 模拟电路少样本实验（服务器 GPU）','',
           '## Material Passport','',
           '- Origin Skill: academic-research-suite / experiment-agent',
           '- Origin Mode: run',
           '- Origin Date: 2026-10-11',
           '- Verification Status: VERIFIED（数据审计、选模及推理重载；不代表完整训练逐位重跑）',
           '- Version Label: fewshot_server_v1','',
           '本轮完成 54 次少样本训练，使用固定 Train/Val/Test 划分及三个抽样/训练种子。以下均为服务器测试集成绩，单位为 %，数值为均值 ± 样本标准差。','',
           '| 电路 | 模型 | 每类 K | 训练总数 | Test Accuracy | Test Macro-F1 | Test Balanced Accuracy |',
           '|---|---|---:|---:|---:|---:|---:|']
    for group in groups:
        scores=[f"{100*group[f'test_{m}_mean']:.2f} ± {100*group[f'test_{m}_std']:.2f}" for m in ('accuracy','macro_f1','balanced_accuracy')]
        lines.append(f"| {group['case'].upper()} | {names[group['model']]} | {group['k']} | {group['train_samples']} | {' | '.join(scores)} |")
    lines+=['','## 固定协议','',
            '- 从去重后既有 Train 按类别抽取 K=10/20/40；SKL 共 130/260/520 条，ABL 共 110/220/440 条。原划分种子 20261009，不重新分配任何 Val/Test 行。',
            '- 抽样与训练种子为 42、43、44。每个种子使用类内随机排列的嵌套前缀，因此 K10⊂K20⊂K40；三个模型使用逐元素相同的训练、验证和测试输入。',
            '- 每个 K/种子单独用抽中的 Train 拟合全局标量均值/标准差。未抽中的训练样本不用于预训练、分类训练、归一化或 BN 重估。',
            '- SKL Val/Test 各 570，ABL 各 465；两个评估集合的原始波形、标签和行顺序与历史实验一致。每个少样本模型使用自己训练子集拟合的统计量。',
            '- CCSAE v1：仓库提交 d4413979d27160c6a96690f9b544738c6ca92bd7，signed_real 重构；batch64、SGD momentum0.9、100+200轮；SKL lr0.0022/λ0.0017，ABL lr0.0102/λ0.0059；温度0.1、梯度裁剪5。特征阶段按验证总损失最小选权重，分类阶段按验证 Macro-F1 最大选权重。',
            '- CNN/ResNet1D：复用服务器原训练代码；batch128、AdamW lr0.001/weight_decay0.0001，最多100轮、验证 Macro-F1 连续15轮不升则早停，Plateau调度factor0.5/patience5。每轮只用所选 Train 重估 BN。逆频数权重由所选 Train 计算，本轮每类 K 条所以均为1。',
            '- 无数据增强、无额外预训练、所有任务从随机初始化训练。固定 epoch 预算下，样本更少也意味着每轮优化器更新次数更少，未为了补偿样本量额外延长训练。',
            '- 全部54次验证选模通过并冻结权重后才统一 Test；不使用 Test 挑选 K、种子、训练轮次或超参数。','',
            '## 完整训练集对照','',
            '- CNN/ResNet1D 的 full 行复用已核验的服务器完整 Train 测试结果，共12次；没有重新训练。',
            '- CCSAE v1 的 full 行使用既有6份完整 Train 最优权重，本轮新增固定 Test 推理及核验，没有重新训练或修改既有权重。',
            '- full 的归一化拟合全部原 Train，类别分布仍有 C0 样本较少的差异；K-shot 数据是按每类 K 条构造的平衡子集。样本量与类别组成均不同，解释差距时需同时考虑。','',
            '## 核验与解释边界','',
            '- 18份子集逐行追溯原 Train，核对每类 K 条、嵌套关系、抽样可复现以及所选 Train 的归一化统计。Val/Test 原始行保持一致。',
            '- 54份权重按对应验证规则选模，独立重算指标；重载后 Val/Test 类别概率逐元素完全一致。CCSAE 解码器在分类阶段保持不变；基线 BN 遍历次数与少样本训练批次数一致。',
            '- 24组汇总的均值和样本标准差由独立加和方式交叉核对。没有计算或声称统计显著性。',
            '- K-shot 标准差同时包含训练子集抽样和模型初始化变化；full 仅包含训练随机性，不能把两者都解释为单纯初始化波动。',
            '- 这是固定验证/测试集上的监督 K-shot 分类实验；验证样本量未限制为 K。精确去重仍不能证明跨器件或独立采集批次泛化。','',
            '## 环境、耗时与文件','',
            f"- GPU：{plan['environment']['gpu']}；PyTorch {plan['environment']['torch']}；CUDA {plan['environment']['cuda_runtime']}。复用既有锁定 uv 环境，确定性 FP32，AMP/TF32关闭，单GPU顺序执行。",
            f"- 54次训练函数累计耗时：{sum(r['train_seconds'] for r in runs if r['k']!='full'):.2f}秒；该值含逐轮验证和权重保存，不含准备、独立核验及测试阶段。",
            '- plan.json / source_snapshot：实验协议及原样源码；prepared：18份共享缓存、统计量和训练来源清单。',
            '- 各任务目录：最优权重、逐轮历史、Val/Test 预测及 result.json。full_train_ccsae：既有完整训练权重的新增 Test 预测。',
            '- summary.json / summary.csv：24组结果；per_seed.csv：72行逐种子结果；training_validation.json / validation.json：独立核验。','',
            f"- plan.json SHA-256：`{fingerprint(output/'plan.json')}`。",
            f"- summary.json SHA-256：`{fingerprint(output/'summary.json')}`。",
            f"- validation.json SHA-256：`{fingerprint(output/'validation.json')}`。"]
    (output/'FEWSHOT_SERVER_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('report generated: 24 groups, 72 per-seed rows',flush=True)
    return summary
