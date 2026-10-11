"""预先指定的迁移增益统计：10配对种子、精确符号翻转、6项Holm校正。

分析单位是固定Test上的配对训练/抽样种子，不把465条波形当465个独立重复。
不重新挑模型、不挑K或种子，不根据结果增加重复次数或搜索超参数。
"""
from itertools import product
from pathlib import Path
import csv, math, statistics
import numpy as np
import experiment as e
from experiment import BASE, SEEDS, SHOTS, CONDITIONS, NAMES, read, dump, file_sha


def exact_paired_test(deltas):
    """穷举2^10个符号分配，检验双侧平均配对差，零差值也保留。

    这是依赖配对差在零假设下可交换/对称的检验，不要求正态分布；10种子
    仍不能保证检验功效。容差只吸收等值排列的浮点求和误差。
    """
    d=np.asarray(deltas,dtype=np.float64)
    signs=np.asarray(list(product((-1,1),repeat=len(d))),dtype=np.float64)
    statistics_null=np.abs((signs*d).mean(1)); observed=abs(float(d.mean()))
    tolerance=8*np.finfo(np.float64).eps*(float(np.abs(d).sum())+observed+1)
    count=int((statistics_null>=observed-tolerance).sum())
    return count/len(signs),count,len(signs)


def holm(pvalues):
    """全6项为同一检验族，按排序后的剩余检验数乘p并累计最大值。"""
    order=sorted(range(len(pvalues)),key=lambda i:pvalues[i]);result=[None]*len(order);previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,(len(order)-rank)*pvalues[index]);result[index]=min(1.,previous)
    return result


def main():
    """全部Test核验通过后，汇总三条件成绩、迁移效应和预指定主比较。"""
    assert read(BASE/'TEST_VERIFICATION.json')['status']=='passed'
    protocol=read(BASE/'PROTOCOL.json');assert protocol['statistics']['family_size']==6
    rows=read(BASE/'TEST_EVALUATION.json')['results']
    assert len(rows)==90
    values={(r['k'],r['seed'],r['condition']):r['metrics']['macro_f1']*100 for r in rows}
    assert len(values)==90
    aggregates=[];contrasts=[];fine_vs_frozen=[]
    rng=np.random.default_rng(20261011);indices=rng.integers(0,10,size=(20000,10));bootstrap={}
    for k in SHOTS:
        for condition in CONDITIONS:
            vals=np.array([values[(k,s,condition)] for s in SEEDS])
            aggregates.append({'k':k,'condition':condition,'test_macro_f1_mean_percent':float(vals.mean()),
                'test_macro_f1_sample_sd_percent':float(vals.std(ddof=1)),'n':10})
        for condition in ('frozen_transfer','finetuning_transfer'):
            d=np.array([values[(k,s,condition)]-values[(k,s,'target_only')] for s in SEEDS])
            raw,count,total=exact_paired_test(d)
            boot=d[indices].mean(1);ci=np.quantile(boot,[.025,.975]);name=f'k{k}_{condition}_minus_target_only'
            bootstrap[name]=boot
            contrasts.append({'k':k,'comparison':condition+'_minus_target_only','condition':condition,
                'paired_delta_mean_pp':float(d.mean()),'paired_delta_sample_sd_pp':float(d.std(ddof=1)),
                'paired_bootstrap_ci95_pp':ci.tolist(),'wins':int((d>0).sum()),'ties':int((d==0).sum()),'n':10,
                'p_raw_two_sided':raw,'null_tail_count':count,'null_total':total,
                'per_seed_delta_pp':dict(zip(map(str,SEEDS),d.tolist()))})
        d=np.array([values[(k,s,'finetuning_transfer')]-values[(k,s,'frozen_transfer')] for s in SEEDS])
        fine_vs_frozen.append({'k':k,'paired_delta_mean_pp':float(d.mean()),'paired_delta_sample_sd_pp':float(d.std(ddof=1)),
            'wins':int((d>0).sum()),'descriptive_only_no_additional_significance_test':True})
    corrected=holm([r['p_raw_two_sided'] for r in contrasts])
    for r,p in zip(contrasts,corrected):
        r['p_holm_two_sided']=p
        r['positive_gain_with_holm_p_below_005']=bool(r['paired_delta_mean_pp']>0 and p<.05)
    # 用Python标准库独立复算均值/标准差和逐个符号排列，避免统计结果只依赖同一矩阵公式。
    for r in contrasts:
        ds=list(r['per_seed_delta_pp'].values());obs=abs(statistics.mean(ds));tol=8*np.finfo(np.float64).eps*(sum(map(abs,ds))+obs+1)
        tail=sum(abs(math.fsum(a*b for a,b in zip(ds,sign))/10)>=obs-tol for sign in product((-1,1),repeat=10))
        assert tail==r['null_tail_count'] and math.isclose(tail/1024,r['p_raw_two_sided'],abs_tol=1e-15)
        assert math.isclose(statistics.mean(ds),r['paired_delta_mean_pp'],rel_tol=0,abs_tol=1e-12)
        assert math.isclose(statistics.stdev(ds),r['paired_delta_sample_sd_pp'],rel_tol=0,abs_tol=1e-12)
    np.savez_compressed(BASE/'bootstrap_means.npz',**bootstrap)
    primary=next(r for r in contrasts if r['k']==20 and r['condition']=='finetuning_transfer')
    summary={'models':90,'seeds':SEEDS,'std_definition':'sample SD, ddof=1','test_evaluated':True,
        'aggregates':aggregates,'contrasts':contrasts,'fine_vs_frozen_descriptive':fine_vs_frozen,
        'primary_comparison':primary,'statistical_plan':protocol['statistics']}
    dump(BASE/'SUMMARY.json',summary)
    dump(BASE/'STATISTICAL_VERIFICATION.json',{'status':'passed','contrasts':6,'paired_runs_per_contrast':10,
        'independent_python_mean_sd_and_exact_permutation_verified':True,'family_size':6,'holm_fwer_alpha':.05,
        'normality_assumption_not_required':True,'null_assumption':'paired-label exchangeability / symmetric paired differences',
        'bootstrap_resamples':20000,'bootstrap_seed':20261011,'test_rows_not_used_as_independent_runs':True,
        'inference_scope':'sampling/training variability conditional on this fixed previously used Test',
        'power_not_established':True})
    scan=[('Simpson','separate K results; no pooled K significance'),
        ('Ecological','does not extrapolate seed-level mean to all physical circuit boards'),
        ('Berkson','fixed dataset sampling; external population representativeness unestablished'),
        ('Collider','no outcome-dependent inclusion or conditioning; all prespecified runs retained'),
        ('Base rate','Macro-F1 primary; full per-class confusion matrices retained; label0 count15 vs others45'),
        ('Regression to mean','no source/run selection by target Test; paired baseline rather than before/after score'),
        ('Survivorship','all 10 source and 90 target completed/verified; no seed dropped'),
        ('Look elsewhere','all six prespecified contrasts reported and Holm corrected'),
        ('Forking paths','protocol saved before training; historical Test exploration remains limitation'),
        ('Causal scope','controlled algorithmic comparison of extra SKL pretraining; includes extra source labels/compute, not physical causal diagnosis'),
        ('Reverse direction','source pretraining precedes target training; no reverse temporal interpretation')]
    dump(BASE/'STATISTICAL_FALLACY_SCAN.json',{'coverage':'11/11','scope':'first-round algorithmic comparison',
        'items':[{'type':n,'assessment':v} for n,v in scan],
        'cautions':['10 seeds and one reused Test do not establish general cross-board/cross-batch significance',
                    'Holm p-values are conditional on sign-flip exchangeability; no distribution-free mean-effect guarantee',
                    'Target labels matched; source labels and pretraining compute are extra resources']})
    with (BASE/'PER_SEED_RESULTS.csv').open('w',encoding='utf-8',newline='') as stream:
        w=csv.writer(stream);w.writerow(['k','seed','condition','best_epoch','test_macro_f1','test_accuracy','test_balanced_accuracy','val_macro_f1','checkpoint_sha256'])
        for r in sorted(rows,key=lambda r:(r['k'],r['seed'],r['condition'])):
            w.writerow([r[n] for n in ('k','seed','condition','best_epoch')]+[r['metrics'][n] for n in ('macro_f1','accuracy','balanced_accuracy')]+[r['val']['macro_f1'],r['checkpoint_sha256']])
    lines=['# ResNet1D：SKL预训练是否帮助ABL少样本诊断？','',
        '## Material Passport','', '- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run + validate','- Origin Date: 2026-10-11',
        '- Verification Status: VERIFIED（权重/真实采样/冻结BN/Val/Test重载）；统计为固定Test上的条件推断',
        '- Version Label: resnet_transfer_v1_server_a4d7a00','',
        '固定原ResNet1D，只比较Target-only、Frozen Transfer、Fine-tuning Transfer。提前固定10个配对种子42–51、K=10/20/40、同一ABL Train/Val/Test，不以90%准确率为目标。','',
        '## 最终ABL Test Macro-F1','', '单位：%，10种子的均值 ± 样本标准差。','',
        '| K | Target-only | Frozen Transfer | Fine-tuning Transfer |','|---:|---:|---:|---:|']
    for k in SHOTS:
        cells=[]
        for c in CONDITIONS:
            a=next(a for a in aggregates if a['k']==k and a['condition']==c)
            cells.append(f"{a['test_macro_f1_mean_percent']:.2f} ± {a['test_macro_f1_sample_sd_percent']:.2f}")
        lines.append(f'| {k} | '+' | '.join(cells)+' |')
    lines+=['','## 相比Target-only的迁移增益','',
        '单位：Macro-F1百分点。95% CI来自10个配对种子的20000次bootstrap。p为双侧精确配对符号翻转检验，穷举1024种符号配置，六项比较共同Holm校正。','',
        '| K | 条件 | 配对均值增益 | 95% CI | 胜/10 | 原始p | Holm p |','|---:|---|---:|---|---:|---:|---:|']
    for r in contrasts:
        lo,hi=r['paired_bootstrap_ci95_pp']
        lines.append(f"| {r['k']} | {NAMES[r['condition']]} | {r['paired_delta_mean_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {r['wins']} | {r['p_raw_two_sided']:.6f} | {r['p_holm_two_sided']:.6f} |")
    p=primary
    conclusion=('观察到正迁移，且在预定六项检验族中Holm校正后p<0.05。' if p['positive_gain_with_holm_p_below_005'] else
                '本轮没有得到“预训练显著提高该设置性能”的统计证据。')
    lines+=['','## 预指定主要问题','',
        f"主比较提前固定为K=20的Fine-tuning Transfer相对Target-only：平均差{p['paired_delta_mean_pp']:+.2f}个百分点，Holm p={p['p_holm_two_sided']:.6f}。{conclusion}",
        '所有K和条件均报告；没有按Test挑最佳K或方法，也没有因为结果不足而追加种子/调整超参数。平均为负的条件保留为负迁移结果。','',
        '## 数据与训练协议','',
        '- 当前P1首出现float32去重池，split_seed=20261009。SKL源Train全部2660条用于训练采样，固定65条Val选源权重；SKL Test不进入网络推理或选模。',
        '- ABL每类K=10/20/40，共110/220/440条Train；同种子嵌套，同三条件样本完全相同。固定5 Val/类，共55条，完整Test465条（标签0有15条，其余10类各45条）。',
        '- 每次运行目标标注预算为11*K+55，分别165/275/495条；源预训练额外使用SKL标签与计算，不能声称总数据/总计算相同。',
        '- 源归一化仅拟合全部SKL Train；目标归一化仅拟合该组ABL K-shot Train，同组共享统计量。剩余ABL Train、额外Val池和Test不参与训练或BN适配。',
        '- 10个源预训练模型是本轮新训练：SKL13类、SGD lr0.0022。源规模固定，不随目标K变化；源最佳骨干由SKL Val选出，两迁移条件使用同一份源权重。',
        '- 目标三条件均为新的11类头，seed+2000固定相同初值；不迁移SKL13类头。Target-only骨干随机；Frozen复制源骨干、只训练头并永久冻结BN；Fine-tuning复制同一骨干、全参数和训练BN适配。',
        '- 源与目标均无Stage1，分类阶段40个验证周期×10步=400次更新；每步每类3 Support+2 Query，全部5条参与CE。目标条件逐步使用完全相同的实际Episode；源Train全量可采样不等于遍历完整数据40轮。',
        '- 目标SGD lr0.0102，momentum0.9、weight_decay0、全局梯度裁剪5，fresh optimizer，无scheduler/early stopping/增强/BN重校准。正式预算提前固定，不另搜索学习率。',
        '- Val Macro-F1最高权重选模，同分最早轮次。全部10源+90目标训练核验后冻结90个目标候选，再统一Test。确定性FP32、禁TF32、eval batch64、4进程×4CPU线程。','',
        '## 解释边界','',
        '- Frozen回答源域特征在不适配时是否有用；Fine-tuning回答源域初始化经过目标适配是否有用。Fine与Frozen同时改变参数更新和BN适配，本轮不拆解BN的单独贡献。',
        '- 本次固定Test已用于历史实验，不是全新盲测；本轮方案和统计在看新结果前冻结。精确去重及跨电路无完全重复，不证明物理采集/电路板独立性。',
        '- 统计单位是配对种子，不把465条Test或30个跨K结果当作独立重复。六项检验共用Test且跨K相关，通过共同Holm处理多重比较。',
        '- 符号翻转依赖零假设下的配对可交换/对称性；bootstrap CI描述固定Test上训练/抽样波动，不是未知采集批次的总体置信区间。10种子不构成功效保证，结论只针对本轮协议。','',
        '## 核验与文件','',
        '- 10源+90目标全部检查随机/迁移初始化、同组新头、真实采样、400次更新、BN计数、冻结骨干及Val最佳选择。90份目标最佳权重均独立重载复现Test分数与预测，混淆矩阵与sklearn指标一致。',
        '- 6组配对差的均值/标准差/精确p另用Python标准库穷举复算；统计风险扫描11/11项有记录。原数据、模型源码、旧输出和环境包版本保持指纹不变。',
        '- 证据：PROTOCOL/PREFLIGHT、provenance划分、SOURCE/TARGET_TRAINING、TRAINING_VERIFICATION、CHECKPOINT_MANIFEST、TEST_EVALUATION/VERIFICATION、SUMMARY、PER_SEED_RESULTS、STATISTICAL_VERIFICATION及逐模型记录。','']
    for stage in ('source','target'):
        t=read(BASE/f'{stage.upper()}_TRAINING.json')
        lines.append(f"- {stage}训练窗口：{t['wall_seconds']:.2f}秒，{t['models']}模型，含进程启动及读取/保存，不含独立核验、测试、报告或同步。")
    lines.append('')
    (BASE/'SERVER_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[13:25]),flush=True)
    print('primary',p['paired_delta_mean_pp'],p['p_holm_two_sided'],conclusion,flush=True)


if __name__=='__main__':main()