"""汇总同数据/同分类协议的ResNet1D，明确保留无Stage1这一方法差异。

只读取已核验结果并计算三种子统计、配对差值；不新增训练或Test评估。
旧实验、模型和环境文件在报告生成前再次逐项核对原字节。
"""
import csv
import importlib.metadata
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'server_experiment'))
from common import dump, file_sha
from audit import BASE, WORK, P1, COMMIT, MODEL_SHA, read, reference_job


def stats(values):
    """n=3均值和样本标准差；配对差值先逐种子计算，不混合独立方差。"""
    assert len(values)==3
    return {'mean':statistics.mean(values),'sample_std':statistics.stdev(values),'n':3}


def main():
    """完整核验与旧文件保持通过后，生成独立ResNet报告。"""
    verification=read(BASE/'VERIFICATION.json')
    assert verification['status']=='passed' and verification['runs']==18
    assert verification['same_seed_initial_state_equal_across_k'] is True
    expected={(c,k,s) for c in ('skl','abl') for k in (10,20,40) for s in (42,43,44)}
    assert {(r['case'],r['k'],r['seed']) for r in verification['results']}==expected
    rows=[]
    for case in ('skl','abl'):
        for k in (10,20,40):
            for seed in (42,43,44):
                job=BASE/'runs'/case/f'k{k}_sample{seed}_train{seed}'
                result=read(job/'summary.json')
                ref=reference_job(case,k,seed)
                previous=read(ref/'summary.json')
                a=read(job/'split_audit.json')
                assert a==read(ref/'split_audit.json')
                assert result['test_evaluated'] is False and result['stage1_epochs']==0
                row={'case':case,'k':k,'seed':seed,'train_size':a['selected_train_size'],
                     'val_size':a['selected_val_size'],'resnet_best_epoch':result['best_epoch'],
                     'resnet_val_macro_f1_percent':result['val']['macro_f1']*100,
                     'resnet_val_accuracy_percent':result['val']['accuracy']*100,
                     'resnet_val_balanced_accuracy_percent':result['val']['balanced_accuracy']*100,
                     'ccsae_fc_val_macro_f1_percent':previous['models']['fc']['val']['macro_f1']*100,
                     'ccsae_proto_val_macro_f1_percent':previous['models']['proto']['val']['macro_f1']*100,
                     'resnet_checkpoint_sha256':result['checkpoint_sha256'],
                     'resnet_relative_to_work':job.relative_to(WORK).as_posix(),
                     'ccsae_reference_relative_to_work':ref.relative_to(WORK).as_posix()}
                row['resnet_minus_ccsae_fc_pp']=row['resnet_val_macro_f1_percent']-row['ccsae_fc_val_macro_f1_percent']
                row['resnet_minus_ccsae_proto_pp']=row['resnet_val_macro_f1_percent']-row['ccsae_proto_val_macro_f1_percent']
                rows.append(row)
    aggregate={}
    for case in ('skl','abl'):
        aggregate[case]={}
        for k in (10,20,40):
            group=[r for r in rows if r['case']==case and r['k']==k]
            a={key:stats([r[key] for r in group]) for key in ('resnet_val_macro_f1_percent',
                'resnet_val_accuracy_percent','ccsae_fc_val_macro_f1_percent','ccsae_proto_val_macro_f1_percent',
                'resnet_minus_ccsae_fc_pp','resnet_minus_ccsae_proto_pp')}
            a['resnet_wins_vs_fc']=sum(r['resnet_minus_ccsae_fc_pp']>0 for r in group)
            a['resnet_wins_vs_proto']=sum(r['resnet_minus_ccsae_proto_pp']>0 for r in group)
            aggregate[case][str(k)]=a
    before=read(BASE/'provenance/input_before.json')
    assert before=={n:file_sha(WORK/n) for n in before}, '旧实验/数据/环境文件改变'
    runtime=read(BASE/'provenance/runtime_before.json')
    assert runtime['packages']=={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    sources=read(P1/'SOURCE_MANIFEST.json')['tracked_sources']
    assert sources=={n:file_sha(ROOT/n) for n in sources}
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==COMMIT
    assert file_sha(Path(__file__).parent/'reference_models.py')==MODEL_SHA
    preserved={'status':'passed','protected_file_count':len(before),'input_before_sha256':file_sha(BASE/'provenance/input_before.json'),
         'old_ccsae_p0_p1_and_historical_baselines_unchanged':True,'data_and_existing_uv_configuration_unchanged':True,
         'installed_package_versions_unchanged':True,'ccsae_model_and_training_source_unchanged':True,
         'original_resnet_architecture_unchanged':True}
    helpers=sorted(Path(__file__).parent.glob('*.py'))+[ROOT/'server_experiment'/n for n in ('common.py','run_shot_scale.py','verify_shot_scale.py')]
    dump(BASE/'SOURCE_MANIFEST.json',{'ccsae_source_commit':COMMIT,'tracked_ccsae_sources':sources,
         'original_resnet_sha256':MODEL_SHA,'runtime_helpers':{p.relative_to(ROOT).as_posix():file_sha(p) for p in helpers}})
    dump(BASE/'COMPLETION_VERIFICATION.json',{'status':'passed','runs':18,'same_train_val_and_scaler_exact':True,
         'same_actual_episode_indices_to_ccsae_fc':True,'initialization_random_and_same_across_k':True,
         'all_400_updates_bn_and_checkpoint_reloads_verified':True,'independent_val_metrics_verified':True,
         'test_evaluated':False,'stage1_epochs_resnet':0,'stage1_epochs_ccsae':10,'preservation':preserved})
    timing=read(BASE/'timing.json');usage=read(BASE/'resource_usage.json')
    output={'protocol':'Original ResNet1D, random initialization, same P1 data and classification protocol only',
         'stage1_epochs_resnet':0,'stage1_epochs_ccsae':10,'stage2_updates_all_models':400,'test_evaluated':False,
         'runs':18,'seeds':[42,43,44],'training_window_seconds':timing['wall_seconds'],
         'summary_by_case_and_k':aggregate,'per_seed':rows,'preservation':preserved,
         'limitations':['相同的是数据及分类训练协议；CCSAE有Stage1，ResNet没有，不能称完整两阶段训练相同',
                        'Val同时用于最佳epoch选择和汇总，非独立Test成绩','n=3，抽样与训练种子耦合，无显著性结论',
                        '原始记录缺少独立板卡/采集批次元数据，去重不保证来源独立']}
    dump(BASE/'SUMMARY.json',output)
    with (BASE/'PER_SEED_RESULTS.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def fmt(a,key):
        return f"{a[key]['mean']:.2f} ± {a[key]['sample_std']:.2f}"
    lines=['# ResNet1D：对齐P1数据与分类训练协议的服务器结果','',
         '**按用户选定方式保留原ResNet1D，从随机初始化开始，不做Stage1。18次训练使用与本轮CCSAE-FC/P1完全一致的Train/Val、标准化与实际Episode顺序，分类阶段均为400次更新。只报告Val，不评估Test。**','',
         '## 1. 三种子结果','',
         '| 电路 | K | CCSAE-FC Val F1 (%) | CCSAE-Proto P1 Val F1 (%) | ResNet1D Val F1 (%) | ResNet−FC (pp) | ResNet胜出FC/Proto |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for case in ('skl','abl'):
        for k in (10,20,40):
            a=aggregate[case][str(k)]
            lines.append(f"| {case.upper()} | {k} | {fmt(a,'ccsae_fc_val_macro_f1_percent')} | {fmt(a,'ccsae_proto_val_macro_f1_percent')} | {fmt(a,'resnet_val_macro_f1_percent')} | {fmt(a,'resnet_minus_ccsae_fc_pp')} | {a['resnet_wins_vs_fc']}/3 / {a['resnet_wins_vs_proto']}/3 |")
    lines+=['','均值 ± 样本标准差（ddof=1，n=3）；差值按相同case/K/seed配对，单位为百分点。FC/Proto成绩只读复用既有服务器结果，没有重新训练。']
    for case in ('skl','abl'):
        means=[aggregate[case][str(k)]['resnet_val_macro_f1_percent']['mean'] for k in (10,20,40)]
        lines+=['',f"{case.upper()} ResNet的平均Val Macro-F1随K=10/20/40为 {means[0]:.2f}% → {means[1]:.2f}% → {means[2]:.2f}%。保留全部三种子波动，不据结果换种子、调学习率或延长训练。"]
    lines+=['','## 2. 数据、模型与明确差异','',
         '逐条对齐的是冻结P1的数据：SKL13类、ABL11类；K=10/20/40，每类Val5条；split_seed/val_seed=20261009，sample_seed/training_seed=42/43/44。同种子Train嵌套，Val原始记录与顺序固定，归一化仅拟合各K选中的Train。SKL Train总数130/260/520，Val65；ABL Train110/220/440，Val55。18个split_audit与对应CCSAE存档完整相同，归一化张量已回查原始float32波形独立复算。','',
         f"原ResNet源码SHA256：`{MODEL_SHA}`，与原服务器基线及历史少样本快照完全一致。采用ResNet-18式一维基本残差块，四阶段各两块，通道32/64/128/256，全局均值池化后线性分类。只将[N,1000]输入增加为[N,1,1000]，未改卷积、宽度、分类头或新增解码器。",'',
         '**方法差异：ResNet没有解码器，本轮不加解码器、不执行重构/SupCon、不载入任何CCSAE或旧ResNet预训练权重。CCSAE-FC/Proto沿用10轮Stage1。本对照控制数据和分类阶段，不是完整两阶段训练完全相同，也不是单独隔离Backbone因素的消融。**','',
         '同case/seed的ResNet在不同K下随机初始化参数及BN缓冲区一致；ResNet与CCSAE仅使用同一随机种子，网络形状不同，不要求两者初始化权重相同。','',
         '## 3. 分类训练冻结配置','',
         '| 项目 | 本轮ResNet配置 |','|---|---|','| 初始化 / Stage1 | 随机初始化 / 0轮 |',
         '| 更新预算 | 40轮×10步=400次 |','| Episode | 每类3 Support+2 Query；所有Support在前、Query在后；全部样本参与FC交叉熵 |',
         '| SGD | SKL lr0.0022，ABL lr0.0102；momentum0.9、weight_decay0、clip5 |',
         '| 额外训练机制 | 无调度器、无早停、无增强、无额外BN重估 |',
         '| BN | 使用原ResNet默认运行统计，只由训练Episode更新；Val设eval |',
         '| Val / 选模 | batch64；每轮Val Macro-F1最高，平局最早；固定40轮，无追加 |',
         '| 数值 | 确定性FP32；禁用AMP/TF32与非math SDP |','',
         '历史ResNet基线使用AdamW、完整Val、早停和Train-only BN重估。本轮按当前CCSAE-FC分类协议改为上述固定SGD/400步；历史Test成绩不可与本表直接混合。','',
         '## 4. 逐种子结果','',
         '| 电路 | K | seed | ResNet Val F1 (%) | ResNet最佳轮 | ResNet−FC (pp) | ResNet−Proto (pp) |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case'].upper()} | {r['k']} | {r['seed']} | {r['resnet_val_macro_f1_percent']:.2f} | {r['resnet_best_epoch']} | {r['resnet_minus_ccsae_fc_pp']:+.2f} | {r['resnet_minus_ccsae_proto_pp']:+.2f} |")
    lines+=['','## 5. 执行与核验','',
         f"在{runtime['gpu']}上复用 `../analog-circuit-crossdomain/.venv/bin/python`（Python3.12.3，torch {runtime['torch']}）。四进程、每进程四CPU线程；没有创建环境、安装或下载包。18次训练批次窗口 {timing['wall_seconds']:.3f} 秒，平均GPU利用率 {usage['mean_gpu_percent']:.2f}%，峰值显存 {usage['peak_gpu_memory_mib']:.0f} MiB。窗口含进程初始化、读取和保存，不含审计、独立核验、同步及报告。",'',
         '服务器新增脚本语法检查通过，18次全部独立核验通过：初始参数由同种子重新构造一致；实际400步Support/Query与对应CCSAE-FC存档逐条相同，独立numpy重放一致，类平衡且互斥；最后权重全部BN计数为400，最佳权重BN计数为最佳轮×10；编码器及分类头均发生更新；40轮历史有限、最早最佳epoch规则正确；最佳权重重载后的Val预测逐条相同，独立混淆矩阵复算指标一致。','',
         f"全部{len(before)}个旧实验/数据/环境文件原字节保持，已安装包版本一致，CCSAE源码固定 `{COMMIT}`。新代码位于 `server_experiment/resnet_shot_scale/`，输出独立保存于 `outputs/resnet_shot_scale/server_a4d7a00/`，没有覆盖P0/P1或历史ResNet。",'',
         '## 6. 解释与复现边界','',
         '固定每类5条Val同时用于最佳epoch选择和结果汇总，不能称Test泛化结果；三种子不足以建立统计显著性，且抽样/训练种子耦合。CCSAE有Stage1而ResNet没有，因此结果代表用户选定的同数据/同分类协议对照，不能宣称只因网络结构导致差异。去重不证明独立器件或采集批次泛化，此前数据探索限制继续存在。','',
         'SUMMARY.json、PER_SEED_RESULTS.csv记录未四舍五入结果和引用路径；VERIFICATION.json、COMPLETION_VERIFICATION.json、SOURCE_MANIFEST.json记录核验及冻结证据；provenance/保存18个划分、环境、旧文件指纹；runs/保存初始化、最佳和最后权重、40轮曲线、实际Episode和Val预测。','',
         '实际服务器命令（本轮归档目标已存在，不可原地重跑覆盖；独立复现需另设空BASE）：','',
         '```bash','export PYTHONPATH="$PWD/src"',
         '../analog-circuit-crossdomain/.venv/bin/python server_experiment/resnet_shot_scale/audit.py',
         '../analog-circuit-crossdomain/.venv/bin/python server_experiment/resnet_shot_scale/run.py',
         '../analog-circuit-crossdomain/.venv/bin/python server_experiment/resnet_shot_scale/verify.py',
         '../analog-circuit-crossdomain/.venv/bin/python server_experiment/resnet_shot_scale/report.py','```','',
         '封装复用既有common.configure、资源采样和独立混淆矩阵函数，实际源文件快照一并留存。数据、环境、权重与NPZ只保留服务器及本地同步包，GitHub只上传文本证据/代码快照。没有Test推理或超参数搜索。']
    (BASE/'SERVER_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'status':'passed','summary':aggregate,'seconds':timing['wall_seconds'],'old_files_preserved':len(before)})

if __name__=='__main__':
    main()
