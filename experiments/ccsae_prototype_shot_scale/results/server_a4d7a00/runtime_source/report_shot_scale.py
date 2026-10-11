"""只汇总冻结 P1 的样本规模验证，独立统计并核查旧结果未被修改。

读取 12 组新配对和 6 组已有 K20 配对；不训练、不重新选模型、不执行 Test。
所有均值/标准差使用未四舍五入的逐种子指标，差值始终为同种子 Proto−FC。
"""
from __future__ import annotations
import csv
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
from common import dump, file_sha
from audit_shot_scale import ROOT, WORK, BASE, P1, COMMIT, old_job


def read(path):
    """读取已有 UTF-8 JSON；缺失或无效文件直接报错，不跳过失败组合。"""
    return json.loads(path.read_text())


def stats(values):
    """三种子均值与样本标准差（ddof=1）；不将 n=3 视为显著性检验。"""
    assert len(values) == 3
    return {'mean':statistics.mean(values),'sample_std':statistics.stdev(values),'n':3}


def main():
    """全组合核验通过后生成汇总及证据；不改动原 K20 归档。"""
    pre = read(BASE/'provenance/preflight.json')
    assert pre['status'] == 'passed'
    new = BASE/'new_k10_k40'
    v = read(new/'verification.json')
    assert v['status'] == 'passed' and v['paired_runs'] == 12 and v['test_evaluated'] is False
    keys = {(r['case'],r['k'],r['seed']) for r in v['results']}
    assert keys == {(c,k,s) for c in ('skl','abl') for k in (10,40) for s in (42,43,44)}
    old_verification = []
    for phase in ('seed42_precheck','seeds43_44_followup'):
        phase_v = read(P1/phase/'verification.json')
        assert phase_v['status'] == 'passed'
        old_verification.extend(phase_v['results'])
    assert {(r['case'],r['seed']) for r in old_verification} == {(c,s) for c in ('skl','abl') for s in (42,43,44)}
    dump(BASE/'REUSED_K20_VERIFICATION.json', {'status':'passed','pairs':6,
         'source_relative_to_work':P1.relative_to(WORK).as_posix(),
         'original_verification_reused_without_retraining':True,'results':old_verification})
    rows = []
    for case in ('skl','abl'):
        for k in (10,20,40):
            for seed in (42,43,44):
                job = old_job(case,seed) if k == 20 else new/case/f'k{k}_sample{seed}_train{seed}_distnone'
                s = read(job/'summary.json')
                config = read(job/'config.json')
                reference = read(old_job(case,seed)/'config.json')
                assert {n:x for n,x in config.items() if n not in ('output','k')} == {n:x for n,x in reference.items() if n not in ('output','k')}
                assert s['test_evaluated'] is False
                a = read(job/'split_audit.json')
                assert a == read(BASE/'provenance/splits'/f'{case}_k{k}_seed{seed}.json')
                r = {'case':case,'k':k,'seed':seed,'train_size':a['selected_train_size'],
                     'val_size':a['selected_val_size'],'origin':'reused_k20' if k==20 else 'new_server_run',
                     'job_relative_to_work':job.relative_to(WORK).as_posix(),
                     'stage1_updates':10*((a['selected_train_size']+63)//64),'stage2_updates_per_method':400}
                for method in ('fc','proto'):
                    for metric in ('macro_f1','accuracy','balanced_accuracy'):
                        r[f'{method}_val_{metric}_percent'] = s['models'][method]['val'][metric]*100
                    r[f'{method}_best_epoch'] = s['models'][method]['best_epoch']
                    r[f'{method}_checkpoint_sha256'] = file_sha(job/f'{method}_best.pt')
                r['proto_minus_fc_pp'] = r['proto_val_macro_f1_percent']-r['fc_val_macro_f1_percent']
                assert abs(r['proto_minus_fc_pp']-s['paired_val_macro_f1_delta_proto_minus_fc']*100) < 1e-10
                rows.append(r)
    aggregates = {}
    metrics = ('fc_val_macro_f1_percent','proto_val_macro_f1_percent','proto_minus_fc_pp',
               'fc_val_accuracy_percent','proto_val_accuracy_percent')
    for case in ('skl','abl'):
        aggregates[case] = {}
        for k in (10,20,40):
            group = [r for r in rows if r['case']==case and r['k']==k]
            result = {name:stats([r[name] for r in group]) for name in metrics}
            result['proto_wins_vs_fc'] = sum(r['proto_minus_fc_pp'] > 0 for r in group)
            result['train_size'],result['val_size'] = group[0]['train_size'],group[0]['val_size']
            result['stage1_updates'] = group[0]['stage1_updates']
            aggregates[case][str(k)] = result
    before = read(BASE/'provenance/input_before.json')
    assert before == {name:file_sha(WORK/name) for name in before}, '旧实验、数据或环境配置改变'
    runtime = read(BASE/'provenance/runtime_before.json')
    assert runtime['packages'] == {d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    source = read(BASE/'provenance/source_before.json')
    assert source['tracked_sources'] == {n:file_sha(ROOT/n) for n in source['tracked_sources']}
    assert source['server_helpers'] == {n:file_sha(ROOT/n) for n in source['server_helpers']}
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == COMMIT
    helper_names = ('common.py','audit_shot_scale.py','run_shot_scale.py','verify_shot_scale.py','report_shot_scale.py')
    dump(BASE/'SOURCE_MANIFEST.json', {'source_commit':COMMIT,'tracked_sources':source['tracked_sources'],
         'server_helpers':{f'server_experiment/{n}':file_sha(ROOT/'server_experiment'/n) for n in helper_names}})
    preservation = {'status':'passed','protected_file_count':len(before),
                    'input_before_sha256':file_sha(BASE/'provenance/input_before.json'),
                    'all_existing_k20_outputs_unchanged':True,'all_existing_d0_p0_data_and_env_unchanged':True,
                    'installed_package_versions_unchanged':True,'training_sources_unchanged':True}
    verification = {'status':'passed','new_paired_runs':12,'reused_k20_paired_runs':6,'total_paired_runs':18,
                    'nested_train_prefixes_exact':True,'fixed_raw_val_exact':True,'same_pools_as_k20':True,
                    'all_configs_match_k20_except_k_and_output':True,'new_checkpoints_reloaded_predictions_exact':True,
                    'manual_metrics_and_actual_paired_episodes_verified':True,'test_evaluated':False,'preservation':preservation}
    dump(BASE/'COMPLETION_VERIFICATION.json', verification)
    timing = read(new/'timing.json')
    usage = read(new/'resource_usage.json')
    output = {'source_commit':COMMIT,'protocol':'Frozen P1 shot-scale validation, Train K=10/20/40, fixed Val 5/class',
              'test_evaluated':False,'distance_normalization':'none','temperature':1,'seeds':[42,43,44],
              'new_pairs':12,'reused_k20_pairs':6,'training_window_seconds_new_pairs':timing['wall_seconds'],
              'summary_by_case_and_k':aggregates,'per_seed':rows,'preservation':preservation,
              'caveats':['Val 同时用于最佳 epoch 选择和汇总，非独立 Test 泛化成绩',
                         'n=3，样本/训练随机种子耦合；无统计显著性结论',
                         '固定 Stage1 轮数导致不同 K 的更新次数不同；Stage2 固定400次',
                         '标准化仅拟合各 K 的 Train，因此固定 raw Val 的标准化值随 K 改变',
                         '旧 Data_deduplicated/历史 CNN、ResNet 结果不可混入本对照',
                         '波形去重不等于独立板卡/采集批次；物理单位和完整标签字典尚未确认']}
    dump(BASE/'SUMMARY.json', output)
    with (BASE/'PER_SEED_RESULTS.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    def fmt(s):
        return f"{s['mean']:.2f} ± {s['sample_std']:.2f}"
    lines = ['# 冻结 P1 的少样本规模验证：服务器结果','',
             '**本轮只改变每类训练样本 K=10/20/40，比较 CCSAE-FC 与 CCSAE-Proto（P1）。K10/40 共12组新配对已完成，K20复用6组已核验结果；全部使用固定5 Val/类，没有 Test 评估。**','',
             '## 1. 主要结果','',
             '| 电路 | K | Train / Val 总数 | FC Val Macro-F1 (%) | Proto P1 Val Macro-F1 (%) | 同种子 Proto−FC (pp) | Proto 胜出种子数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for case in ('skl','abl'):
        for k in (10,20,40):
            r=aggregates[case][str(k)]
            lines.append(f"| {case.upper()} | {k} | {r['train_size']} / {r['val_size']} | {fmt(r['fc_val_macro_f1_percent'])} | {fmt(r['proto_val_macro_f1_percent'])} | {fmt(r['proto_minus_fc_pp'])} | {r['proto_wins_vs_fc']}/3 |")
    lines += ['', '均值 ± 样本标准差（ddof=1，n=3）；pp 为百分点。同种子差值先逐组计算再汇总，不以两个独立标准差相减。未四舍五入的指标见 JSON/CSV。','']
    for case in ('skl','abl'):
        a=aggregates[case]
        means=[a[str(k)]['proto_val_macro_f1_percent']['mean'] for k in (10,20,40)]
        monotonic=means[0] <= means[1] <= means[2]
        trend='随 K 增加单调上升' if monotonic else '没有随 K 增加单调上升'
        lines.append(f"{case.upper()} 的 Proto 平均 Val Macro-F1 为 {means[0]:.2f}% → {means[1]:.2f}% → {means[2]:.2f}%，{trend}。各 K 的优势、波动及配对胜出数量见表，不能只挑最高种子报告。")
        lines.append('')
    lines += ['## 2. 冻结内容与数据控制','',
              f'实际训练源码固定为 `{COMMIT}`，分支 `research/ccsae-prototype-diagnostics-v1`；模型/训练/损失源码逐文件与该提交及已有 P1 一致。本轮只新建独立审计、调度、核验和报告脚本。','',
              'P1 判别公式固定为 `logit_c = −Σ_d(z_d−p_cd)² / τ`，τ=1，不除以嵌入维度。没有调整温度、Backbone、损失、支持/查询数量或训练轮数，没有重新搜索超参数。SKL 的复数投影维度500（实虚拼接1000），ABL 为300（拼接600）；保留完整模型，tiny=False。','',
              '原包 SHA256：`11623819dcbb9287ff7ffea4d763d0a6fe3dd77d74c61bfe581bd8ea792b4e49`。合并官方 Train/Test 后，按 float32 波形原字节 SHA256 保留首次出现，SKL 3800、ABL 3100 条唯一波形；按类 70/15/15 分池，split_seed=20261009。SKL 池2660/570/570，ABL 池2170/465/465。这里沿用当前 P1 池，与历史 Data_deduplicated 的逐条划分不同。','',
              '每个电路、每个 sample_seed=42/43/44，Train 都从同一逐类打乱池取前 K 条，已逐条核验 K10=K20 前10条、K20=K40 前20条。Val 从 Val 池固定 val_seed=20261009、每类5条，跨 K 和三种子原始记录及顺序一致；各池无同波形哈希交叉。Train/Val 类数分别 SKL13、ABL11。','',
              '标准化为全局均值/标准差，仅拟合当前选中的 K-shot Train。各 K 的均值/标准差会随 Train 改变，因此固定的是原始 Val 样本和标签，其标准化张量并非跨 K 完全相同。这是既有规则，没有使用全部 Train 池、Val 或留出 Test 来拟合标准化。','',
              '独立重建去重、分池、逐类采样和标准化，与18个上游 loader 输出一致；六个 K20 的完整 split_audit 与旧存档精确一致。全部逐类样本指纹及归一化参数保存于 provenance/splits/。','',
              '## 3. 训练与选择协议','',
              '| 参数 | 固定值 |','|---|---|',
              '| Stage1 | 10 epoch，batch64，signed_real 重构 MSE + SupCon（温度0.1），取最后一轮 |',
              '| Stage2 | 40 epoch × 10 step，每类3 Support + 2 Query，共400次更新/方法 |',
              '| 优化器 | SGD，momentum0.9，梯度裁剪5；SKL lr0.0022/λ0.0017，ABL lr0.0102/λ0.0059 |',
              '| FC | Support+Query 交叉熵，训练 encoder 与 classifier |',
              '| Proto | 可微 Support 均值原型，Query 交叉熵，训练 encoder |',
              '| Val 推理 | FC 使用 classifier；Proto 使用全部 K Train 特征均值原型 |',
              '| 最佳权重 | 固定 Val Macro-F1 最高，平局保留最早；无早停或追加轮数 |','',
              '每个 K/电路/seed 内，FC 与 Proto 共享同一个 Stage1 初始化，使用完全相同的实际 Episode 索引流（seed+901）；训练后的 decoder 以及 Proto classifier 保持共享初始化值。不同 K 的 Stage1 权重和 Episode 数字索引不要求相同，因为选中 Train 和每类候选数量改变；跨 K 控制的是同种子、嵌套样本、固定采样算法与预算。','',
              '| 电路 | K10 Stage1 更新 | K20 Stage1 更新 | K40 Stage1 更新 | Stage2 更新/方法 |',
              '|---|---:|---:|---:|---:|',
              '| SKL | 30 | 50 | 90 | 400 |','| ABL | 20 | 40 | 70 | 400 |','',
              'Stage1 固定轮数而非更新次数；K 增大时每轮 batch 数增加。结果应解释为“固定训练配置下的样本规模效果”，不把全部差异单独归因为标注数量、或声称三档总计算量完全相等。','',
              '## 4. 逐种子结果','',
              '| 电路 | K | seed | FC F1 (%) | Proto F1 (%) | Proto−FC (pp) | FC / Proto 最佳 epoch | 来源 |',
              '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        origin='复用已核验 K20' if r['k']==20 else '本轮服务器'
        lines.append(f"| {r['case'].upper()} | {r['k']} | {r['seed']} | {r['fc_val_macro_f1_percent']:.2f} | {r['proto_val_macro_f1_percent']:.2f} | {r['proto_minus_fc_pp']:+.2f} | {r['fc_best_epoch']} / {r['proto_best_epoch']} | {origin} |")
    negative=[r for r in rows if r['proto_minus_fc_pp']<0]
    lines += ['', f"Proto 低于 FC 的配对共 {len(negative)}/18 组；全部组合都保留，没有因负结果换种子、调参或延长训练。",'',
              '## 5. 服务器、并发与核验','',
              f"复用 `../analog-circuit-crossdomain/.venv/bin/python`，Python3.12.3 / torch {runtime['torch']} / {runtime['gpu']}。通过 PYTHONPATH 导入当前 src，没有重建 uv 环境、安装或下载依赖。四进程、每进程四 CPU 线程；确定性 FP32，禁用 TF32/AMP 与非 math SDP，数值配置与 K20 完全一致。",'',
              f"12组新配对训练窗口 {timing['wall_seconds']:.3f} 秒；平均 GPU 利用率 {usage['mean_gpu_percent']:.2f}%，峰值显存 {usage['peak_gpu_memory_mib']:.0f} MiB。窗口包含工作进程初始化、数据加载和保存，不含审计、独立核验、同步及文档时间；K20不重跑。本轮未测串行正式批次，不能据此宣称实测加速倍数。",'',
              '新封装语法检查通过；18个划分独立重建通过；12组新配对的24个最佳权重重新加载后，Val 预测与存档逐条一致；独立混淆矩阵复算 Accuracy/Macro-F1/Balanced Accuracy 一致；全部40轮曲线、最早最佳 epoch、400步实际 Episode、类平衡、Support/Query 互斥及冻结参数核验通过。原型距离另外对照直接平方距离和 FP64 参考，检查尺度相关浮点误差界，不重新选择 checkpoint。模型源码未改变，沿用上轮已通过的16项上游测试，本轮没有声称重新运行这16项测试。','',
              f"K20沿用原 checkpoint 重载核验，且包括全部旧 K20 文件在内的 {len(before)} 个旧数据/实验/环境文件 SHA256 前后完全相同；已安装包版本清单一致。源文件与训练提交一致。新结果独立保存于 `outputs/proto_shot_scale/server_a4d7a00/`，旧 P1、D0、P0 目录均未覆盖。",'',
              '## 6. 结论的适用范围','',
              '这是固定小 Val 上的学习规模对照：Val 同时用于选最佳 epoch 和汇总，因此表中数值不能标成独立 Test 准确率。每档只有3个耦合采样/训练种子，不是显著性证明；较大波动或非单调变化应据实保留。固定每类5条 Val 限制估计精度，数据此前经过探索也限制完全独立的确认性结论。','',
              '精确波形去重只排除字节相同记录，不能证明板卡、采集批次或仿真来源独立；原始物理单位和完整CSV标签字典仍需发布者核实。本轮只按0…C−1多分类，不依赖健康类别映射。与历史 CNN/ResNet/跨域迁移的划分及预算不同，不混合汇总。','',
              '本轮没有更换距离公式、温度、Backbone 或损失，也没有增加模型模块、使用 Test 搜索方案或因结果不理想追加训练。若要形成论文最终泛化结论，需另立固定候选后的独立 Test 评估阶段；该阶段不属于本次规模验证。','',
              '## 7. 证据与复现边界','',
              '`SUMMARY.json`、`PER_SEED_RESULTS.csv`、`COMPLETION_VERIFICATION.json`、`SOURCE_MANIFEST.json` 保存结果及冻结证据；provenance/ 保存版本/环境/旧文件指纹和18个划分；new_k10_k40/ 保存本轮配置、Train/Val 曲线、预测、配对验证、独立验证、时间及资源采样。K20原始记录见旧 P1 归档，REUSED_K20_VERIFICATION.json 与逐种子路径/权重指纹明确其来源。','',
              '完整权重、Episode 和验证分数 NPZ 保留服务器及本地同步包；GitHub 仅上传文本证据与运行脚本快照，不上传原始 Data.zip、环境或模型权重。服务器入口依赖已有 Data.zip、环境和 K20 结果，输出目标必须为空，不覆盖现有运行。实际命令：','',
              '```bash','export PYTHONPATH="$PWD/src"',
              '../analog-circuit-crossdomain/.venv/bin/python server_experiment/audit_shot_scale.py',
              '../analog-circuit-crossdomain/.venv/bin/python server_experiment/run_shot_scale.py --output outputs/proto_shot_scale/server_a4d7a00/new_k10_k40 --workers 4',
              '../analog-circuit-crossdomain/.venv/bin/python server_experiment/verify_shot_scale.py outputs/proto_shot_scale/server_a4d7a00/new_k10_k40',
              '../analog-circuit-crossdomain/.venv/bin/python server_experiment/report_shot_scale.py','```','',
              '上面是本轮已经执行的归档路径，直接再次运行预检/训练会因目录已存在退出。独立复现需为 BASE 和训练输出指定新的空目录，并保留当前 K20 只读参考路径；不可删除已有结果后重跑。']
    (BASE/'SERVER_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','summary':aggregates,'old_files_preserved':len(before),
                      'new_seconds':timing['wall_seconds']},ensure_ascii=False))

if __name__=='__main__':
    main()
