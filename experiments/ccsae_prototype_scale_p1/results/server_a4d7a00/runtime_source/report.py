"""汇总 D0 与 P1 服务器实测结果，独立计算三种子统计并保留冻结证据。

输出只归档新阶段，不覆盖 P0；原型提升和 FC 配对差值分别计算，避免混淆。
聚合使用样本标准差 ddof=1；所有摘要只包含 Train/Val，无 Test 结果。
"""
from __future__ import annotations
import csv
import importlib.metadata
import json
import statistics
import subprocess
from datetime import datetime,timezone,timedelta
from pathlib import Path

from common import dump,file_sha

ROOT=Path(__file__).resolve().parent.parent
WORK=ROOT.parent
OLD=WORK/'Conference-Paper-ccsae-prototype-k20/outputs/server_proto_k20_7ef4042'
D0=ROOT/'outputs/proto_diagnostics/server_a4d7a00'
P1=ROOT/'outputs/proto_scale_p1/server_a4d7a00'


def stats(values):
    """n=3 的均值和样本标准差；不把三次试验当成显著性证明。"""
    assert len(values)==3
    return {'mean':statistics.mean(values),'sample_std':statistics.stdev(values),'n':3}


def main():
    """读取已通过的核验，冻结源版本、对照表和旧文件保留清单。"""
    d0=json.loads((D0/'SUMMARY.json').read_text())
    assert json.loads((D0/'VERIFICATION.json').read_text())['status']=='passed'
    phases=('seed42_precheck','seeds43_44_followup')
    verification=[]
    timings=[]
    resources=[]
    for phase in phases:
        verification.extend(json.loads((P1/phase/'verification.json').read_text())['results'])
        timings.append(json.loads((P1/phase/'timing.json').read_text()))
        resources.append({'phase':phase,**json.loads((P1/phase/'resource_usage.json').read_text())})
    assert len(verification)==6 and all(r['status']=='passed' for r in verification)
    rows=[]
    for phase in phases:
        for job in sorted((P1/phase).glob('*/k20*')):
            s=json.loads((job/'summary.json').read_text())
            case,seed=s['config']['case'],s['config']['seed']
            old=json.loads((OLD/case/f'k20_sample{seed}_train{seed}'/'summary.json').read_text())
            row={'case':case,'seed':seed,'p0_fc_val_macro_f1_percent':old['models']['fc']['val']['macro_f1']*100,
                 'p0_proto_val_macro_f1_percent':old['models']['proto']['val']['macro_f1']*100,
                 'p1_fc_val_macro_f1_percent':s['models']['fc']['val']['macro_f1']*100,
                 'p1_proto_val_macro_f1_percent':s['models']['proto']['val']['macro_f1']*100,
                 'p1_proto_best_epoch':s['models']['proto']['best_epoch'],
                 'phase':phase,'job_relative_path':str(job.relative_to(P1))}
            assert row['p0_fc_val_macro_f1_percent']==row['p1_fc_val_macro_f1_percent']
            row['p1_proto_minus_fc_pp']=row['p1_proto_val_macro_f1_percent']-row['p1_fc_val_macro_f1_percent']
            row['p1_proto_minus_p0_proto_pp']=row['p1_proto_val_macro_f1_percent']-row['p0_proto_val_macro_f1_percent']
            rows.append(row)
    rows.sort(key=lambda r:(r['case'],r['seed']))
    assert {(r['case'],r['seed']) for r in rows}=={(c,s) for c in ('skl','abl') for s in (42,43,44)}
    summary={}
    for case in ('skl','abl'):
        case_rows=[r for r in rows if r['case']==case]
        summary[case]={key:stats([r[key] for r in case_rows]) for key in (
            'p1_fc_val_macro_f1_percent','p0_proto_val_macro_f1_percent','p1_proto_val_macro_f1_percent',
            'p1_proto_minus_fc_pp','p1_proto_minus_p0_proto_pp')}
        summary[case]['p1_proto_wins_vs_fc']=sum(r['p1_proto_minus_fc_pp']>0 for r in case_rows)
        summary[case]['p1_proto_wins_vs_p0_proto']=sum(r['p1_proto_minus_p0_proto_pp']>0 for r in case_rows)
    input_before=json.loads((D0/'input_before.json').read_text())
    input_now={name:file_sha(WORK/name) for name in input_before}
    assert input_before==input_now,'旧结果、数据、环境文件发生变化'
    runtime=json.loads((D0/'runtime.json').read_text())
    assert runtime['packages']=={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    preserve={'status':'passed','protected_file_count':len(input_before),
              'before_sha256':file_sha(D0/'input_before.json'),
              'old_p0_all_files_preserved':True,'raw_data_and_frozen_splits_preserved':True,
              'existing_uv_configuration_and_packages_preserved':True}
    source_files=subprocess.check_output(['git','ls-files','src','tests','pyproject.toml','docs/research/CCSAE_PROTO_K20_FAILURE_ANALYSIS.md'],cwd=ROOT,text=True).splitlines()
    source_hashes={name:file_sha(ROOT/name) for name in source_files}
    # 校验当前上游源码原字节与实际运行的 Git 提交一致；服务器封装单独留存。
    for name,h in source_hashes.items():
        import hashlib
        blob=subprocess.check_output(['git','show','a4d7a008d7e7597a864624f63f4185d91fada144:'+name],cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest()==h
    dump(P1/'SOURCE_MANIFEST.json',{'source_commit':'a4d7a008d7e7597a864624f63f4185d91fada144','tracked_sources':source_hashes,
         'server_helpers':{str(p.relative_to(ROOT)):file_sha(p) for p in sorted((ROOT/'server_experiment').glob('*.py'))}})
    output={'source_commit':'a4d7a008d7e7597a864624f63f4185d91fada144','p0_training_commit':'7ef4042fe18275f2f366c06c5aa19aa817c45258',
            'stage':'D0 checkpoint diagnostics then gated P1 distance normalization ablation',
            'test_evaluated':False,'distance_normalization':'none','temperature':1,
            'd0_seconds':d0['wall_seconds'],'p1_training_window_sum_seconds':sum(t['wall_seconds'] for t in timings),
            'summary_by_case':summary,'per_seed':rows,'old_artifact_preservation':preserve,
            'next_action':'本轮止于预先指定的 D0/P1；不新增模块、温度搜索或 Test 方案筛选。',
            'caveats':['只评估固定 5 Val/类','n=3，样本选择与训练随机种子耦合','ABL seed43 略低于 FC，整体波动较大',
                       'D0 分析 Stage1 last，不能等同于 Proto best 的最终表征因果机制','先前数据探索限制持续存在']}
    dump(P1/'SUMMARY.json',output)
    dump(P1/'COMPLETION_VERIFICATION.json',{'status':'passed','d0_runs':6,'p1_paired_runs':6,
        'source_files_match_training_commit':True,'p0_stage1_and_fc_state_exact':True,'p0_episode_stream_exact':True,
        'independent_checkpoint_reload_predictions_and_metrics_verified':True,
        'all_old_input_hashes_match':True,'test_evaluated':False,'preservation':preserve})
    with (P1/'PER_SEED_RESULTS.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# CCSAE 原型诊断 D0 与距离尺度 P1：服务器验证结果',
           '', '实际源码提交：`a4d7a008d7e7597a864624f63f4185d91fada144`；分支 `research/ccsae-prototype-diagnostics-v1`。原 P0 训练提交 `7ef4042`。',
           '', '**结果：距离尺度由除以嵌入维度 D 改为不除以 D 后，六组 P1 原型均优于各自 P0 原型；SKL 三组均优于 FC，ABL 两组优于 FC、一组略低。仅为固定小验证集结果，没有 Test 评估。**',
           '', '## 1. 执行顺序与版本边界',
           '', '按最新 MD：16 项上游测试通过 → 六组旧 Stage1 checkpoint 的 D0 → D0 有尺度/梯度证据后，两电路 seed42 正式预算预检 → 两组原型明显改善，补 seed43/44 → 独立核验全部六组。没有跳过门槛，也没有在 Test 选择方案。',
           '', '本地及服务器新建独立诊断分支目录，旧 P0 checkout 仍在原目录且训练源码固定。复用已有 `analog-circuit-crossdomain/.venv`，Python 3.12.3 / torch 2.14.1+cu130 / RTX 5070 Ti，确定性 FP32、禁用 TF32/AMP、math SDP、每工作进程 4 CPU 线程。通过 PYTHONPATH 直接导入新源码，没有执行 pip install，没有创建环境或下载包。',
           '', '## 2. D0：旧权重只读诊断',
           '', '加载的是 `shared_stage1_last.pt`，不是训练后的 `proto_best.pt`。构造原 P0 的 K=20 Train 和 5 Val/类，逐类样本和 Train/Val/Test 池哈希均核验一致。为重建这一划分，loader 会读取原包的官方 Train/Test 文件；没有对新协议的留出 Test 池生成模型预测。',
           '', '| 电路 | seed | 固定 Stage1 全 K 原型 Val F1 (%) | 3-shot Val F1 均值±SD (%) | 投影梯度 none/dim | 卷积梯度 none/dim | 类内距离/最近类中心距离 | Support 漂移均值 | BN 相对 L2 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for j in sorted(d0['jobs'],key=lambda r:(r['case'],r['seed'])):
        r=j['result'];g=r['one_episode_train_mode_gradients'];s=r['support_stability'];sp=s['val_macro_f1_across_sampled_supports']
        lines.append(f"| {j['case'].upper()} | {j['seed']} | {r['stage1_full_K_val_macro_f1']*100:.2f} | {sp['mean']*100:.2f} ± {sp['std']*100:.2f} | {g['none']['projection_gradient_l2']/g['dim']['projection_gradient_l2']:.2f} | {g['none']['first_complex_conv_gradient_l2']/g['dim']['first_complex_conv_gradient_l2']:.2f} | {r['embedding_geometry']['mean_within_to_nearest_center_ratio']:.3f} | {s['three_shot_to_full_center_relative_drift']['mean']:.3f} | {r['batchnorm_proxy']['same_batch_train_vs_eval_mean_relative_l2']:.3f} |")
    lines += ['', '32 次 Support 抽样的 SD 是同一固定 checkpoint 内的抽样波动；它与后面的三种子 SD 不是同一种不确定性。Support 漂移按最近异类中心间距归一，BN 为同一 Episode batch 下 train/eval 特征的代理比较。几何统计按类分别计算再平均，类内/类间比值均值不等于两个均值之比。',
              '', '全部六组 dim/none 固定特征预测一致，且与 FP64 cdist 参考一致；仅改推理分数不能提高 F1。原 dim 的 Episode CE 接近 ln(13) 或 ln(11)，softmax 近均匀，梯度路径正常但相对 none 更小。投影梯度倍率约 20–157，卷积约 17–155；它们是两个参数张量的未裁剪梯度，不能当作整个模型优化步长倍率。',
              '', '类中心重叠和 3-shot 原型漂移也存在，因而不能只凭 D0 断言尺度是唯一原因。D0 模型参数和 BN buffers 运行后逐字节恢复；确定性复算一致；独立直接 FP64 平方距离和 sklearn 指标复算一致。',
              '', '## 3. P1：严格单因素对照',
              '', '唯一算法因素：原型 logits 从 `−||z−p||²/(D·τ)` 改为 `−||z−p||²/τ`，τ 固定为 1，SKL D=1000、ABL D=600。FC 的训练及判别公式不变。',
              '', '其他条件：K=20、Val=5/类、split/val seed=20261009、sample/training seed=42/43/44；Stage1 10 epoch、batch64、signed_real MSE+SupCon；Stage2 40 epoch×10 step；每类 3 Support+2 Query；SGD momentum0.9、clip5，SKL lr0.0022/λ0.0017，ABL lr0.0102/λ0.0059。全部 K Train 生成验证原型，Val Macro-F1 选最佳 epoch，平局保留最早。',
              '', '本轮重新执行 Stage1 与 FC，用于核验与 P0 的严格复现。六组 Stage1 参数/buffer、FC 最佳参数/buffer、FC 全训练曲线及实际 Episode 样本流均与 P0 完全一致。FC checkpoint 文件含新增配置字段，因此比较的是 state_dict 指纹，不能要求整个 checkpoint 文件字节相同。',
              '', '| 电路 | FC Val Macro-F1 (%) | P0 Proto dim (%) | P1 Proto none (%) | P1−FC (pp) | P1−P0 Proto (pp) |',
              '|---|---:|---:|---:|---:|---:|']
    def fmt(s):
        return f"{s['mean']:.2f} ± {s['sample_std']:.2f}"
    for case,s in summary.items():
        lines.append(f"| {case.upper()} | {fmt(s['p1_fc_val_macro_f1_percent'])} | {fmt(s['p0_proto_val_macro_f1_percent'])} | {fmt(s['p1_proto_val_macro_f1_percent'])} | {fmt(s['p1_proto_minus_fc_pp'])} | {fmt(s['p1_proto_minus_p0_proto_pp'])} |")
    lines += ['', '均值 ± 样本标准差（ddof=1，n=3）；差值为同种子配对差值的均值与 SD。',
              '', '| 电路 | seed | FC (%) | P0 Proto (%) | P1 Proto (%) | P1−FC (pp) | P1 最佳 epoch |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case'].upper()} | {r['seed']} | {r['p1_fc_val_macro_f1_percent']:.2f} | {r['p0_proto_val_macro_f1_percent']:.2f} | {r['p1_proto_val_macro_f1_percent']:.2f} | {r['p1_proto_minus_fc_pp']:+.2f} | {r['p1_proto_best_epoch']} |")
    lines += ['', 'ABL seed43 比 FC 低约 0.40 pp；ABL 的 P1 三种子 SD 为 7.03 pp。不能写成“每组全面优于 FC”或已验证稳定 Test 泛化。ABL 三组最佳 epoch 均在 39–40，本轮按预设预算结束，没有因结果追加训练轮数。',
              '', '## 4. 时间与资源',
              '', f"D0 六组四进程诊断窗口 {d0['wall_seconds']:.3f} 秒；seed42 两进程训练 {timings[0]['wall_seconds']:.3f} 秒；seed43/44 四进程训练 {timings[1]['wall_seconds']:.3f} 秒。P1 两批训练窗口合计 {sum(t['wall_seconds'] for t in timings):.3f} 秒，不含同步、核验、报告生成。没有测量整批串行正式运行，不推算正式并发加速比。",'', '| 阶段 | 平均 GPU 利用率 | 峰值显存 MiB |', '|---|---:|---:|']
    for r in resources:
        lines.append(f"| {r['phase']} | {r['mean_gpu_percent']:.2f}% | {r['peak_gpu_memory_mib']:.0f} |")
    lines += ['', '资源每秒采样，窗口包括子进程启动、加载与保存。原始采样和时间戳保存于各批次 JSON。',
              '', '## 5. 核验、结论和限制',
              '', f"16 项测试通过；6 组 D0 和 6 组 P1 配对独立核验通过。{len(input_before)} 个旧结果、数据及环境配置文件指纹与执行前一致，已安装包版本清单一致。六组 P1 checkpoint 重载后的验证预测与保存结果完全一致，独立混淆矩阵复算 F1/Accuracy/Balanced Accuracy 一致，Support/Query 互斥且按类平衡，未训练 decoder 和 Proto classifier 保持原值。", 
              '', '单因素对照支持“距离分数尺度明显影响本协议下的原型训练优化结果”。它没有证明尺度是所有差距的唯一原因，也没有证明普通原型在所有数据、训练预算或真实部署条件下优于 FC。D0 的几何和 BN 代理检查仍不能单独建立因果归因。',
              '', '只评估固定每类 5 条 Val，n=3且采样/训练随机种子耦合；没有 Test 评估。P0 新划分与历史 Data_deduplicated 不同，不能混用旧 CNN/ResNet/full-data/few-shot 成绩。原始物理单位、完整故障列索引、独立板卡/批次来源未被本轮解决；本轮多分类不依赖 C0=健康的假设。数据曾用于先前探索的限制持续存在。',
              '', '本轮完成文档预定的 D0/P1，停止追加模块与参量搜索。若继续研究，应先冻结主协议和候选配置，再单独决定下一阶段评估；不因某个结果追加 Test 筛选。',
              '', '## 6. 原始证据与复现',
              '', 'D0：`outputs/proto_diagnostics/server_a4d7a00/`；P1：`outputs/proto_scale_p1/server_a4d7a00/`。所有完整权重与 Episode NPZ 留在服务器及本地同步包，GitHub 只归档 JSON/CSV/日志/源代码快照，禁止上传 Data.zip 或模型权重。',
              '', '复用实际 Python 环境，以新项目根为工作目录：',
              '', '```bash', 'export PYTHONPATH="$PWD/src"', 'export CUBLAS_WORKSPACE_CONFIG=:4096:8', 'export OMP_NUM_THREADS=4', 'export MKL_NUM_THREADS=4',
              '../analog-circuit-crossdomain/.venv/bin/python -m pytest -q',
              '../analog-circuit-crossdomain/.venv/bin/python -m ccsae.proto_diagnostics --data ../Data.zip --run-dir ../Conference-Paper-ccsae-prototype-k20/outputs/server_proto_k20_7ef4042/skl/k20_sample42_train42 --seed 42 --device cuda --repeats 32 --output outputs/proto_diagnostics/fresh/skl42.json',
              '```',
              '', '实际并发封装与验证脚本见 `server_experiment/` 的运行源快照。为复现 FP32 数值，实际封装还调用 common.configure() 禁用 TF32/AMP 和非 math SDP；仅复制 CLI 命令不保证默认后端完全一致。新输出必须另选空目录，正式权重不能从 GitHub 文本归档取得。',
              '', '关键证据：D0 VERIFICATION.json、P1 各批 verification.json、COMPLETION_VERIFICATION.json、SOURCE_MANIFEST.json、PER_SEED_RESULTS.csv、全部 Train/Val 曲线和固定最佳预测。']
    (P1/'SERVER_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','summary':summary,'old_files_preserved':len(input_before)},ensure_ascii=False))

if __name__=='__main__':
    main()
