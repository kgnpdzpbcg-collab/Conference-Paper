"""汇总真实服务器验证结果、配对差值和证据；不继续训练、不选用测试集。"""
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import numpy as np
from audit import existing_records
from common import dump, file_sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/server_proto_k20_7ef4042"
PROVENANCE = ROOT / "server_experiment/provenance"


def summarize(values):
    """百分比均值与样本标准差（ddof=1），三个种子原值全部保留。"""
    values=np.asarray(values,dtype=float)*100
    return {"mean":float(values.mean()),"sample_std":float(values.std(ddof=1)),"values":values.tolist()}


def formatted(value):
    return f"{value['mean']:.2f} ± {value['sample_std']:.2f}"


def main():
    verified=json.loads((OUT / "verification.json").read_text())
    assert verified["status"] == "passed" and verified["paired_runs"] == 6
    assert verified["test_evaluated"] is False
    before=json.loads((PROVENANCE / "existing_artifacts_before.json").read_text())
    after=existing_records()
    assert before == after, "发现旧实验或去重数据变化，停止生成完成报告"
    dump(PROVENANCE / "existing_artifacts_after.json", {"status":"passed","groups":after})
    source_manifest=json.loads((OUT / "source_snapshot/manifest.json").read_text())
    # 核验脚本因浮点误差检查而调整；训练源码、模型和实际运行器必须仍与训练快照一致。
    for name,expected in source_manifest.items():
        if name != "server_experiment/verify.py":
            assert file_sha(ROOT / name) == expected, f"训练后源码变动：{name}"
    snapshot=OUT / "verification_snapshot"
    snapshot.mkdir(exist_ok=True)
    final_files=[ROOT / "server_experiment" / name for name in ("verify.py","report.py","common.py","audit.py")]
    for file in final_files:
        (snapshot / file.name).write_bytes(file.read_bytes())
    dump(snapshot / "manifest.json",{p.name:file_sha(p) for p in final_files})
    env=json.loads((PROVENANCE / "environment.json").read_text())
    audits=json.loads((PROVENANCE / "data_audit.json").read_text())
    timing=json.loads((OUT / "timing.json").read_text())
    resource=json.loads((OUT / "resource_usage.json").read_text())
    groups={}
    rows=[]
    for case in ("skl","abl"):
        pairs=sorted([r for r in verified["results"] if r["case"]==case],key=lambda r:r["seed"])
        assert [r["seed"] for r in pairs] == [42,43,44]
        groups[case]={m:{metric:summarize([r["methods"][m]["metrics"][metric] for r in pairs])
            for metric in ("macro_f1","accuracy","balanced_accuracy")} for m in ("fc","proto")}
        groups[case]["paired_delta_pp"]=summarize([r["paired_delta"] for r in pairs])
        for r in pairs:
            numeric=r["methods"]["proto"]["distance_numerics"]
            assert numeric["fp64_distance_prediction_differences"] == 0
            assert numeric["direct_formula_prediction_differences"] == 0
            rows.append({"case":case,"seed":r["seed"],
                "fc_val_macro_f1_pct":r["methods"]["fc"]["metrics"]["macro_f1"]*100,
                "proto_val_macro_f1_pct":r["methods"]["proto"]["metrics"]["macro_f1"]*100,
                "proto_minus_fc_pp":r["paired_delta"]*100,
                "fc_best_epoch":r["methods"]["fc"]["best_epoch"],
                "proto_best_epoch":r["methods"]["proto"]["best_epoch"]})
    summary={"status":"passed","experiment":"CCSAE V1 FC/Proto K20 P0 validation only",
        "source_commit":env["source_commit"],"environment":env,"groups":groups,"per_seed":rows,
        "test_evaluated":False,"timing":timing,"resource_summary":{k:v for k,v in resource.items() if k!="samples"},
        "previous_split_identical":{case:audits["cases"][case]["old_split_identical"] for case in groups},
        "negative_result":"两个电路全部三个种子 Proto Val Macro-F1 低于 FC；当前快速协议不支持增益假设。"}
    dump(OUT / "SUMMARY.json",summary)
    with (OUT / "PER_SEED_RESULTS.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shanghai=timezone(timedelta(hours=8))
    start=datetime.fromisoformat(timing["started_utc"]).astimezone(shanghai).strftime("%Y-%m-%d %H:%M:%S")
    end=datetime.fromisoformat(timing["ended_utc"]).astimezone(shanghai).strftime("%Y-%m-%d %H:%M:%S")
    lines=["# CCSAE v1 FC/Prototype K=20 服务器实验结果", "",
        "本轮结论：在最新分支固定的 P0 快速验证协议下，SKL 和 ABL 的三个配对种子均为 FC 优于 Prototype，原型模块的性能增益假设未得到支持。这是验证集结果，没有评估 Test。", "",
        "## 核心结果", "", "所有指标单位为 %，均值 ± 样本标准差，n=3；差值为同种子 Proto−FC，单位为百分点。", "",
        "| 电路 | FC Val Macro-F1 | Proto Val Macro-F1 | 配对差值 | FC Val Acc | Proto Val Acc |",
        "|---|---:|---:|---:|---:|---:|"]
    for case,g in groups.items():
        lines.append(f"| {case.upper()} | {formatted(g['fc']['macro_f1'])} | {formatted(g['proto']['macro_f1'])} | {formatted(g['paired_delta_pp'])} | {formatted(g['fc']['accuracy'])} | {formatted(g['proto']['accuracy'])} |")
    lines += ["", "验证集每类恰好 5 条，因此本轮 Balanced Accuracy 与 Accuracy 相同。三次重复同时改变采样种子和训练种子；两方法同种子严格配对。", "",
        "## 各种子完整结果", "", "| 电路 | 种子 | FC Val F1 (%) | Proto Val F1 (%) | Proto−FC (pp) | FC 最佳轮 | Proto 最佳轮 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['case'].upper()} | {row['seed']} | {row['fc_val_macro_f1_pct']:.2f} | {row['proto_val_macro_f1_pct']:.2f} | {row['proto_minus_fc_pp']:+.2f} | {row['fc_best_epoch']} | {row['proto_best_epoch']} |")
    lines += ["", "## 训练协议与环境", "",
        "- 仓库分支 `research/ccsae-prototype-k20`，提交 `7ef4042fe18275f2f366c06c5aa19aa817c45258`。上游 `src/ccsae` 和单元测试文件保持原字节；新增服务器运行、记录和独立核验脚本。",
        f"- 实际复用 `{env['python_executable']}`，通过显式 PYTHONPATH 载入本轮独立 src，没有向旧环境安装或更新包。Python 3.12.3、PyTorch 2.14.1+cu130、NumPy 2.5.3、scikit-learn 1.9.1、CUDA 13.0、cuDNN 92400。",
        "- 准备阶段曾从 uv 缓存创建本轮 .venv，未重下载 PyTorch；按用户要求，预检及全部正式训练使用上述既有环境，新 .venv 没有用于正式训练。",
        "- RTX 5070 Ti 16 GB。FP32，确定性算法，关闭 AMP/TF32/CuDNN benchmark 和 Flash/memory-efficient/cuDNN SDP，仅启用 math SDP。4 个进程，每个进程 4 个 CPU 线程；这是经本轮预检验证的并发配置，没有声称穷举得到最优并发数。",
        "- K=20/类；SKL 260 Train + 65 Val，ABL 220 Train + 55 Val。真实标签预算是每类 20 条训练标签加额外 5 条验证标签。",
        "- 共享 Stage 1：10 epochs，SGD momentum=0.9，batch=64，signed_real 重构 + SupCon，最后一轮权重作为两路共同初始化。SKL LR=0.0022、SCL=0.0017；ABL LR=0.0102、SCL=0.0059。",
        "- Stage 2：两路各 40 epochs × 10 steps=400 次更新，SGD momentum=0.9、clip=5，沿用上述 LR。每类 3 Support + 2 Query，SKL 每步 65 条，ABL 每步 55 条。",
        "- FC 对 Support+Query 做 CE；Proto 用有梯度的 Support 均值构造原型，对 Query 做距离 CE；平方欧氏距离除以嵌入维度，tau=1。两路损失监督位置不同，这是方法定义的差异，不能称所有训练目标完全相同。",
        "- 只从选定 K-shot Train 拟合均值/标准差；验证原型使用该 K-shot Train 全部样本；按固定小 Val Macro-F1 选最佳轮，平局取最早轮。没有早停和超参搜索，没有 Test 推理。", "",
        "## 数据划分与历史结果关系", "",
        "新分支从原始 Data.zip 合并数据，以 float32 波形 SHA256 真正去重后重新生成 70/15/15 池。唯一波形集合与旧数据一致、没有跨池精确重复，但逐条划分不同。相同 split_seed 并不保证不同实现获得相同划分。", "",
        "| 电路 | 新 Train Pool | 新 Val Pool | 新 Test Pool | 新旧 Train 重合 | 新旧 Val 重合 | 新旧 Test 重合 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for case in groups:
        a=audits["cases"][case]
        p=a["pool_sizes"]
        x=a["old_new_intersections"]
        lines.append(f"| {case.upper()} | {p['train']} | {p['val']} | {p['test']} | {x['train']['train']} | {x['val']['val']} | {x['test']['test']} |")
    lines += ["", "SKL 新训练池含旧 Test 中的 402 条波形，ABL 含 327 条；本轮每类 20 条是从新训练池抽取。新 Test 从未在本轮推理，但同一数据集此前已参与其他实验探索，不能把本轮重新划分描述为全新、从未见过的最终评测。完整交集矩阵和样本指纹见 data_audit.json。", "",
        "本轮只有 5 Val/类，旧少样本实验使用完整 Val，且训练预算也不同，因此不能将本轮 FC/Proto 成绩与历史 CCSAE、CNN、ResNet 成绩直接并列比较。需要正式比较时，应冻结相同划分、标签预算和训练协议后重新训练。去重不能证明独立器件或采集批次评估。", "",
        "## 运行时间与核验", "",
        f"- 正式训练批次窗口：北京时间 {start} 至 {end}，耗时 {timing['wall_seconds']:.3f} 秒。只计 6 组配对训练批次（含子进程启动、加载和保存），不含代码准备、环境检查、预检、后续核验和同步；各组耗时并不能相加视为实测串行时间。",
        f"- 窗口内 GPU 平均使用率 {resource['mean_gpu_percent']:.2f}%，峰值显存 {resource['peak_gpu_memory_mib']:.0f} MiB（每秒采样）。",
        "- 在复用的环境中 14 个上游单元测试通过；4 组完整结构 GPU 预检通过；6 组正式配对独立核验通过。核验实际 Episode 索引严格相同、共享初始化完全一致、Stage 2 decoder 冻结、Proto classifier 冻结、最佳轮规则正确，重新加载权重的验证预测与原记录完全一致，混淆矩阵独立复算指标一致。",
        "- 数值检查发现短预检中 4/240 条验证记录的近并列原型距离会受 float32 cdist 舍入影响；最大分数差约 7.9e−8。保留上游实现，未据此改变训练算法。正式最佳权重在全部 360 条验证记录上，cdist、直接平方距离和 FP64 距离三种计算的预测完全一致。正式最大分数差约 1.33e−5，已按嵌入维度与特征范数核验舍入预算；FP64 是固定 checkpoint 诊断，没有重新选模。",
        "- 原去重数据及之前少样本、CCSAE v1、CNN/ResNet 和两类跨域实验的全部记录文件 SHA256 前后完全一致。训练快照保留实际运行源码，verification_snapshot 保存修正后的最终核验脚本。", "",
        "## 结论的使用范围", "",
        "该固定快速协议下，两套电路均未出现原型分类收益，暂不支持将普通原型头写成已验证的性能贡献。SKL 两个原型模型很早达到最佳验证轮，训练预算和表征变化值得后续诊断，但本轮没有额外调参或追加实验。三次重复和小验证集不足以证明所有原型方法、温度或更长训练预算都会失败，也不能充当最终论文 Test 性能。", "",
        "## Material Passport / 证据索引", "",
        "- 固定协议：上游 docs/CCSAE_PROTOTYPE_FEWSHOT_V1.md；核心实现 src/ccsae/{fewshot_data,fewshot_train,prototype}.py。",
        "- 数据来源：用户提供 Data.zip，SHA256 `11623819dcbb9287ff7ffea4d763d0a6fe3dd77d74c61bfe581bd8ea792b4e49`；没有外部补充数据。",
        "- 每种子 config.json、split_audit.json、shared_stage1_history.json、fc_history.json、proto_history.json、summary.json、train.log；共享和两路最佳权重保留在服务器。",
        "- episodes.npz 与 pair_verification.json 记录实际索引流和初始化；verification.json、*_verified_val.npz 保存独立重载、逐类别混淆及预测证据。",
        "- SUMMARY.json、PER_SEED_RESULTS.csv、timing.json、resource_usage.json；server_experiment/provenance 内保存数据、环境、单元测试、数值诊断和旧结果指纹。",
        "- 本报告数字完全来自服务器实测，没有本地训练数字、推测 Test 成绩或挑选种子。", ""]
    (OUT / "SERVER_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    completion={"status":"passed","paired_runs":6,"stage2_runs":12,"source_commit":env["source_commit"],
        "environment_reused":True,"test_evaluated":False,"old_artifacts_unchanged":True,
        "verified_validation_records":360,"fp64_final_prediction_differences":0,
        "report_sha256":file_sha(OUT / "SERVER_RESULTS.md"),"summary_sha256":file_sha(OUT / "SUMMARY.json"),
        "final_verification_script_sha256":file_sha(ROOT / "server_experiment/verify.py")}
    dump(OUT / "COMPLETION_VERIFICATION.json",completion)
    print(json.dumps({"groups":groups,"wall_seconds":timing["wall_seconds"],"old_artifacts_unchanged":True},ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
