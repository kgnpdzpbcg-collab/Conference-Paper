# CCSAE v1 FC/Proto K20 服务器验证记录

本目录发布提交 `7ef4042fe18275f2f366c06c5aa19aa817c45258` 在服务器上的固定 P0 实验结果。两套电路各三个配对种子中 Proto 均低于 FC。本轮只有验证结果，没有 Test 指标，不能作为最终论文测试性能。

[完整报告](../../docs/CCSAE_PROTOTYPE_K20_SERVER_RESULTS.md) · [原始协议](../../docs/CCSAE_PROTOTYPE_FEWSHOT_V1.md)

## 结果与证据

- [逐种子指标](results/server_7ef4042/PER_SEED_RESULTS.csv)、[机器可读汇总](results/server_7ef4042/SUMMARY.json)：Val Macro-F1、Accuracy、Balanced Accuracy、配对差值和最佳轮。
- [独立核验](results/server_7ef4042/verification.json)、[完成记录](results/server_7ef4042/COMPLETION_VERIFICATION.json)：共享初始化、实际 Episode 流、冻结参数、最佳轮、权重重载及混淆矩阵指标检查。
- [训练时间](results/server_7ef4042/timing.json)、[资源采样](results/server_7ef4042/resource_usage.json)：4 个进程，52.907 秒的正式训练批次窗口；不包含准备、预检、核验和同步时间。
- [数据审计](results/server_7ef4042/provenance/data_audit.json)：每类选中 Train/Val 的波形指纹、标准化参数、无跨池精确重复，以及新旧划分完整交集矩阵。新划分和旧实验不同。
- [实际复用环境](results/server_7ef4042/provenance/environment.json)、[数值诊断](results/server_7ef4042/provenance/prototype_numerics_formal.json)、[14 项单元测试记录](results/server_7ef4042/provenance/reused_environment_tests.log)。
- [上游源码指纹](results/server_7ef4042/provenance/upstream_sources.json)、[训练快照指纹](results/server_7ef4042/provenance/training_source_manifest.json)、[最终核验源码指纹](results/server_7ef4042/provenance/final_verification_source_manifest.json)、[旧结果保留核查](results/server_7ef4042/provenance/previous_artifact_preservation.json)。
- [发布前复核](results/server_7ef4042/PUBLICATION_VERIFICATION.json)：复制文件 SHA256 和独立重算汇总指标的检查结果。

`results/server_7ef4042/{skl,abl}/k20_sample<seed>_train<seed>/` 保存每组的配置、数据审计、训练曲线、验证预测、日志与配对指纹，种子为 42、43、44。标签只使用 CSV 类别编号，没有新增未经验证的故障名称映射。

原始波形、数据包、模型权重、完整 Episode 二进制索引、二进制验证分数、虚拟环境不上传 GitHub。全部正式权重及完整运行产物保存在服务器，本地同步包也保留全部正式权重。公开文件不包含原始波形。

## 实际运行来源与脚本快照

实际服务器根目录为 `/home/wang/projects/会议论文/Conference-Paper-ccsae-prototype-k20`，实际复用 `/home/wang/projects/会议论文/analog-circuit-crossdomain/.venv/bin/python`。不安装或更新旧环境包，通过 `PYTHONPATH` 读取本分支代码。

当时实际使用的 [运行器](results/server_7ef4042/runtime_source/run.py)、[数值配置](results/server_7ef4042/runtime_source/common.py)、[数据审计](results/server_7ef4042/runtime_source/audit.py)、[独立核验](results/server_7ef4042/runtime_source/verify.py)、[汇总脚本](results/server_7ef4042/runtime_source/report.py) 按原字节作为审计快照发布。运行器调用上游 `fewshot_train.run`，不修改模型、优化器、更新预算、数据抽样或原型公式，只记录共享权重指纹和实际 Episode 索引流。

这些是原服务器根目录 `server_experiment/` 下的脚本快照，**不能在当前快照目录直接运行**。原数据审计脚本还会严格检查训练提交为 `7ef4042`、只读比对同一工作目录中的旧数据和旧实验。因此后续重现应固定原训练提交和目录结构，将快照复制到仓库根 `server_experiment/` 后使用既有环境，并选择新的输出目录；新增报告提交不是模型版本或重新训练记录。汇总脚本用于固定批次 `outputs/server_proto_k20_7ef4042`，不是通用训练入口。

环境快照文件保留当时锁定版本，实际实验复用既有环境。主模型和配对训练入口始终是原提交的 `src/ccsae/fewshot_train.py`。

## 为什么没有 Test

原分支协议是 P0：先用每类 20 个 Train 标签和额外 5 个 Val 标签检验普通原型头是否值得继续研究。原入口只输出 Val，未实现 Test 预测；本轮按该固定协议执行。Test 不应用于选最佳轮或决定是否保留某个种子。

目前公开内容如实记录 `test_evaluated=false`。若补评 Test，应在固定现有最佳权重、归一化统计和原型构造规则后单独执行，保留本次 Val 报告，不重新训练或根据 Test 更换模型。由于同一数据集已用于之前实验探索，后续也须披露这一限制。
