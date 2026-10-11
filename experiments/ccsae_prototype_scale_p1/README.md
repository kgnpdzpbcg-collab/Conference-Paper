# CCSAE 原型 D0 / 距离尺度 P1 实验归档

正式服务器源码提交为 `a4d7a008d7e7597a864624f63f4185d91fada144`，继承 P0 `7ef4042`。本阶段按最新研究文档先做六组只读 Stage1 诊断，再依据诊断执行一次 `dim→none` 距离尺度重训；seed42 预检有改善后，补齐43、44。

[完整服务器报告](../../docs/CCSAE_PROTOTYPE_DIAGNOSTICS_P1_SERVER_RESULTS.md) · [研究方案与门槛](../../docs/research/CCSAE_PROTO_K20_FAILURE_ANALYSIS.md)

| Val Macro-F1，均值 ± 样本 SD (%)，n=3 | FC | P0 Proto dim | P1 Proto none |
|---|---:|---:|---:|
| SKL | 59.74 ± 1.87 | 46.99 ± 3.10 | 70.27 ± 3.43 |
| ABL | 61.94 ± 3.14 | 42.98 ± 1.21 | 68.86 ± 7.03 |

全部六组 P1 原型优于 P0 原型；对 FC 则 SKL 3/3、ABL 2/3。ABL seed43 低约0.40 pp，只评估固定5 Val/类，没有 Test，不能作为最终泛化结论。

## 文件与核验证据

- `results/server_a4d7a00/d0/`：六组 Stage1 权重机制诊断、32次 Support 抽样、输入保留指纹、运行环境、独立验证。
- `results/server_a4d7a00/p1/`：总汇总、逐种子表、两个正式预算批次的配置/训练曲线/最佳 Val 预测/实际配对核验及资源记录。
- `results/server_a4d7a00/runtime_source/`：服务器实际封装与核验脚本、16项测试及执行日志的原字节快照。
- `SOURCE_MANIFEST.json`：固定训练源码与封装指纹；`COMPLETION_VERIFICATION.json`：六组独立核验全部通过。
- `PUBLICATION_VERIFICATION.json`：已同步服务器文本与归档文件 SHA256 对应。

六组 P1 的 Stage1 state_dict、FC 最佳 state_dict/全曲线、实际 Episode 序列精确复现 P0；所有170个旧输出/数据/环境配置文件及环境包版本保持一致。旧 P0 归档没有改写。

## 复现边界

完整 Data.zip、旧 Stage1/P0 权重、新 P1 权重、Episode/score NPZ、uv环境只保存在服务器和本地同步包，不上传GitHub。这里的文本快照不能替代权重。

实际服务器工作区为 `/home/wang/projects/会议论文/Conference-Paper-ccsae-prototype-diagnostics-v1`，复用其上一级 `analog-circuit-crossdomain/.venv`，从本项目 src 导入新代码；没有安装或更新包。原始权重位于相邻的 P0 项目。

`runtime_source` 是审计快照，不是可直接在当前目录执行的新入口。复现实际封装应固定源码到训练提交 `a4d7a00`，将这些 Python 脚本放入项目根 `server_experiment/`，保持历史数据/权重的目录关系，并为每轮运行选择新的空输出目录。D0 封装的源码版本断言要求训练提交；本次归档提交仅更新文档和证据，不改变已运行模型代码。

P0 新划分与历史 Data_deduplicated 不同，本结果只能与同池、同预算、同种子的 P0 FC/Proto 比较。所有方法选择基于Train/Val；已有探索和缺失采集来源元数据的限制继续保留。
