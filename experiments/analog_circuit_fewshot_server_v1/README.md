# 历史 CNN / ResNet1D / CCSAE v1 少样本服务器 Test 归档

[六行K=10/20/40结果与协议说明](../../docs/FEWSHOT_BASELINES_CCSAE_V1_SERVER_TEST_RESULTS.md)

实验标识`fewshot_server_v1`，54次少样本训练，固定历史去重划分，三种子42/43/44。与后续P0、P1和FC/Proto规模验证分别归档，不覆盖旧结果。

## 文件

- `results/server_fewshot_v1/FEWSHOT_SERVER_RESULTS.md`：原服务器完整报告；原汇总也保留完整Train对照上下文。
- `summary.json`、`summary.csv`、`per_seed.csv`：原始汇总及逐种子指标。
- `plan.json`：训练配置、数据/源码指纹、原环境信息；`prepared/`：18个子集索引和Train-only归一化元数据。
- 各K/模型/seed目录：原始result.json、训练曲线、选模及数据审计记录；模型权重与预测NPZ省略。
- `training_validation.json`、`validation.json`：原训练阶段及最终Test核验记录。
- `TRAINING_DIAGNOSTICS.md`、`training_diagnostics.json`：SKL低样本CNN的训练诊断，保留训练不充分及预算差异说明。
- `source_snapshot/`：实际服务器Python源文件快照，不修改当前仓库src。
- `SERVER_SOURCE_MANIFEST.json`：284个服务器原文件的路径、字节数及SHA256；`PUBLICATION_VERIFICATION.json`：本次文本与统计核验。

## 本次同步核验

284个文本文件与服务器原件一致。54组存档Val/Test混淆矩阵指标、18个少样本组合的三种子均值及样本标准差独立复算通过。没有新训练或模型推理，权重重载与原始数据追溯采用此前已通过的核验记录。

在仓库根运行，只需Python标准库：

```bash
python experiments/analog_circuit_fewshot_server_v1/verify_archive.py
```

本脚本只读检查文本指纹与已存混淆矩阵，不会重训、修改原件或替代完整权重重载验证。

完整复现实验还需服务器原Data_deduplicated、权重、数组缓存和既有uv环境；这些文件不上传GitHub。原服务器路径为`/home/wang/projects/会议论文/analog-circuit-fewshot/outputs/server_fewshot_v1`，少样本运行源码见快照中的fewshot/run_fewshot.py。
