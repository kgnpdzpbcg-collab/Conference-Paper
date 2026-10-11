# CNN、ResNet1D、CCSAE v1 历史少样本实验：服务器 Test 结果

本报告同步已经完成的历史少样本实验 `fewshot_server_v1`。本次没有重新训练或执行模型推理，结果来自服务器原始存档；与后续 CCSAE-FC/Proto 的固定5 Val/类协议独立保存。

## Test Macro-F1

单位为%；种子42、43、44的均值 ± 样本标准差（ddof=1，n=3）。共54次少样本训练：两个电路×三个模型×三个K×三个种子。

| 电路 | 模型 | K=10 | K=20 | K=40 |
|---|---|---:|---:|---:|
| SKL | CNN | 2.43 ± 2.26 | 4.01 ± 3.15 | 17.83 ± 3.98 |
| SKL | ResNet1D | 20.30 ± 9.68 | 49.69 ± 3.84 | 78.00 ± 3.00 |
| SKL | CCSAE v1 | 50.89 ± 4.45 | 59.49 ± 1.64 | 61.98 ± 2.71 |
| ABL | CNN | 23.28 ± 9.52 | 51.87 ± 4.09 | 58.48 ± 5.69 |
| ABL | ResNet1D | 60.58 ± 3.30 | 69.61 ± 5.37 | 81.27 ± 2.27 |
| ABL | CCSAE v1 | 68.55 ± 2.29 | 70.68 ± 2.38 | 75.27 ± 6.45 |

**观察：在各自既定训练配置下，两套电路的K=10、20均为CCSAE v1平均Test Macro-F1更高，K=40均为ResNet1D更高。没有统计显著性或架构因果结论。**

SKL低样本CNN的训练损失接近ln(13)，K=10最佳模型只预测1–2个类别，提示该配置下训练未充分学到类别差异。因此不能将其低分直接归因于CNN架构。具体训练轮数、损失、更新次数和预测类别数见[原始训练诊断](../experiments/analog_circuit_fewshot_server_v1/results/server_fewshot_v1/TRAINING_DIAGNOSTICS.md)。

## 固定数据与训练协议

- 使用原先保存的去重Train/Val/Test划分，split_seed=20261009；SKL池2660/570/570，ABL池2170/465/465。
- 从原Train逐类抽K=10/20/40，种子42/43/44，同种子K10⊂K20⊂K40；三模型使用相同输入及标签。SKL Train总数130/260/520，ABL为110/220/440。
- 各K归一化仅拟合选中的Train；未选Train不用于预训练、分类训练、归一化或BN重估。Val/Test保留原始波形、标签及顺序，SKL各570条、ABL各465条。

| 模型 | 训练配置与选模 |
|---|---|
| CCSAE v1 | signed_real重构；Stage1 100轮、按完整Val总损失最低选权重；Stage2 200轮、按完整Val Macro-F1最高选权重；batch64、SGD momentum0.9、clip5；SKL lr0.0022/λ0.0017，ABL lr0.0102/λ0.0059，SupCon温度0.1 |
| CNN / ResNet1D | batch128、AdamW lr0.001/weight_decay0.0001；最多100轮，Val Macro-F1连续15轮不升早停；Plateau factor0.5/patience5；每轮只使用选中的Train重估BN |

CCSAE v1模型源码基于提交`d4413979d27160c6a96690f9b544738c6ca92bd7`，实际服务器代码快照和注入固定划分的运行脚本一并归档。无数据增强、无额外预训练，均从随机初始化训练。各模型的训练预算不同，不能将结果差异完全归因于模型结构。

全部54次少样本训练先按Val选模、冻结权重，再统一评估Test；没有用Test选择K、种子、训练轮数或超参数。原始汇总还含完整训练集上下文；本表仅报告K=10/20/40。

## 与当前 FC/Proto 实验的区别

CCSAE-FC沿用V1网络与原FC头，但当前配对协议的数据池逐条划分、训练预算和验证规模都已改变：Stage1为10轮、Stage2为400次Episode更新，Val每类5条，报告Val而非Test。本表只能与同一历史协议的CNN/ResNet1D比较，不能直接混入当前P1表或解释为P1相对历史V1的提升。

## 环境、核验与原始证据

原训练在RTX 5070 Ti上运行，torch2.14.1+cu130/CUDA13.0，复用既有uv环境、确定性FP32、禁用AMP/TF32。54次训练函数累计575.14秒，包含逐轮验证和保存，不含准备、独立核验及测试阶段。

原训练/测试核验已通过，包括子集来源、嵌套采样、Train-only归一化、选模规则、权重重载预测和独立指标。本次同步额外核对284个文本文件与服务器原字节一致，并从54组存档Val/Test混淆矩阵重新计算指标，复算18个少样本组合的三种子均值/样本标准差。没有重新执行权重推理或训练。

精确去重不等于独立板卡/采集批次泛化；标准差同时包含抽样和训练随机性。上述限制随历史结果一起保留。

[实验索引及可运行文本核验脚本](../experiments/analog_circuit_fewshot_server_v1/README.md) · [服务器原始完整报告](../experiments/analog_circuit_fewshot_server_v1/results/server_fewshot_v1/FEWSHOT_SERVER_RESULTS.md)

GitHub保存结果、曲线、划分元数据、配置、核验记录和源码快照。原始波形数据、缓存数组、预测NPZ、模型权重和环境不上传。
