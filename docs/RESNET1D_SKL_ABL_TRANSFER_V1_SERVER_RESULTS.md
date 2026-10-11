# ResNet1D：SKL预训练是否帮助ABL少样本诊断？

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-10-11
- Verification Status: VERIFIED（权重/真实采样/冻结BN/Val/Test重载）；统计为固定Test上的条件推断
- Version Label: resnet_transfer_v1_server_a4d7a00

固定原ResNet1D，只比较Target-only、Frozen Transfer、Fine-tuning Transfer。提前固定10个配对种子42–51、K=10/20/40、同一ABL Train/Val/Test，不以90%准确率为目标。

**本轮结论：在相同ABL标注预算下，没有观察到SKL预训练带来经多重比较校正后的显著提升。** Fine-tuning在K=40平均提高1.05个百分点，但置信区间跨0、Holm p=1；Frozen在K=40下降7.11个百分点、Holm p=0.046875，达到预定显著性标准。其余负向比较未通过校正后的显著性标准。

## 最终ABL Test Macro-F1

单位：%，10种子的均值 ± 样本标准差。

| K | Target-only | Frozen Transfer | Fine-tuning Transfer |
|---:|---:|---:|---:|
| 10 | 56.81 ± 3.48 | 52.94 ± 3.05 | 56.10 ± 4.08 |
| 20 | 58.72 ± 4.71 | 54.18 ± 3.01 | 56.46 ± 2.73 |
| 40 | 59.11 ± 3.14 | 52.01 ± 4.44 | 60.16 ± 9.17 |

## 相比Target-only的迁移增益

单位：Macro-F1百分点。95% CI来自10个配对种子的20000次bootstrap。p为双侧精确配对符号翻转检验，穷举1024种符号配置，六项比较共同Holm校正。

| K | 条件 | 配对均值增益 | 95% CI | 胜/10 | 原始p | Holm p |
|---:|---|---:|---|---:|---:|---:|
| 10 | Frozen Transfer | -3.86 | [-6.91, -0.66] | 2 | 0.041016 | 0.164062 |
| 10 | Fine-tuning Transfer | -0.70 | [-3.94, +2.07] | 5 | 0.662109 | 1.000000 |
| 20 | Frozen Transfer | -4.54 | [-8.39, -1.35] | 2 | 0.019531 | 0.097656 |
| 20 | Fine-tuning Transfer | -2.27 | [-5.71, +1.00] | 5 | 0.248047 | 0.744141 |
| 40 | Frozen Transfer | -7.11 | [-10.19, -3.73] | 2 | 0.007812 | 0.046875 |
| 40 | Fine-tuning Transfer | +1.05 | [-2.75, +5.49] | 4 | 0.669922 | 1.000000 |

## 预指定主要问题

主比较提前固定为K=20的Fine-tuning Transfer相对Target-only：平均差-2.27个百分点，Holm p=0.744141。本轮没有得到“预训练显著提高该设置性能”的统计证据。
所有K和条件均报告；没有按Test挑最佳K或方法，也没有因为结果不足而追加种子/调整超参数。平均为负的条件保留为负迁移结果。

## 数据与训练协议

- 当前P1首出现float32去重池，split_seed=20261009。SKL源Train全部2660条用于训练采样，固定65条Val选源权重；SKL Test不进入网络推理或选模。
- ABL每类K=10/20/40，共110/220/440条Train；同种子嵌套，同三条件样本完全相同。固定5 Val/类，共55条，完整Test465条（标签0有15条，其余10类各45条）。
- 每次运行目标标注预算为11*K+55，分别165/275/495条；源预训练额外使用SKL标签与计算，不能声称总数据/总计算相同。
- 源归一化仅拟合全部SKL Train；目标归一化仅拟合该组ABL K-shot Train，同组共享统计量。剩余ABL Train、额外Val池和Test不参与训练或BN适配。
- 10个源预训练模型是本轮新训练：SKL13类、SGD lr0.0022。源规模固定，不随目标K变化；源最佳骨干由SKL Val选出，两迁移条件使用同一份源权重。
- 目标三条件均为新的11类头，seed+2000固定相同初值；不迁移SKL13类头。Target-only骨干随机；Frozen复制源骨干、只训练头并永久冻结BN；Fine-tuning复制同一骨干、全参数和训练BN适配。
- 源与目标均无Stage1，分类阶段40个验证周期×10步=400次更新；每步每类3 Support+2 Query，全部5条参与CE。目标条件逐步使用完全相同的实际Episode；源Train全量可采样不等于遍历完整数据40轮。
- 目标SGD lr0.0102，momentum0.9、weight_decay0、全局梯度裁剪5，fresh optimizer，无scheduler/early stopping/增强/BN重校准。正式预算提前固定，不另搜索学习率。
- Val Macro-F1最高权重选模，同分最早轮次。全部10源+90目标训练核验后冻结90个目标候选，再统一Test。确定性FP32、禁TF32、eval batch64、4进程×4CPU线程。

## 源域预训练质量与结论范围

10个源模型的SKL Val Macro-F1为**53.61 ± 7.22%**；平均训练损失从首验证周期2.56降至末周期0.87。训练发生了有效更新，但源域验证性能有限。本轮只能回答当前400步、当前归一化和优化配置的迁移效果，不能推出“任何SKL预训练都无效”，也不能仅凭本轮结果确定负迁移的机制。没有使用SKL Test补选源权重。

Bootstrap置信区间是各比较的未校正区间；显著性结论依据六项共同Holm校正后的p值。K10、K20 Frozen的区间虽低于0，校正后的检验未达到0.05。

## 解释边界

- Frozen回答源域特征在不适配时是否有用；Fine-tuning回答源域初始化经过目标适配是否有用。Fine与Frozen同时改变参数更新和BN适配，本轮不拆解BN的单独贡献。
- 本次固定Test已用于历史实验，不是全新盲测；本轮方案和统计在看新结果前冻结。精确去重及跨电路无完全重复，不证明物理采集/电路板独立性。
- 统计单位是配对种子，不把465条Test或30个跨K结果当作独立重复。六项检验共用Test且跨K相关，通过共同Holm处理多重比较。
- 符号翻转依赖零假设下的配对可交换/对称性；bootstrap CI描述固定Test上训练/抽样波动，不是未知采集批次的总体置信区间。10种子不构成功效保证，结论只针对本轮协议。

## 核验与文件

- 10源+90目标全部检查随机/迁移初始化、同组新头、真实采样、400次更新、BN计数、冻结骨干及Val最佳选择。90份目标最佳权重均独立重载复现Test分数与预测，混淆矩阵与sklearn指标一致。
- 6组配对差的均值/标准差/精确p另用Python标准库穷举复算；统计风险扫描11/11项有记录。原数据、模型源码、旧输出和环境包版本保持指纹不变。
- 证据：PROTOCOL/PREFLIGHT、provenance划分、SOURCE/TARGET_TRAINING、TRAINING_VERIFICATION、CHECKPOINT_MANIFEST、TEST_EVALUATION/VERIFICATION、SUMMARY、PER_SEED_RESULTS、STATISTICAL_VERIFICATION及逐模型记录。

- source训练窗口：32.46秒，10模型，含进程启动及读取/保存，不含独立核验、测试、报告或同步。
- target训练窗口：178.84秒，90模型，含进程启动及读取/保存，不含独立核验、测试、报告或同步。

## 独立归档与复现

- 训练/核验提交：`a4d7a008d7e7597a864624f63f4185d91fada144`；原ResNet定义SHA256见PROTOCOL，未修改模型主体。
- 运行目录：`/home/wang/projects/会议论文/Conference-Paper-ccsae-prototype-diagnostics-v1/outputs/resnet_transfer_v1/server_a4d7a00/`。
- 所有300份initial/best/last权重保留服务器，WEIGHTS_MANIFEST记录路径、大小和SHA256；权重、原始波形及uv环境不进入Git或本轮传输归档。
- 完整非权重归档：`resnet1d-skl-abl-transfer-v1-a4d7a00-server-results.tar.gz`，保留逐步采样、Val/Test分数、bootstrap数组以及文本证据；传输清单逐文件核查SHA256。
- Git仅发布JSON/CSV/MD/PY/LOG文本和运行脚本快照，NPZ保留在完整归档与服务器；脚本快照及原路径对照见SNAPSHOT_MAP，按原路径还原依赖，不直接从快照目录执行。
- 以下是已执行的阶段命令，使用既有环境。在训练提交上恢复脚本、原始Data.zip和之前固定Test数据审计；已完成的目录保持只读，任何新复现需单独输出目录并在开跑前明确更改路径，不能覆盖本轮。脚本预检校验提交与原模型哈希。

```bash
cd /home/wang/projects/会议论文/Conference-Paper-ccsae-prototype-diagnostics-v1
export PYTHONPATH="$PWD/src"
PY=/home/wang/projects/会议论文/analog-circuit-crossdomain/.venv/bin/python
"$PY" server_experiment/resnet_transfer_v1/experiment.py preflight
"$PY" server_experiment/resnet_transfer_v1/experiment.py source
"$PY" server_experiment/resnet_transfer_v1/experiment.py target
"$PY" server_experiment/resnet_transfer_v1/verify.py training
"$PY" server_experiment/resnet_transfer_v1/verify.py test
"$PY" server_experiment/resnet_transfer_v1/verify.py verify-test
"$PY" server_experiment/resnet_transfer_v1/transfer_report.py
```

SERVER_RESULTS.md在自动统计报告后补充源域质量、显著性解释和归档说明；训练脚本、协议和统计JSON保持原字节。
