# CCSAE v1 FC/Prototype K=20 服务器实验结果

本轮结论：在最新分支固定的 P0 快速验证协议下，SKL 和 ABL 的三个配对种子均为 FC 优于 Prototype，原型模块的性能增益假设未得到支持。这是验证集结果，没有评估 Test。

## 核心结果

所有指标单位为 %，均值 ± 样本标准差，n=3；差值为同种子 Proto−FC，单位为百分点。

| 电路 | FC Val Macro-F1 | Proto Val Macro-F1 | 配对差值 | FC Val Acc | Proto Val Acc |
|---|---:|---:|---:|---:|---:|
| SKL | 59.74 ± 1.87 | 46.99 ± 3.10 | -12.74 ± 4.96 | 62.56 ± 0.89 | 48.72 ± 3.20 |
| ABL | 61.94 ± 3.14 | 42.98 ± 1.21 | -18.96 ± 3.09 | 66.06 ± 2.10 | 44.85 ± 2.78 |

验证集每类恰好 5 条，因此本轮 Balanced Accuracy 与 Accuracy 相同。三次重复同时改变采样种子和训练种子；两方法同种子严格配对。

## 各种子完整结果

| 电路 | 种子 | FC Val F1 (%) | Proto Val F1 (%) | Proto−FC (pp) | FC 最佳轮 | Proto 最佳轮 |
|---|---:|---:|---:|---:|---:|---:|
| SKL | 42 | 61.43 | 44.14 | -17.29 | 39 | 2 |
| SKL | 43 | 57.74 | 50.29 | -7.45 | 35 | 1 |
| SKL | 44 | 60.04 | 46.55 | -13.49 | 33 | 40 |
| ABL | 42 | 58.40 | 42.95 | -15.45 | 22 | 32 |
| ABL | 43 | 64.39 | 44.20 | -20.19 | 38 | 36 |
| ABL | 44 | 63.03 | 41.79 | -21.24 | 31 | 38 |

## 训练协议与环境

- 仓库分支 `research/ccsae-prototype-k20`，提交 `7ef4042fe18275f2f366c06c5aa19aa817c45258`。上游 `src/ccsae` 和单元测试文件保持原字节；新增服务器运行、记录和独立核验脚本。
- 实际复用 `/home/wang/projects/会议论文/analog-circuit-crossdomain/.venv/bin/python`，通过显式 PYTHONPATH 载入本轮独立 src，没有向旧环境安装或更新包。Python 3.12.3、PyTorch 2.14.1+cu130、NumPy 2.5.3、scikit-learn 1.9.1、CUDA 13.0、cuDNN 92400。
- 准备阶段曾从 uv 缓存创建本轮 .venv，未重下载 PyTorch；按用户要求，预检及全部正式训练使用上述既有环境，新 .venv 没有用于正式训练。
- RTX 5070 Ti 16 GB。FP32，确定性算法，关闭 AMP/TF32/CuDNN benchmark 和 Flash/memory-efficient/cuDNN SDP，仅启用 math SDP。4 个进程，每个进程 4 个 CPU 线程；这是经本轮预检验证的并发配置，没有声称穷举得到最优并发数。
- K=20/类；SKL 260 Train + 65 Val，ABL 220 Train + 55 Val。真实标签预算是每类 20 条训练标签加额外 5 条验证标签。
- 共享 Stage 1：10 epochs，SGD momentum=0.9，batch=64，signed_real 重构 + SupCon，最后一轮权重作为两路共同初始化。SKL LR=0.0022、SCL=0.0017；ABL LR=0.0102、SCL=0.0059。
- Stage 2：两路各 40 epochs × 10 steps=400 次更新，SGD momentum=0.9、clip=5，沿用上述 LR。每类 3 Support + 2 Query，SKL 每步 65 条，ABL 每步 55 条。
- FC 对 Support+Query 做 CE；Proto 用有梯度的 Support 均值构造原型，对 Query 做距离 CE；平方欧氏距离除以嵌入维度，tau=1。两路损失监督位置不同，这是方法定义的差异，不能称所有训练目标完全相同。
- 只从选定 K-shot Train 拟合均值/标准差；验证原型使用该 K-shot Train 全部样本；按固定小 Val Macro-F1 选最佳轮，平局取最早轮。没有早停和超参搜索，没有 Test 推理。

## 数据划分与历史结果关系

新分支从原始 Data.zip 合并数据，以 float32 波形 SHA256 真正去重后重新生成 70/15/15 池。唯一波形集合与旧数据一致、没有跨池精确重复，但逐条划分不同。相同 split_seed 并不保证不同实现获得相同划分。

| 电路 | 新 Train Pool | 新 Val Pool | 新 Test Pool | 新旧 Train 重合 | 新旧 Val 重合 | 新旧 Test 重合 |
|---|---:|---:|---:|---:|---:|---:|
| SKL | 2660 | 570 | 570 | 1847 | 76 | 74 |
| ABL | 2170 | 465 | 465 | 1521 | 72 | 73 |

SKL 新训练池含旧 Test 中的 402 条波形，ABL 含 327 条；本轮每类 20 条是从新训练池抽取。新 Test 从未在本轮推理，但同一数据集此前已参与其他实验探索，不能把本轮重新划分描述为全新、从未见过的最终评测。完整交集矩阵和样本指纹见 data_audit.json。

本轮只有 5 Val/类，旧少样本实验使用完整 Val，且训练预算也不同，因此不能将本轮 FC/Proto 成绩与历史 CCSAE、CNN、ResNet 成绩直接并列比较。需要正式比较时，应冻结相同划分、标签预算和训练协议后重新训练。去重不能证明独立器件或采集批次评估。

## 运行时间与核验

- 正式训练批次窗口：北京时间 2026-10-11 14:24:12 至 2026-10-11 14:25:05，耗时 52.907 秒。只计 6 组配对训练批次（含子进程启动、加载和保存），不含代码准备、环境检查、预检、后续核验和同步；各组耗时并不能相加视为实测串行时间。
- 窗口内 GPU 平均使用率 92.15%，峰值显存 2374 MiB（每秒采样）。
- 在复用的环境中 14 个上游单元测试通过；4 组完整结构 GPU 预检通过；6 组正式配对独立核验通过。核验实际 Episode 索引严格相同、共享初始化完全一致、Stage 2 decoder 冻结、Proto classifier 冻结、最佳轮规则正确，重新加载权重的验证预测与原记录完全一致，混淆矩阵独立复算指标一致。
- 数值检查发现短预检中 4/240 条验证记录的近并列原型距离会受 float32 cdist 舍入影响；最大分数差约 7.9e−8。保留上游实现，未据此改变训练算法。正式最佳权重在全部 360 条验证记录上，cdist、直接平方距离和 FP64 距离三种计算的预测完全一致。正式最大分数差约 1.33e−5，已按嵌入维度与特征范数核验舍入预算；FP64 是固定 checkpoint 诊断，没有重新选模。
- 原去重数据及之前少样本、CCSAE v1、CNN/ResNet 和两类跨域实验的全部记录文件 SHA256 前后完全一致。训练快照保留实际运行源码，verification_snapshot 保存修正后的最终核验脚本。

## 结论的使用范围

该固定快速协议下，两套电路均未出现原型分类收益，暂不支持将普通原型头写成已验证的性能贡献。SKL 两个原型模型很早达到最佳验证轮，训练预算和表征变化值得后续诊断，但本轮没有额外调参或追加实验。三次重复和小验证集不足以证明所有原型方法、温度或更长训练预算都会失败，也不能充当最终论文 Test 性能。

## Material Passport / 证据索引

- 固定协议：上游 docs/CCSAE_PROTOTYPE_FEWSHOT_V1.md；核心实现 src/ccsae/{fewshot_data,fewshot_train,prototype}.py。
- 数据来源：用户提供 Data.zip，SHA256 `11623819dcbb9287ff7ffea4d763d0a6fe3dd77d74c61bfe581bd8ea792b4e49`；没有外部补充数据。
- 每种子 config.json、split_audit.json、shared_stage1_history.json、fc_history.json、proto_history.json、summary.json、train.log；共享和两路最佳权重保留在服务器。
- episodes.npz 与 pair_verification.json 记录实际索引流和初始化；verification.json、*_verified_val.npz 保存独立重载、逐类别混淆及预测证据。
- SUMMARY.json、PER_SEED_RESULTS.csv、timing.json、resource_usage.json；server_experiment/provenance 内保存数据、环境、单元测试、数值诊断和旧结果指纹。
- 本报告数字完全来自服务器实测，没有本地训练数字、推测 Test 成绩或挑选种子。
