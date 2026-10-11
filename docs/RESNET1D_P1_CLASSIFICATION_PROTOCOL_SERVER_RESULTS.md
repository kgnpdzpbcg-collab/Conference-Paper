# ResNet1D：对齐P1数据与分类训练协议的服务器结果

**按用户选定方式保留原ResNet1D，从随机初始化开始，不做Stage1。18次训练使用与本轮CCSAE-FC/P1完全一致的Train/Val、标准化与实际Episode顺序，分类阶段均为400次更新。只报告Val，不评估Test。**

## 1. 三种子结果

| 电路 | K | CCSAE-FC Val F1 (%) | CCSAE-Proto P1 Val F1 (%) | ResNet1D Val F1 (%) | ResNet−FC (pp) | ResNet胜出FC/Proto |
|---|---:|---:|---:|---:|---:|---:|
| SKL | 10 | 59.76 ± 9.32 | 65.28 ± 5.74 | 58.59 ± 3.58 | -1.17 ± 12.90 | 1/3 / 0/3 |
| SKL | 20 | 59.74 ± 1.87 | 70.27 ± 3.43 | 61.09 ± 4.31 | 1.36 ± 5.19 | 2/3 / 0/3 |
| SKL | 40 | 61.78 ± 4.97 | 70.79 ± 5.65 | 61.49 ± 4.30 | -0.29 ± 3.06 | 1/3 / 0/3 |
| ABL | 10 | 64.98 ± 8.05 | 70.20 ± 10.42 | 62.25 ± 3.64 | -2.73 ± 6.55 | 1/3 / 0/3 |
| ABL | 20 | 61.94 ± 3.14 | 68.86 ± 7.03 | 61.20 ± 3.79 | -0.75 ± 5.43 | 1/3 / 0/3 |
| ABL | 40 | 62.83 ± 2.76 | 68.14 ± 12.33 | 63.30 ± 0.80 | 0.47 ± 2.79 | 2/3 / 1/3 |

均值 ± 样本标准差（ddof=1，n=3）；差值按相同case/K/seed配对，单位为百分点。FC/Proto成绩只读复用既有服务器结果，没有重新训练。

SKL ResNet的平均Val Macro-F1随K=10/20/40为 58.59% → 61.09% → 61.49%。保留全部三种子波动，不据结果换种子、调学习率或延长训练。

ABL ResNet的平均Val Macro-F1随K=10/20/40为 62.25% → 61.20% → 63.30%。保留全部三种子波动，不据结果换种子、调学习率或延长训练。

## 2. 数据、模型与明确差异

逐条对齐的是冻结P1的数据：SKL13类、ABL11类；K=10/20/40，每类Val5条；split_seed/val_seed=20261009，sample_seed/training_seed=42/43/44。同种子Train嵌套，Val原始记录与顺序固定，归一化仅拟合各K选中的Train。SKL Train总数130/260/520，Val65；ABL Train110/220/440，Val55。18个split_audit与对应CCSAE存档完整相同，归一化张量已回查原始float32波形独立复算。

原ResNet源码SHA256：`b9c96fa1d78ca0c4a1e7a7626a4f5a6a5e29456faa091ff5db4dcd47048e5263`，与原服务器基线及历史少样本快照完全一致。采用ResNet-18式一维基本残差块，四阶段各两块，通道32/64/128/256，全局均值池化后线性分类。只将[N,1000]输入增加为[N,1,1000]，未改卷积、宽度、分类头或新增解码器。

**方法差异：ResNet没有解码器，本轮不加解码器、不执行重构/SupCon、不载入任何CCSAE或旧ResNet预训练权重。CCSAE-FC/Proto沿用10轮Stage1。本对照控制数据和分类阶段，不是完整两阶段训练完全相同，也不是单独隔离Backbone因素的消融。**

同case/seed的ResNet在不同K下随机初始化参数及BN缓冲区一致；ResNet与CCSAE仅使用同一随机种子，网络形状不同，不要求两者初始化权重相同。

## 3. 分类训练冻结配置

| 项目 | 本轮ResNet配置 |
|---|---|
| 初始化 / Stage1 | 随机初始化 / 0轮 |
| 更新预算 | 40轮×10步=400次 |
| Episode | 每类3 Support+2 Query；所有Support在前、Query在后；全部样本参与FC交叉熵 |
| SGD | SKL lr0.0022，ABL lr0.0102；momentum0.9、weight_decay0、clip5 |
| 额外训练机制 | 无调度器、无早停、无增强、无额外BN重估 |
| BN | 使用原ResNet默认运行统计，只由训练Episode更新；Val设eval |
| Val / 选模 | batch64；每轮Val Macro-F1最高，平局最早；固定40轮，无追加 |
| 数值 | 确定性FP32；禁用AMP/TF32与非math SDP |

历史ResNet基线使用AdamW、完整Val、早停和Train-only BN重估。本轮按当前CCSAE-FC分类协议改为上述固定SGD/400步；历史Test成绩不可与本表直接混合。

## 4. 逐种子结果

| 电路 | K | seed | ResNet Val F1 (%) | ResNet最佳轮 | ResNet−FC (pp) | ResNet−Proto (pp) |
|---|---:|---:|---:|---:|---:|---:|
| SKL | 10 | 42 | 62.73 | 35 | +13.73 | -6.13 |
| SKL | 10 | 43 | 56.69 | 25 | -8.36 | -1.97 |
| SKL | 10 | 44 | 56.36 | 30 | -8.87 | -11.96 |
| SKL | 20 | 42 | 56.98 | 34 | -4.45 | -9.92 |
| SKL | 20 | 43 | 60.73 | 22 | +2.99 | -9.42 |
| SKL | 20 | 44 | 65.57 | 40 | +5.53 | -8.20 |
| SKL | 40 | 42 | 60.01 | 36 | +2.94 | -6.03 |
| SKL | 40 | 43 | 66.33 | 23 | -0.64 | -2.96 |
| SKL | 40 | 44 | 58.13 | 10 | -3.16 | -18.91 |
| ABL | 10 | 42 | 61.66 | 30 | -9.90 | -10.39 |
| ABL | 10 | 43 | 66.15 | 30 | -1.22 | -13.41 |
| ABL | 10 | 44 | 58.94 | 21 | +2.93 | -0.04 |
| ABL | 20 | 42 | 63.07 | 22 | +4.66 | -2.60 |
| ABL | 20 | 43 | 63.69 | 29 | -0.70 | -0.30 |
| ABL | 20 | 44 | 56.83 | 14 | -6.20 | -20.09 |
| ABL | 40 | 42 | 63.84 | 27 | +3.37 | +9.51 |
| ABL | 40 | 43 | 62.38 | 25 | +0.22 | -9.62 |
| ABL | 40 | 44 | 63.69 | 24 | -2.19 | -14.40 |

## 5. 执行与核验

在NVIDIA GeForce RTX 5070 Ti上复用 `../analog-circuit-crossdomain/.venv/bin/python`（Python3.12.3，torch 2.14.1+cu130）。四进程、每进程四CPU线程；没有创建环境、安装或下载包。18次训练批次窗口 53.217 秒，平均GPU利用率 89.87%，峰值显存 2146 MiB。窗口含进程初始化、读取和保存，不含审计、独立核验、同步及报告。

服务器新增脚本语法检查通过，18次全部独立核验通过：初始参数由同种子重新构造一致；实际400步Support/Query与对应CCSAE-FC存档逐条相同，独立numpy重放一致，类平衡且互斥；最后权重全部BN计数为400，最佳权重BN计数为最佳轮×10；编码器及分类头均发生更新；40轮历史有限、最早最佳epoch规则正确；最佳权重重载后的Val预测逐条相同，独立混淆矩阵复算指标一致。

全部1100个旧实验/数据/环境文件原字节保持，已安装包版本一致，CCSAE源码固定 `a4d7a008d7e7597a864624f63f4185d91fada144`。新代码位于 `server_experiment/resnet_shot_scale/`，输出独立保存于 `outputs/resnet_shot_scale/server_a4d7a00/`，没有覆盖P0/P1或历史ResNet。

## 6. 解释与复现边界

固定每类5条Val同时用于最佳epoch选择和结果汇总，不能称Test泛化结果；三种子不足以建立统计显著性，且抽样/训练种子耦合。CCSAE有Stage1而ResNet没有，因此结果代表用户选定的同数据/同分类协议对照，不能宣称只因网络结构导致差异。去重不证明独立器件或采集批次泛化，此前数据探索限制继续存在。

SUMMARY.json、PER_SEED_RESULTS.csv记录未四舍五入结果和引用路径；VERIFICATION.json、COMPLETION_VERIFICATION.json、SOURCE_MANIFEST.json记录核验及冻结证据；provenance/保存18个划分、环境、旧文件指纹；runs/保存初始化、最佳和最后权重、40轮曲线、实际Episode和Val预测。

实际服务器命令（本轮归档目标已存在，不可原地重跑覆盖；独立复现需另设空BASE）：

```bash
export PYTHONPATH="$PWD/src"
../analog-circuit-crossdomain/.venv/bin/python server_experiment/resnet_shot_scale/audit.py
../analog-circuit-crossdomain/.venv/bin/python server_experiment/resnet_shot_scale/run.py
../analog-circuit-crossdomain/.venv/bin/python server_experiment/resnet_shot_scale/verify.py
../analog-circuit-crossdomain/.venv/bin/python server_experiment/resnet_shot_scale/report.py
```

封装复用既有common.configure、资源采样和独立混淆矩阵函数，实际源文件快照一并留存。数据、环境、权重与NPZ只保留服务器及本地同步包，GitHub只上传文本证据/代码快照。没有Test推理或超参数搜索。
