# CCSAE for Analog Circuit Incipient Fault Diagnosis — V1 Adaptation

## 最新阶段：冻结 P1 的少样本规模验证

在同一数据池、同种子嵌套 Train（K=10/20/40）与固定5 Val/类上，补跑K10/40共12组配对，复用K20的6组结果；P1温度1、距离不除维度、Backbone/损失/预算冻结。六档Proto均值均高于FC，15/18配对胜出；**ABL均值没有随K增大上升，K40波动较大。只有Val结果，没有Test评估。**

按相同P1数据和分类协议新增18次原ResNet1D对照：ResNet不做Stage1，固定SGD/400步；与FC均值接近，17/18组低于Proto。该比较保留CCSAE有Stage1、ResNet无Stage1的差异，只报告Val。

- [ResNet分类协议对照报告](docs/RESNET1D_P1_CLASSIFICATION_PROTOCOL_SERVER_RESULTS.md)
- [ResNet独立归档及核验](experiments/resnet1d_p1_classification_protocol/README.md)
- [完整规模验证报告](docs/CCSAE_PROTOTYPE_SHOT_SCALE_SERVER_RESULTS.md)
- [独立归档与逐种子证据](experiments/ccsae_prototype_shot_scale/README.md)
- [历史CNN / ResNet1D / CCSAE v1少样本Test成绩](docs/FEWSHOT_BASELINES_CCSAE_V1_SERVER_TEST_RESULTS.md)（不同划分与预算，独立归档）

## 上一阶段：D0 诊断与 P1 距离尺度服务器验证结果

本分支保留 P0 全部历史结果，现已按研究文档完成六组只读 D0 和一次受控 P1。仅将原型距离由除以嵌入维度改为不除以维度；Stage1、FC 和实际样本流均精确复现 P0。P1 Val Macro-F1：SKL **70.27 ± 3.43%**，ABL **68.86 ± 7.03%**（n=3）；对 FC 分别3/3、2/3种子更高。**只评估固定5 Val/类，没有 Test。**

- [完整 D0/P1 服务器结果](docs/CCSAE_PROTOTYPE_DIAGNOSTICS_P1_SERVER_RESULTS.md)
- [新阶段原始证据与复现边界](experiments/ccsae_prototype_scale_p1/README.md)
- [详细负结果分析及诊断执行顺序](docs/research/CCSAE_PROTO_K20_FAILURE_ANALYSIS.md)
- [全部文档索引](docs/README.md)
- [P0 冻结实验报告](docs/CCSAE_PROTOTYPE_K20_SERVER_RESULTS.md)

**新命令（不需要重新训练，但需本地保留 shared_stage1_last.pt）：**

~~~bash
python -m pip install -e '.[dev]'
pytest -q
ccsae-proto-diagnose --data data/Data.zip --run-dir outputs/server_proto_k20_7ef4042/skl/k20_sample42_train42 --device auto
~~~

**P1 单因素重训复现入口（服务器实测已完成；只有显式指定 none 才修改）：**

~~~bash
ccsae-fewshot --data data/Data.zip --case skl --k 20 --seed 42 --distance-normalization none --output outputs/proto_scale_p1
~~~

新 P1 运行保存到带 _distnone 后缀的独立目录，不覆盖 P0；任何新实验都先用 Val 比较，默认没有 Test 推理。诊断详细参数和结果判据见上面的 MD。

---

## 当前开发分支：CCSAE＋Prototype（K=20 少标注实验）

此分支从 **reproduction/ccsae-v1** 继承，保留 V1 signed_real 有符号重构，新增 **FC 与 Proto 的配对验证训练**；原有 ccsae-train 命令仍可用。

**完整算法、数据去重差异、研究假设、数学公式和实验纪律：** [docs/CCSAE_PROTOTYPE_FEWSHOT_V1.md](docs/CCSAE_PROTOTYPE_FEWSHOT_V1.md)。

### 最短运行

```bash
python -m pip install -e '.[dev]'
pytest -q
# 将 Data.zip 放到 data/Data.zip，不要提交到 GitHub
ccsae-fewshot --case skl --data data/Data.zip --k 20 --seed 42 --stage1-epochs 0 --epochs 1 --steps-per-epoch 1 --tiny --output outputs/smoke
ccsae-fewshot --case skl --data data/Data.zip --k 20 --seed 42 --stage1-epochs 10 --epochs 40 --steps-per-epoch 10
ccsae-fewshot --case abl --data data/Data.zip --k 20 --seed 42 --stage1-epochs 10 --epochs 40 --steps-per-epoch 10
```

默认 mode=both：同 K-shot Train、相同 Episode 样本和优化步数、**共享 Stage 1 权重**下比较原 FC 与新的 Prototype Head；只输出固定 **5 Val 标签/类**的指标，**本入口没有 Test 评估**。正式对照应再运行 seed 43、44 并比较配对差值。

**数据处理的重要区别**：这里按波形 SHA256 真正去重，SKL 3800、ABL 3100 条唯一波形；原 V1 data.py 的 grouped 模式只分组但保留重复行。服务器核查已确认新旧逐条划分不同，不要直接混用历史成绩。

### 最新服务器结果（2026-10-11）

已在 RTX 5070 Ti 上完成 K=20、种子 42/43/44 的 6 组 FC/Proto 配对实验，复用已有 uv 环境，4 进程并发，正式训练批次耗时 52.907 秒。上游模型与训练算法保持提交 `7ef4042` 的原实现。

| 电路 | FC Val Macro-F1 (%) | Proto Val Macro-F1 (%) | Proto−FC (pp) |
|---|---:|---:|---:|
| SKL | 59.74 ± 1.87 | 46.99 ± 3.10 | −12.74 ± 4.96 |
| ABL | 61.94 ± 3.14 | 42.98 ± 1.21 | −18.96 ± 3.09 |

均值 ± 样本标准差，n=3。**全部 6 个配对种子中 Proto 均低于 FC，本轮不支持原型头的收益假设。仅评估固定 5 Val/类，没有评估 Test。** 14 项单元测试及正式权重、预测和指标独立核验通过。数据核查已确认：新旧唯一波形集合一致，但逐条划分不同，因此不能直接混用历史成绩。

[完整服务器报告](docs/CCSAE_PROTOTYPE_K20_SERVER_RESULTS.md) · [逐种子结果与核验证据](experiments/ccsae_prototype_k20/README.md)

---

**状态：V1 独立研究适配，修正重构目标与输出符号不匹配的问题，并将数据划分种子与训练种子分离。不是原作者官方源码，也不声称严格复现论文准确率。**

论文：Gao et al., *IEEE Transactions on Industrial Electronics*, 2024, [DOI: 10.1109/TIE.2023.3310075](https://doi.org/10.1109/TIE.2023.3310075)。

## 工程包含什么

- 三层复数卷积编码器（Complex Conv + 2×2 Complex BN + ACON）与实数输入的虚部构造。
- 复数自注意力（MHA＋轻量局部卷积的复数扩展）；复数全连接幅值解码器；分类头。
- **Stage 1**：重构 MSE + 正确符号的监督对比损失（默认 100 epoch）。
- **Stage 2**：编码器与分类头联合微调，交叉熵（默认 200 epoch）。
- SKL 13类与 ABL 11类独立运行；精确重复波形分组切分；验证最佳 checkpoint、日志和原始数据审计报告。

> 与原文逐层未公开细节的差异、Eq.(22) 符号问题和 Eq.(13) 有符号重构歧义，见 [docs/REPRODUCTION_NOTES.md](docs/REPRODUCTION_NOTES.md)。

## 安装

```bash
# Python 3.10+，建议独立环境
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux: source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

将从 IEEE DataPort 获得的 `Data.zip` 放在 `data/Data.zip`；它应包含 `SKLCircuit_INCI_*` 和 `ABLCircuit_INCI_*` 的共 8 个 CSV。**不要上传数据到 GitHub**。

## 运行

```bash
# 推荐：去重分组后的 SKL 两阶段训练，只调验证集
ccsae-train --data data/Data.zip --case skl --output outputs --seed 42

# ABL：自动采用论文 Table I 对应超参数
ccsae-train --data data/Data.zip --case abl --output outputs --seed 42

# 仅检查训练链路（缩小结构，非论文配置）
ccsae-train --data data/Data.zip --case skl --feature-epochs 1 --classification-epochs 1 --tiny --batch-size 16

# 在所有架构和超参数固定之后，才启用正式 Test
ccsae-train --data data/Data.zip --case skl --output outputs --evaluate-test
```

V1 默认采用 `signed_real`：使用原复数解码器的实部拟合有符号波形；`paper_signed` 保留 V0 中的非负输出/有符号目标矛盾，仅作为历史对照；`magnitude` 是将模长输出与绝对值目标配对的另一适配方案。这些选择都不能视为对原作者未公开实现的确认。

## 输出

`outputs/<case>/seed_<seed>/`: `stage1_best.pt`、`stage2_best.pt`、`history.json`、`summary.json`、`data_audit.json`。默认不运行 Test。

## 数据集划分说明

上传的数据中，官方 Train/Test 的正常类存在完全相同的波形：SKL 90 条、ABL 90 条。本工程默认合并后按 SHA256 完整波形分组、分层划分约 70/15/15，防止精确重复样本跨 Train/Val/Test；这并不意味着独立采集批次评估，也不能与作者原始数值直接比较。

## 后续实验路线

1. 完整训练两套电路并核验 Stage 1/2 曲线和独立测试集。
2. 在相同去重划分上跑普通 1D-CNN、ResNet1D，与 CCSAE 比较。
3. 按原文 Fig. 9：去复数结构、去 CSA、去 SCL 三组消融。
4. 仅在复现可信后研究额外模块，不在复现阶段提前改造核心结构。
## V1 对照实验（只用 Train / Val 选模）

固定数据分组划分种子 `--split-seed 20261009`，用训练种子 42、43、44 重复实验。默认不预测 Test。阶段一日志新增 train/val reconstruction_loss 与 scl_loss。

```bash
# V1：有符号解码 + 有符号输入（推荐主实验）
ccsae-train --data data/Data.zip --case skl --seed 42 --split-seed 20261009 --output outputs/v1_signed --reconstruction-target signed_real
# V0：原非负模长解码 + 有符号输入（仅用于历史对照）
ccsae-train --data data/Data.zip --case skl --seed 42 --split-seed 20261009 --output outputs/v0_control --reconstruction-target paper_signed
# V2：跳过 Stage1，从随机初始化直接训练相同编码器与分类头
ccsae-train --data data/Data.zip --case skl --seed 42 --split-seed 20261009 --output outputs/no_pretrain --feature-epochs 0
```

将 `skl` 换为 `abl`，分别运行三个种子。完整计划见 [CCSAE_V1_EXPERIMENT_PLAN.md](docs/CCSAE_V1_EXPERIMENT_PLAN.md)。**本版本只完成代码修正与测试，尚无 V1 正式性能结果。**
