# CCSAE for Analog Circuit Incipient Fault Diagnosis — Independent Reproduction

**状态：V0 论文核心结构与两阶段训练的独立复现代码；不是作者官方源码，也没有声称复现论文99.92%/98.25%的结果。**

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

训练配置默认优先复现原文，使用 `paper_signed` 重构目标；注意非负重构值与原波形负值不一致。`--reconstruction-target magnitude` 为可单独实验的有明确标注的替代选项。

## 输出

`outputs/<case>/seed_<seed>/`: `stage1_best.pt`、`stage2_best.pt`、`history.json`、`summary.json`、`data_audit.json`。默认不运行 Test。

## 数据集划分说明

上传的数据中，官方 Train/Test 的正常类存在完全相同的波形：SKL 90 条、ABL 90 条。本工程默认合并后按 SHA256 完整波形分组、分层划分约 70/15/15，防止精确重复样本跨 Train/Val/Test；这并不意味着独立采集批次评估，也不能与作者原始数值直接比较。

## 后续实验路线

1. 完整训练两套电路并核验 Stage 1/2 曲线和独立测试集。
2. 在相同去重划分上跑普通 1D-CNN、ResNet1D，与 CCSAE 比较。
3. 按原文 Fig. 9：去复数结构、去 CSA、去 SCL 三组消融。
4. 仅在复现可信后研究额外模块，不在复现阶段提前改造核心结构。