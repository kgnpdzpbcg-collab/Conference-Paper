# CCSAE 论文核对与复现边界（必读）

来源：T. Gao, J. Yang, S. Jiang, and Y. Li, *An Incipient Fault Diagnosis Method Based on Complex Convolutional Self-Attention Autoencoder for Analog Circuits*, IEEE TIE 71(8):9727–9736, 2024, DOI:10.1109/TIE.2023.3310075。用户上传全文作为实现依据。

## 与论文一致的核心机制

- Fig. 1 / Sec. II-B: 实数脉冲响应作为实部，BN+小卷积构造虚部；复数编码器、特征增强、复数解码器、分类器。
- Eqs. (1)-(2): 复数卷积交叉项 `(Wr*xr-Wi*xi, Wi*xr+Wr*xi)`。
- Eqs. (3)-(4): ACON 分别作用于实部、虚部。
- Eqs. (5)-(8): 2×2 实虚协方差白化 Complex BN，仿射交叉项。
- Eqs. (12)-(13): 复数线性层，解码幅值 `sqrt(real²+imag²)`。
- Fig. 2 / Eqs. (14)-(19): 实数多头 self-attention + k=3 局部卷积，再以复数乘法形式组合实/虚分量。
- Eqs. (21)-(24): 阶段 1 最小化重构损失 + 带标签监督对比损失，阶段 2 微调编码器和分类器的交叉熵。
- Sec. III / Table I: 两套电路 13/11 类，首层卷积核 64/128、首层通道 20/50，后两层核 5、通道 20；特征维度 500/300，SCL 系数 0.0017/0.0059；学习率 0.0022/0.0102；100+200 epochs。

## 无法从原文恢复的实现细节（独立选择，不可声称逐位复现）

- 每一层 stride、padding、解码器中间层与卷积上采样布局、self-attention 的头数、局部分支 shift/aggregation 的具体边界处理、batch size、SGD momentum、特征规范化与 warmup。
- 本实现固定 strides=4/4/2，使用 4-head self-attention，局部深度可分离 k=3 卷积代替未完全公开的 shift/aggregation，实现单层复数 FC 幅值解码，并采用 SGD momentum=0.9。默认 batch_size=64。
- 论文说「编码器＋特征增强」，但 Fig.1 示意并不能确定每一层注意力插入点。本实现只在 CNN 编码结束后应用一次 CSA。
- Table I `feature dimension` 在本文解释为**复数潜在表示的每部分维数**：实部和虚部各 D 维，分类器输入 2D。若原作者实际定义的是拼接后的 D 维，该点需获得作者源码验证。
- 论文 Eq. (22) 印刷表达没有常见的**负号**，与“最小化监督对比并拉近正样本”目标不一致；实现按标准 SupCon **补充负号**。
- 论文 Eq. (13) 重构输出为模长（恒非负），但 Fig.5 有带负号的原始响应。`--reconstruction-target paper_signed` 按纸面 Eq.(21) 与原始输入做 MSE；可选 `--reconstruction-target magnitude` 与 |输入| 做 MSE，属于明确标注的适配实验，而非声称论文方法。
- 论文未披露数据采集批次/group ID，因此按数值一致性分组只能去除**精确重复波形**，不能证明独立测量或消除近似重复。
- 论文采用独立重复 70/30 随机划分并对训练集再划验证；本项目优先采用 `grouped` 70/15/15（整段相同波形不跨集合），结果**不直接与论文报告的99.92% / 98.25%可比**。
- 需要另行检索作者开源代码或联系作者才能称为官方严格复现；本仓库为从论文重建的研究实现。

## 数据风险核查

上传 `Data.zip` 测得两个电路各有 **90** 条官方 Test 正常类波形与官方 Train 完全一致。官方划分不适合直接作为无泄漏泛化测试。CLI 默认采用合并官方 Train/Test 后按**完整波形 SHA256 分组**的分层切分；`--split official` 会因重叠报错，若出于历史比较强制指定 `--allow-official-overlap`，结果必须标注为存在泄漏风险。

## 实验纪律

1. 主实验每个电路都使用相同的 `grouped` manifest/seed/预处理和类别集合，记录处理版本。
2. 训练集拟合标量 mean/std；不进行每样本 zscore 以避免抹除参数漂移幅度。
3. 仅 Train/Val 训练与选模；`--evaluate-test` 是显式开关，最终设计冻结后才运行。
4. 后续消融：普通实数网络、去 CSA、去 SCL；以及 1D-CNN/ResNet1D；保持同一严格划分。无需在 V0 伪造论文数值。
## V1 适配说明（2026-10-10）

- 新增 signed_real 模式：使用原复数 FC 解码器的实部进行有符号波形重构，拟合 Train-only 标准化后的有符号输入，不增加模型参数。
- 保留 V0 的 paper_signed：模长输出恒非负，却拟合有符号波形；这是已确认存在数学不一致的历史对照，V1 不再默认使用。
- 保留 magnitude 模式：模长输出拟合输入的绝对值；是另一种训练适配，不是原论文精确实现。
- 上述修改只修复我们研究实现的训练目标一致性，不能推断原作者真实代码必然使用 signed_real。原论文 Eq.(13) 与有符号输入之间的细节仍需作者源码确认。
- 默认通过 --split-seed 20261009 固定数据划分，与控制模型初始化的 --seed 解耦，避免多随机种子比较混入数据划分差异。
- 每轮 Stage1 history 新增 train_reconstruction_loss、train_scl_loss、reconstruction_loss（验证集）、scl_loss（验证集），保留原 total loss。
- 使用 --feature-epochs 0 可不做 Stage1，直接从随机初始化开始 Stage2 分类训练。该运行是算法选择用的研究对照，不声称复现原论文。
- 原 V0 分支和其服务器实验结果不变。比较 V0、V1、无预训练的因果效果时，应在同一 V1 运行协议下重新运行三组，不能直接把历史测试集数值作为新方法选择依据。
