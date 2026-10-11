# CCSAE + Prototype：有限标注早期模拟电路故障诊断（方案 V1 / P0）

> 2026-10-11；状态：代码实现，已完成服务器 P0 快速验证，仅有 Val 结果，没有 Test 评估。目标是形成普通 EI 会议论文，而非严格复现 CCSAE。**本轮唯一结构改动是把 Stage 2 的 FC+CE 替换为 episodic Prototype+Distance CE**；暂不增加加权原型、额外注意力、Mamba、其他 backbone 或额外损失。服务器结果见 [实验报告](CCSAE_PROTOTYPE_K20_SERVER_RESULTS.md)。

## 1. 研究问题与论文故事

原论文：Gao et al., An Incipient Fault Diagnosis Method Based on Complex Convolutional Self-Attention Autoencoder for Analog Circuits, IEEE Transactions on Industrial Electronics, 2024, DOI 10.1109/TIE.2023.3310075。

原 CCSAE 用复数卷积编码器（A）和复数自注意力（B），结合第一阶段重构+监督对比、第二阶段线性分类，在其每类约 210 个训练样本、随机 70/30 同分布闭集条件下取得 SKL 99.92%、ABL 98.25% Accuracy；这不能证明每类仅有 10–40 个标注样本时仍有同样表现。

**本研究问题**：当模拟电路各类早期故障只有少量已标注波形时，类别判别边界的学习可能不稳定。能否利用标注训练波形构造类别原型，使 CCSAE 表示更适合低标注分类？

**研究假设 H1**：当 K=20 时，基于原型的 Stage 2，在同一编码器与相同样本和更新预算条件下，比普通 FC 分类取得更高或更稳的验证 Macro-F1。**本轮固定 P0 协议的服务器结果不支持该收益假设：两套电路各三个种子中 Proto 均低于 FC，不能写成已取得的性能贡献。该结果不推断其他训练设置的表现。**

论文定位是 limited-label closed-set fault classification：SKL 全部 13 类、ABL 全部 11 类均在训练中可见，不是元学习测试未见类别，也不涉及跨域/新故障零样本识别。

经典方法出处：Snell et al., Prototypical Networks for Few-Shot Learning, NeurIPS 2017。故障诊断参考：Certainty and Transferability Guided Few-Shot Open-Set Cross-Domain Fault Diagnosis, IEEE TII 2025, DOI 10.1109/TII.2024.3514213。后者作者开源 AnYiyao-cqu/CTFDA 中的 Support 原型是普通均值；本实验并未照搬其跨域/开放集模块，也未实现 weighted prototype。Prototype 和 episodic training 均不是本文首次提出的算法。

## 2. 网络：A+B+C，控制一个变量

~~~text
输入：x [B,1000] 单通道波形
       |
   实部=x；虚部=BN+Conv(x)
       |
A  三层复数卷积 + Complex BN + ACON
       |
B  Complex Self-Attention (CSA)
       |
   ComplexLinear --> concat(real,imag) --> z
                                      SKL D=1000
                                      ABL D=600
       |
       +------------------------+
       |                        |
Control: FC Head        Proposed: Prototype Head
 Linear(D,C)            Support 特征按类均值 p_c
 CE(all batch)          Query 与 p_c 距离
       |                CE(Query)
       +------------------------+
       | 同一个 Stage 1 权重；相同 Episode 数据流 |
~~~

V1 编码器及 CSA 保留原研究实现。这里的复数虚部是 BN+Conv 学得的表示，不能称为物理 FFT 相位。原作者未公开的结构细节与我们的独立实现差异见 docs/REPRODUCTION_NOTES.md。

### 2.1 共享第一阶段

Stage 1 采用 CCSAE V1 修复的 signed_real 重构：解码器实部可表示负值，与有符号输入做 MSE；辅助损失为 SupCon：

\[
L_1=L_{\rm recon}+\lambda_s L_{\rm SupCon}.
\]

**仅用当前 K-shot Train 的标签训练 Stage 1**，禁止使用未选入 K-shot 的全部 Train 标签做 SupCon 后再宣称 K-shot。阶段一训练完的同一权重传给 FC 和 Proto；P0 简化为固定 epoch，使用最后一轮 checkpoint；不使用 Val 选 Stage 1 以减少调参变量。支持 stage1=0 从相同随机初始化直接对照，但默认 Stage 1 为 10 epochs，并非原论文的 100 epochs 完整训练。

### 2.2 第二阶段原型模块

CCSAE 编码 f_theta；Support 集 S_c 的原型：

\[
p_c = |S_c|^{-1}\sum_{x_i \in S_c} f_\theta(x_i).
\]

Query z_q 的距离分类 logit：

\[
\ell_{q,c}=-\frac{\|z_q-p_c\|_2^2}{D\tau},
\quad L_{\rm proto}=\mathrm{CE}(\ell_q,y_q).
\]

D 为嵌入维度 1000/600，除以 D 用来控制高维平方距离的数值尺度，是本实现的选择；tau 默认 1.0 固定。Support 和 Query 特征**都保留梯度**。同一次前向的 Support 求类别中心，Query 求距离并训练。推理阶段类别原型仅使用当前 K-shot 的全部 K 条/类有标签 Train 样本，不使用验证或测试数据构造原型。

Stage 2 不再另外加 SupCon、Center、Triplet 等损失；否则无法区分新 Head 的收益。允许以后对欧氏/余弦距离做小规模验证诊断，但 P0 不扫参数。Proto 和 FC 的参数量不同，不能称训练目标完全等价。

## 3. 数据处理和标注预算

IEEE DataPort Incipient Fault Dataset for Analog Circuits, DOI 10.21227/gf4r-5q68；只用 SKL/ABL，每条 1000 点。

| 数据集 | 原始行 | 唯一波形 | Train Pool | Val Pool | Test Pool |
|---|---:|---:|---:|---:|---:|
| SKL（13 类） | 3900 | 3800 | 2660 | 570 | 570 |
| ABL（11 类） | 3300 | 3100 | 2170 | 465 | 465 |

**重要数据修正**：V1 原始 data.py 的 grouped 模式是把完全相同的波形放同一集合，实际仍保留所有重复行。本分支的 fewshot_data.py 是按 float32 波形 SHA256 **真正每个哈希保留一条**。旧 V1 文件不修改。

固定协议：

1. 合并官方 Train/Test，检查相同波形是否有冲突标签，按出现次序去重。
2. 各类别内按 split_seed=20261009 分层 70/15/15；各池完全相同的波形交集必须为空。
3. 只从 Train Pool 抽每类 K 条，sample_seed 单独控制；只从 Val Pool 抽每类 5 条，val_seed 固定为 20261009。
4. P0 的 K=20：SKL Train 260 / Val 65；ABL Train 220 / Val 55。**标注预算是 K 条训练+额外 5 条验证/类，而非总共 K 条标签。**
5. 标准化均值/标准差仅从**已选 K-shot Train**拟合；不使用完整 Train Pool 的统计量、不对每条波形独立 z-score。
6. sample_seed 固定时，K=10、20、40 使用同一类别随机排列的前 K 个样本，方便后续学习曲线比较。
7. split_audit.json 保存各池指纹、选中的 Train/Val 波形 SHA256、随机种子、标准化参数、重复数量。
8. Test 仅用于预先固定分层的标签比例，不参与训练、验证、原型估计或当前脚本的预测。

去重只能去掉精确重复，不能证明采集批次/器件独立。**旧服务器实验虽然报告同样的 3800/3100 数量，但没有将完整 split manifest 提交到这个仓库；本分支的 split_seed 及算法不保证和旧服务器逐条索引一致，不能直接把两个实验当成完全相同协议的比较。**

## 4. FC 与 Proto 如何公平对照

每一步用相同 episode 索引：SKL 13-way，3 Support/类+2 Query/类，总 65；ABL 11-way 总 55。Support 与 Query 当步不重叠，每步重新抽样。

- 同一 case/K/seed 的 FC、Proto 共享 Stage 1 权重和编码器结构。
- 两者在 Stage 2 接收**完全相同的 65/55 条样本**、相同次序、相同 BN batch、同优化器、学习率、梯度裁剪和优化更新次数。
- FC 对当前 batch 全部 Support+Query 做 CE；Proto 用 Support 计算原型、Query 做 CE。这是两种方法必要的损失定义差异，不声称监督位置也完全一样。
- 两者都在固定 5/类 Val 上按 Macro-F1 选最佳 checkpoint，并列选择更早 epoch。
- 验证 Proto 时用全部 K-shot Train 计算原型，Val 只作为 Query；Val 标签不参与原型。

P0 固定设置：

| 参数 | 值 |
|---|---|
| K | 20 / 类 |
| Support/Query | 3 / 2 / 类 |
| Stage 1 | 10 epochs，signed_real+SupCon，共享 |
| Stage 2 | 40 epochs x 10 steps/epoch |
| Optimizer | SGD, momentum=0.9 |
| LR | SKL 0.0022；ABL 0.0102 |
| SCL 系数 | SKL 0.0017；ABL 0.0059 |
| 距离与温度 | 欧氏距离平方 / D；tau=1.0 |
| 种子 | 42、43、44；同种子配对 |
| 指标 | Val Macro-F1，辅助 Acc/Balanced Acc |
| 测试集 | 不做推理，不报告 Test |

这是**快速验证**的训练预算，不能直接充当论文最终性能。首轮建议比较三个配对差值（Proto Val F1 - FC Val F1），看两套电路是否存在稳定增益。

## 5. 本地操作

从已建立的新分支获取代码：

~~~bash
git clone -b research/ccsae-prototype-k20 https://github.com/kgnpdzpbcg-collab/Conference-Paper.git
cd Conference-Paper
python -m pip install -e '.[dev]'
pytest -q
~~~

将数据包放到 data/Data.zip，勿上传到 GitHub。数据和 outputs/ 已被 .gitignore 排除。

**第一步：烟雾测试**（只检查代码，非正式结果）：

~~~bash
ccsae-fewshot --data data/Data.zip --case skl --k 20 --seed 42 --stage1-epochs 0 --epochs 1 --steps-per-epoch 1 --tiny --output outputs/smoke
~~~

**第二步：两套电路、3 seed 的快速实验**（默认 --mode both）：

~~~bash
for case in skl abl; do
  for seed in 42 43 44; do
    ccsae-fewshot --data data/Data.zip --case "$case" --k 20 --seed "$seed" --sample-seed "$seed" --stage1-epochs 10 --epochs 40 --steps-per-epoch 10 --output outputs/fewshot_proto
  done
done
~~~

Windows PowerShell（单行循环，无反引号续行）：

~~~powershell
foreach ($case in @("skl","abl")) { foreach ($seed in @(42,43,44)) { ccsae-fewshot --data data/Data.zip --case $case --k 20 --seed $seed --sample-seed $seed --stage1-epochs 10 --epochs 40 --steps-per-epoch 10 --output outputs/fewshot_proto } }
~~~

日志位置：outputs/fewshot_proto/<case>/k20_sample<seed>_train<seed>/。保存 config.json、split_audit.json、shared_stage1_history.json、shared_stage1_last.pt（当 stage1 > 0）、fc_best.pt、proto_best.pt、两个 history.json、两个 val_predictions.json 与 summary.json。summary.json 内明确 test_evaluated=false。本版本故意**没有提供 Test 评估 CLI**，先在 Val 上完成算法决策。

## 6. 方法门槛和论文实验

**Gate 0 数据和代码**：pytest 通过、3800/3100 唯一样本、正确 Train/Val 数量与无重复哈希、Support/Query 不重合、Proto 的梯度能回传 Support/Query。

**Gate 1 K=20 核心对照**：SKL 和 ABL 各 3 个采样/训练种子；对比 FC 和普通 Proto 的 Macro-F1、逐类别、配对差值；如果 Proto 无稳定改进，不直接堆 weighted prototype。

**Gate 2 诊断与有限创新**：只有发现普通原型有实际收益且原型估计随 Support 变化不稳，才考虑研究类内一致性加权原型。加权原型需与普通均值原型独立消融，本版本未实现。

**Gate 3 正式论文实验**：冻结方法后 K=10/20/40，5 个配对随机重复，并在 Full 条件下作参照；同协议重训 CNN1D、ResNet1D、CCSAE-FC、CCSAE-Proto；报告 Macro-F1 Mean±Std、Accuracy、Balanced Accuracy、配对差值、困难类混淆、必要的模型大小与时间。

**Gate 4 最终评估**：模型结构、数据处理、选模规则全部冻结后再启用固定 Test；披露之前该数据集测试集已参与探索的限制。

## 7. 失败诊断

- Stage 1 loss 高：先检查 signed_real 重构、数据数值尺度、SCL 的正样本数量。
- Proto 接近随机：先检查原型类别映射、距离范围、温度、梯度、BatchNorm 及全 K 验证原型；不直接添加大模型。
- FC 更好：可能原型单中心假设不成立、当前嵌入分布不理想、两种损失优化不同；保持负结果，不夸大。
- 3 个 seed 波动大：扩大 paired repeats，不能只报最好的一次。
- 旧 Full ResNet 更高：K-shot 的可用标签数完全不同，不能直接拿旧 Full 数值下结论。

## 8. 实现文件

- src/ccsae/fewshot_data.py：去重、固定划分、K-shot、标准化、数据审计。
- src/ccsae/prototype.py：episode、类别原型、距离 logits、指标。
- src/ccsae/fewshot_train.py：共享 Stage 1，配对 FC/Proto 训练、验证与保存。
- tests/test_fewshot.py：采样独立性、K 子集嵌套、原型梯度等。
- pyproject.toml：ccsae-fewshot 命令。

**本文件记录算法设计和可复现实验协议；已获得的 P0 服务器验证结果单独保存在 [实验报告](CCSAE_PROTOTYPE_K20_SERVER_RESULTS.md)，没有最终 Test 性能。**
