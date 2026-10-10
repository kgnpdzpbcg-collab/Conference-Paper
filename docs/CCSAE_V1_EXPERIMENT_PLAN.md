# CCSAE V1：最小修正与对照计划

目标：只解决当前 V0 重构目标的数学不一致，判断其是否影响模型的有效特征学习。最终目的是为普通 EI 会议论文选择有效模块，**不是无限投入严格复现**。

## 比较三个版本

| 版本 | Stage 1 | Stage 2 | 研究目的 |
|---|---|---|---|
| V0-control | 原模长输出（非负）拟合有符号原始波形；100 epoch | 同一编码器+分类器，200 epoch CE | 复现旧实现的矛盾 |
| V1-signed | 复数解码器实部（允许负值）拟合有符号波形；100 epoch | 同一编码器+分类器，200 epoch CE | 检验修正是否有收益 |
| V2-no-pretrain | 跳过 Stage 1 | 从随机初始化直接进行 200 epoch CE | 检验预训练是否必要 |

可选控制：magnitude 模式将模长输出与绝对值目标配对，避免重构目标符号矛盾。V1 的 signed_real 只是研究性适配，非作者官方未公开代码。

## 固定实验协议

- SKL 13 类与 ABL 11 类分别进行，1000 点原始波形保持一致。
- --split grouped --split-seed 20261009 固定精确去重后分层划分；归一化仅拟合 Train。
- --seed 42 / 43 / 44 控制训练随机性，同种子不同变体共用分区。
- V0/V1 均用已存档的 SGD momentum=0.9、Stage1 MSE + lambda*SupCon 及 Stage2 未加权 CE；V2 使用相同 Stage2 配置。编码器和分类器均不更改。
- 只读 Train/Val，用验证 Macro-F1 选择模型，不进行新一轮 Test 预测，避免持续对同一 Test 选模型。
- 原 V0 服务器实验经过环境适配；为了清楚比较修正效果，应在当前 V1 CLI 中以相同参数重新跑 V0/V1/V2。
- 数据集去重只能防止完全相同波形跨集合，并不证明跨独立采集批次或器件泛化。

## 分析重点

1. 比较训练及验证的 Stage1 reconstruction loss、SupCon loss，检查重构目标是否合理。
2. 比较两套电路的验证 Macro-F1、逐类别 Recall/F1、混淆矩阵，以及三个种子同种子配对差值。
3. 若 V1 稳定好于 V0、V2，则考虑保留两阶段预训练；若 V2 不差，会议论文新模型可以简化训练。
4. 若修正后仍无法取得有竞争力的效果，停止在 CCSAE 细节上过度投入，转向新的算法结构设计。

## 可执行命令

~~~bash
pip install -e '.[dev]'
pytest -q

for case in skl abl; do
  for seed in 42 43 44; do
    ccsae-train --data data/Data.zip --case "$case" --seed "$seed" --split-seed 20261009 --output outputs/v1_signed --reconstruction-target signed_real
    ccsae-train --data data/Data.zip --case "$case" --seed "$seed" --split-seed 20261009 --output outputs/v0_control --reconstruction-target paper_signed
    ccsae-train --data data/Data.zip --case "$case" --seed "$seed" --split-seed 20261009 --output outputs/v2_no_pretrain --feature-epochs 0
  done
done
~~~

本计划仅给出运行方式；不代表正式 GPU 实验已经完成。
