# 当前固定Test最终评估：CCSAE-Proto、CCSAE-FC、ResNet1D

仅加载此前由Val Macro-F1选出的最佳权重，完成两电路、K=10/20/40、种子42/43/44的54次固定Test评估。未训练、调参、Test选权重或模型重选。

## 最终Test Macro-F1

单位：%，3种子的均值 ± 样本标准差（ddof=1）。Test固定，种子改变Train抽样和训练。

| 电路 | 模型 | K=10 | K=20 | K=40 |
|---|---|---:|---:|---:|
| SKL | CCSAE-Proto (P1) | 57.71 ± 1.51 | 62.66 ± 2.14 | 61.42 ± 2.34 |
| SKL | CCSAE-FC | 50.69 ± 5.26 | 52.66 ± 4.13 | 54.65 ± 2.14 |
| SKL | ResNet1D | 56.81 ± 2.25 | 58.64 ± 3.73 | 54.42 ± 7.52 |
| ABL | CCSAE-Proto (P1) | 66.89 ± 5.34 | 67.91 ± 9.56 | 70.88 ± 9.33 |
| ABL | CCSAE-FC | 60.61 ± 5.83 | 58.57 ± 2.98 | 59.53 ± 7.11 |
| ABL | ResNet1D | 55.06 ± 2.18 | 60.25 ± 4.88 | 59.98 ± 7.49 |

## 结果解读

Proto在六个电路/K设置的Test均值均最高，相比FC的18个种子配对全部提高；提升均值为6.27–11.35个百分点。相比ResNet为14/18个种子配对更高，六档均值差为0.90–11.82个百分点。SKL K=10的差值仅0.90 ± 3.54个百分点，不能据三种子均值宣称稳定显著优势。

SKL Proto从K=10到20提高，但K=40的61.42%低于K=20的62.66%，本轮不支持“标注增加必然单调提高”。ABL Proto均值随K提高，但K=20/40的标准差约9个百分点，存在明显种子波动。这些是固定候选的最终测试结果，不用于据Test结果改变模型、权重或超参数。

## 冻结协议与边界

- 当前P1：原始Train/Test合并后，按float32整行指纹首出现去重，再按类70%/15%/15%划分，split_seed=20261009。不是历史Data_deduplicated测试集。
- 使用完整固定Test：SKL 570条（标签0为30条，其余12类各45条）；ABL 465条（标签0为15条，其余10类各45条），未抽成5条/类。
- 每组使用已保存K条/类Train与5条/类Val。标准化统计量只从该组Train拟合并冻结；三模型使用同一原始Test及同一组Train统计量，Test不参与拟合。
- 最佳模型由40轮Stage2的最高Val Macro-F1选择，同分保留最早轮次；推理前54份权重指纹均与上轮独立核验记录相同。
- Proto固定P1欧氏平方距离、温度1、不除嵌入维度；类原型仅由全部已标注K条/类Train的eval特征均值构建，不用Test拟合。
- eval、批大小64、确定性FP32、关闭TF32，无增强或BN重新估计。沿用服务器原uv环境，4进程×4CPU线程执行推理。
- CCSAE-FC/Proto有10轮重构+SupCon的Stage1，ResNet1D随机初始化、Stage1为0；分类阶段对齐相同Episode和400次SGD更新。这是已选定基线协议，不应作单独骨干差异的因果解释。
- 当前模型Train/Val/Test波形指纹互斥，但数据集和部分波形已用于历史实验，不能称全新外部盲测；也缺少独立采集批次/电路板标识，整行去重不能证明物理采集独立性。

## 配对差值

同电路、K、种子配对；单位：百分点，均值 ± 样本标准差。

| 电路 | K | Proto − FC | Proto − ResNet1D |
|---|---:|---:|---:|
| SKL | 10 | +7.02 ± 4.57（胜3/3） | +0.90 ± 3.54（胜2/3） |
| SKL | 20 | +10.00 ± 3.59（胜3/3） | +4.02 ± 5.48（胜2/3） |
| SKL | 40 | +6.76 ± 4.12（胜3/3） | +6.99 ± 9.64（胜2/3） |
| ABL | 10 | +6.27 ± 3.09（胜3/3） | +11.82 ± 3.17（胜3/3） |
| ABL | 20 | +9.34 ± 6.60（胜3/3） | +7.66 ± 10.82（胜2/3） |
| ABL | 40 | +11.35 ± 8.86（胜3/3） | +10.90 ± 5.46（胜3/3） |

## 验证与证据

- 模型源码固定于`a4d7a008d7e7597a864624f63f4185d91fada144`；仅增加独立Test脚本与输出。
- 54份权重均复现原Val预测；独立重载54模型，用encode/features和classifier分开重建Test输出，逐项scores与预测完全一致。
- 完整类别混淆矩阵复算Macro-F1，与sklearn一致；保存逐条预测、每类F1、混淆矩阵。权重、BN缓冲、旧文件、数据和包版本不变。
- Proto另用广播平方距离与同一FP32特征的FP64距离检查舍入误差；正式指标仍用冻结FP32 cdist预测，诊断不替换预测或重选模型。
- 54模型首次Test推理窗口4.58秒（含进程启动、数据读取和Val复核；不含独立核验、报告及同步）。
- 证据：TEST_MANIFEST、CHECKPOINT_MANIFEST、PREFLIGHT、EVALUATION、VERIFICATION、SUMMARY、PER_SEED_RESULTS及runs逐模型记录。
- 旧训练报告test_evaluated:false保留当时状态，新报告为true。本地完整归档另有scores/特征/原型NPZ；GitHub仅上传文本证据及脚本，不上传波形、权重或环境。

## 核验入口修正与复现命令

第一次独立核验在模型创建外层使用inference_mode，导致参数成为inference tensors；首个模型logits与正式评估最多相差1.1444e-5，预测没有差异，逐位核验因此停止。只读诊断确认：常规模型创建时，forward与分解路径均逐位复现正式分数；在inference_mode内创建则出现该差异。因此新增verify_reload.py移除外层上下文，只保留infer内部无梯度推理。

此修正没有改evaluate.py、容差、54份模型权重或首次Test输出。失败日志、诊断与入口修正记录均归档；随后54模型分数和预测逐位核验全部通过，Proto的直接FP32与FP64距离诊断也没有改变任何预测。

服务器仓库目录下复现（复用原环境；需保留原Data.zip和54份best权重；已有Test输出不能覆盖）：

```bash
export PYTHONPATH="$PWD/src"
PY=../analog-circuit-crossdomain/.venv/bin/python
$PY server_experiment/fixed_test/evaluate.py preflight
$PY server_experiment/fixed_test/evaluate.py run
$PY server_experiment/fixed_test/verify_reload.py
$PY server_experiment/fixed_test/evaluate.py report
```

runtime_source保存原服务器脚本层次及依赖、日志；按SNAPSHOT_MAP恢复原路径后才能执行。GitHub只归档可审阅的文本，不含数据、权重、NPZ或uv环境。SERVER_RESULTS.md在自动汇总后补充上述解读和核验入口说明，其主要数值与SUMMARY/PER_SEED_RESULTS一致。
