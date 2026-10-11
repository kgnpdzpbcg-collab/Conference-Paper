# CCSAE-Proto K=20 负结果分析与最小诊断修订（2026-10-11）

> 状态：分析基于已完成的服务器 P0 实验；诊断工具与距离尺度选项已实现，**新诊断及 P1 重训尚无正式结果**。本文件存放在 docs/research；历史结果与原始代码在原分支保持不变。

## 1. 核心结论（已观测，不是推测）

事实来源：../CCSAE_PROTOTYPE_K20_SERVER_RESULTS.md，以及 ../../experiments/ccsae_prototype_k20/results/server_7ef4042/SUMMARY.json 和 PER_SEED_RESULTS.csv。

固定 K=20/类、额外 5 Val 标签/类；SKL 13 类、ABL 11 类；3 个配对随机种子 42/43/44。共享 Stage1 10 epochs（signed_real MSE + SupCon）；Stage2 40×10 SGD 更新。每个 Episode 每类 3 Support+2 Query；Proto 使用负平方欧氏距离 / 嵌入维度 D，温度 1。**没有进行 Test 评估。**

| 电路 | FC Val Macro-F1 | Proto Val Macro-F1 | Proto − FC |
|---|---:|---:|---:|
| SKL | 59.74 ± 1.87% | 46.99 ± 3.10% | −12.74 ± 4.96 pp |
| ABL | 61.94 ± 3.14% | 42.98 ± 1.21% | −18.96 ± 3.09 pp |

六组配对 Proto 都落后 FC；独立核验确认初始化、Episode 采样、验证指标及预测重载一致。因此**P0 下普通均值原型的增益假设未获支持**。这不能证明所有 Proto 方法都无效。

SKL seed42：Proto Train CE 2.5627→1.8925，但 Val F1 最好在 epoch2（44.14%），最后一轮 37.00%，FC 最好 epoch39（61.43%）。SKL seed43 Proto 最好 epoch1；seed44 最好 epoch40。ABL 的 Proto 最好为 epoch32/36/38，均不如 FC。不能说全部 Proto 都在第一轮就过拟合。

Stage1 SKL seed42：重构 MSE 1.0743→0.4934，SupCon 3.9658→4.0372；ABL seed43：重构 1.0351→0.0938，SupCon 3.7858→4.0033。说明重构明显改善，**并未从这些数字证明类内更聚集**，也不能仅凭损失系数确定梯度贡献。

服务器数值核验发现训练初期个别 float32 cdist 近并列，**正式最佳 checkpoint 在全部验证记录上与 FP64 距离计算预测一致**；不能把数值舍入认定为主要性能差距原因。

## 2. 三个需要诊断的机制

### H-A：距离尺度影响训练优化

原代码：

\[
\ell_{q,c}=-\frac{\|z_q-p_c\|_2^2}{D\tau}.
\]

SKL D=1000、ABL D=600、温度 1。若初始特征向量范数小，类别 logits 和 margin 也可能小，CE 接近 ln(C)。但**不能由 logits 小就断言梯度为零**，需直接测第一层复数卷积和投影层的真实梯度。

新增受控选项：

\[
\ell_{q,c}^{\rm none}=-\frac{\|z_q-p_c\|_2^2}{\tau}.
\]

**数学关键**：固定 z 与 p 时，正尺度倍乘在精确算术下不会改变 argmax，因此仅修改推理尺度不能改善 F1；是否帮助**重新训练**需试验。训练中的梯度变大也不等于更好，可能过大/饱和。只做一个因素 dim→none，暂不网格搜索温度。

### H-B：嵌入几何未必适合均值原型

CCSAE Stage1 以重构为重要目标，未保证每类故障为一个紧致、单峰的簇。FC 更容易表达非球形类间边界。测量类内 RMS 距离、最近异类中心间距及其比值，并用固定 Stage1 嵌入的全部 K-shot 训练原型分类 Val；这属于**描述性机制检验**，不是少样本效果提升的证明。

### H-C：Support 从 3 条训练到 20 条推理以及 BatchNorm 状态差异

同类三个 Support 的均值可能对抽样敏感，而 Val 使用每类全部 K=20 条训练样本求原型。随机 32 次 Support 子集，统计其相对全部 K 原型的位置偏移及对应 Val F1 分布。模型含多处 BN，train/eval 特征可能变化；用**相同 Episode batch**在两种模式下的余弦相似度/相对 L2 作为代理诊断。诊断后恢复权重与 running buffers，不保存临时 BN 更新。

上述三项是**合理假设**，不是现有结果已证实的因果机制。对照必须固定其他参数才能归因。

## 3. 仓库修改清单与边界

| 路径 | 本轮新增/修改 |
|---|---|
| src/ccsae/prototype.py | 在 prototype_logits 增加 distance_normalization=dim/none，**默认 dim 与 P0 完全兼容** |
| src/ccsae/fewshot_train.py | 新 CLI 参数 --distance-normalization、保证 Train 与 Val 一致；新运行 none 自动在目录名后加 _distnone |
| src/ccsae/proto_diagnostics.py | 新只读诊断入口，加载已有 Stage1 服务器权重、验证数据哈希，不读取 Test 做预测；输出单独存放 |
| tests/test_proto_diagnostics.py | 默认等价性、距离尺度、梯度以及诊断运行后模型状态不变等测试 |
| pyproject.toml | 注册命令 ccsae-proto-diagnose |
| docs/research/CCSAE_PROTO_K20_FAILURE_ANALYSIS.md | 本文，分析与后续执行门槛 |
| docs/README.md | 文档索引 |
| README.md | 新阶段入口 |
| .github/workflows/ccsae-tests.yml | 让新分支触发 CI |

**未修改**：CCSAE V1 的基础编码器、CSA、signed_real 重构、数据预处理；experiments/ccsae_prototype_k20/results/server_7ef4042 下全部历史证据。**未实现**加权原型。

分支管理：
- 原 P0：research/ccsae-prototype-k20（已完成训练，冻结）。
- 新诊断及候选 P1：research/ccsae-prototype-diagnostics-v1（在原结果之上仅增加工具和单因素开关）。
- 原始模型权重及数据留在本地/服务器，不放 GitHub；未来 P1 实测结果应单独归档到 experiments/ccsae_prototype_scale_p1/，不追加写入 P0 结果目录。

## 4. 本地执行（先诊断，后考虑重训）

先切分支：

~~~bash
git fetch origin
git switch research/ccsae-prototype-diagnostics-v1
python -m pip install -e '.[dev]'
pytest -q
~~~

### D0：利用已存在的权重，**不重新训练**

服务器 K20 权重未上传 GitHub；必须指向服务器/本地真实输出目录（包含 config.json、split_audit.json 和 shared_stage1_last.pt），不能指向仅有 JSON 的 Github 归档目录。例如：

~~~bash
ccsae-proto-diagnose --data /home/wang/projects/会议论文/Data.zip \
  --run-dir outputs/server_proto_k20_7ef4042/skl/k20_sample42_train42 \
  --device cuda --repeats 32 \
  --output outputs/proto_diagnostics/skl/k20_seed42_stage1.json
~~~

另外对 skl 43/44 和 abl 42/43/44 各跑一次。程序重新计算哈希，校验新建 Train/Val 划分与服务器原先的指纹一致，失败立即报错；只用这次 K Train 和指定 Val。输出含固定 checkpoint 的 dim/none 类别分数、差异预测数、投影与卷积真实梯度、类内外距离、3-shot Support 原型漂移、BN 状态影响。报告 test_evaluated=false。

**注意：原 P0 checkpoint 是 Stage1 权重而不是 Proto best。诊断结果仅解释 Stage1 的可用特征与初始化状态，不等于对 Proto best 训练结束表征的完整分析。**

### P1：只改训练时原型距离分数尺度（需要 D0 支持）

~~~bash
for case in skl abl; do
  ccsae-fewshot --data data/Data.zip --case "$case" \
    --k 20 --seed 42 --sample-seed 42 \
    --stage1-epochs 10 --epochs 40 --steps-per-epoch 10 \
    --distance-normalization none \
    --output outputs/proto_scale_p1
done
~~~

当前默认 mode=both，仍同步训练 FC/Proto；与 P0 的历史命令仅一个区别是 dim→none。新目录形如 outputs/proto_scale_p1/skl/k20_sample42_train42_distnone，**不覆盖 P0**。若 seed42 呈现有价值的 Val 改善，才补 43/44；检验同 seed FC 的复现一致性。梯度可能过大，出现 loss 非有限时视为诊断结果，不进行大规模参量搜索。

## 5. 决策门槛

1. D0 首先排除数据哈希、checkpoint、数值/梯度异常。
2. 若当前特征的类别中心混叠严重、3-shot 类中心漂移高：**不能认为只调距离缩放必然有效**，应先承认单均值 Proto 假设可能不适配。
3. 若距离尺度确实使训练梯度弱，可做至多一轮受控 P1，比较 3 seed 验证 Macro-F1（当前只运行 seed42 是预检，不能写论文结论）。
4. 若 P1 仍明显落后 FC，停止普通 Proto Head 主线；不要为了事后维持论文故事继续无依据叠模块。
5. 无论怎样，不在 Test 上筛选新模块。本数据集此前已用于多次实验探索，后续论文如进行最终 Test 还需披露此限制。

**研究目标仍然是可信、简单的 EI 会议论文，而非把负结果强行包装为成功。**
