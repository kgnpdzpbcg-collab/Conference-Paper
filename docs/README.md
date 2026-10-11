# Conference-Paper 文档索引

## 当前算法研究（先读）

- [ResNet1D第一轮SKL→ABL少样本迁移](RESNET1D_SKL_ABL_TRANSFER_V1_SERVER_RESULTS.md) —— 相同ABL标签，三条件、10种子、90目标Test；无显著正向收益，Frozen K40显著下降。
- [迁移独立归档与核验](../experiments/resnet1d_skl_abl_transfer_v1/README.md) —— 代码快照、冻结BN、真实Episode、逐种子预测与六项共同Holm检验。

- [当前固定Test最终Macro-F1](CCSAE_RESNET_FIXED_TEST_SERVER_RESULTS.md) —— CCSAE-Proto/FC/ResNet，54份Val最佳权重只读评估，完整Test 570/465条，未重训或调参。
- [固定Test独立归档及核验](../experiments/ccsae_resnet_fixed_test/README.md) —— 权重冻结清单、逐条预测、分数重载、完整Test指纹与历史探索边界。

- [ResNet1D对齐P1数据与分类训练协议](RESNET1D_P1_CLASSIFICATION_PROTOCOL_SERVER_RESULTS.md) —— 原ResNet无Stage1，400次分类更新，固定5 Val/类，无Test。
- [ResNet对照独立归档](../experiments/resnet1d_p1_classification_protocol/README.md) —— 18次训练、真实Episode/权重/指标核验。

- [冻结P1的K=10/20/40规模验证](CCSAE_PROTOTYPE_SHOT_SCALE_SERVER_RESULTS.md) —— 固定5 Val/类，12组新配对+6组K20复用，无Test。
- [少样本规模验证独立归档](../experiments/ccsae_prototype_shot_scale/README.md) —— 嵌套样本、逐种子结果、独立核验与冻结证据。

- [D0 / P1 最新服务器验证结果](CCSAE_PROTOTYPE_DIAGNOSTICS_P1_SERVER_RESULTS.md) —— 单因素距离尺度对照，固定5 Val/类，未评估Test。
- [D0 / P1 实验原始证据](../experiments/ccsae_prototype_scale_p1/README.md) —— 与P0独立归档。

- [K=20 Proto 负结果分析、数值/特征诊断与 P1 最小修订](research/CCSAE_PROTO_K20_FAILURE_ANALYSIS.md) —— 原负结果与预定门槛，D0/P1执行状态见最新报告。
- [CCSAE＋Prototype 原始 K-shot 研究方案](CCSAE_PROTOTYPE_FEWSHOT_V1.md) —— P0 冻结协议、模型公式与实验设计。
- [K=20 服务器验证结果](CCSAE_PROTOTYPE_K20_SERVER_RESULTS.md) —— FC 优于 Prototype 的六组验证对照；没有 Test。
- [K=20 完整实验索引与原始证据](../experiments/ccsae_prototype_k20/README.md) —— 训练曲线、逐 seed 指标、核验与环境信息。

## 基础算法与历史实验

- [CNN / ResNet1D / CCSAE v1 历史少样本Test结果](FEWSHOT_BASELINES_CCSAE_V1_SERVER_TEST_RESULTS.md) —— K=10/20/40，三种子，完整Val选模后评估Test；与P1协议分开。
- [历史少样本实验原始证据](../experiments/analog_circuit_fewshot_server_v1/README.md) —— 原配置、曲线、逐种子成绩及核验。

- [CCSAE V1 修正实验计划](CCSAE_V1_EXPERIMENT_PLAN.md)
- [CCSAE 论文实现边界说明](REPRODUCTION_NOTES.md)
- [历史服务器 GPU 实验](SERVER_GPU_EXPERIMENT_RESULTS.md)

## 文件管理规则

1. 模型原始源码保留在 src/ccsae；新增诊断使用独立模块 proto_diagnostics.py。
2. 分支 research/ccsae-prototype-k20 的 P0 历史结果和实验记录保持不变；新诊断分支为 research/ccsae-prototype-diagnostics-v1。
3. D0/P1距离尺度实测归档到experiments/ccsae_prototype_scale_p1/；冻结P1规模验证归档到experiments/ccsae_prototype_shot_scale/，均不覆盖旧结果。
4. 本地 data/、outputs/、模型权重与环境文件禁止提交 GitHub；记录可复验的文本摘要和哈希。
5. 方法选择只参考 Train/Val；最终 Test 需在模型冻结后使用，并报告已存在的先前探索限制。
