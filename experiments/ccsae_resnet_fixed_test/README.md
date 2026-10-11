# 当前固定Test：CCSAE-Proto、CCSAE-FC、ResNet1D

本阶段只加载已保存的Val最佳权重，测试SKL/ABL、K=10/20/40、种子42/43/44的54个模型；与旧训练、Val规模验证和历史少样本Test独立归档。

- [最终Test Macro-F1与完整说明](../../docs/CCSAE_RESNET_FIXED_TEST_SERVER_RESULTS.md)
- [服务器原字节报告](results/server_a4d7a00/SERVER_RESULTS.md)
- [逐种子Test结果](results/server_a4d7a00/PER_SEED_RESULTS.csv)
- [完整Test指纹与顺序](results/server_a4d7a00/TEST_MANIFEST.json)
- [推理前冻结的54份最佳权重清单](results/server_a4d7a00/CHECKPOINT_MANIFEST.json)
- [独立重载核验](results/server_a4d7a00/VERIFICATION.json)
- [本地标准库复算及传输核验](results/server_a4d7a00/LOCAL_VERIFICATION.json)
- [核验上下文修正证据](results/server_a4d7a00/VERIFICATION_CONTEXT_CORRECTION.json)
- [发布文件与完整归档指纹](results/server_a4d7a00/PUBLICATION_MANIFEST.json)

固定完整Test：SKL 570条、ABL 465条。沿用当前P1首出现float32去重划分，不使用历史Data_deduplicated Test。标准化只用各组K条/类Train统计量，原型仅由Train的eval特征构建，Test未用于拟合或选权重。

CCSAE-FC/Proto有10轮Stage1，ResNet无Stage1；分类阶段对齐400次SGD和相同Episode。当前模型Train/Test互斥，但数据集与部分波形曾用于历史实验，不能称全新外部盲测。详见结果文档的协议边界。

`runs/<case>/k<K>_seed<S>/`包含三模型逐条预测、混淆矩阵、每类F1与最佳轮次。`runtime_source/`保存原服务器脚本层次及日志；`SNAPSHOT_MAP.json`记录原路径和指纹。先恢复原路径再按报告命令运行；核验使用`verify_reload.py`，避免在外层inference_mode创建模型。

完整归档`ccsae-resnet-fixed-test-a4d7a00-server-results.tar.gz`（30,472,804字节，206文件）的SHA256为`5ef4169aab79362d39b9bb355f21962e4a79fd5d0a8eea28d0de59bfe8acfabc`，保存在服务器和本地工作目录的仓库父目录。归档额外包含54份NPZ分数（Proto另含特征与中心），不复制模型权重。GitHub只发布文本、脚本、日志及核验；TRANSFER_MANIFEST列出完整归档而非仅Git文件。

旧报告的`test_evaluated:false`保持原历史状态，本阶段明确为`true`。复用原uv环境，未安装或下载依赖，没有重训、调参或Test重选。首次Test输出始终保留。
