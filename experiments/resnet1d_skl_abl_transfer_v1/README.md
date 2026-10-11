# ResNet1D第一轮SKL→ABL少样本迁移

**本轮未发现显著正向迁移收益。** 相同ABL标注下比较Target-only、Frozen Transfer、Fine-tuning Transfer；K=10/20/40、10个配对种子42–51。在六项共同Holm校正后，Frozen K40显著下降7.11个百分点；Fine K40平均提高1.05个百分点但不显著。源预训练Val为53.61 ± 7.22%，结论限于当前配置，不能推断全部预训练方法无效。

- [服务器完整结果、协议和解释范围](../../docs/RESNET1D_SKL_ABL_TRANSFER_V1_SERVER_RESULTS.md)
- [冻结协议](results/server_a4d7a00/PROTOCOL.json)
- [逐种子Test结果](results/server_a4d7a00/PER_SEED_RESULTS.csv)
- [九组汇总与六组配对统计](results/server_a4d7a00/SUMMARY.json)
- [100次训练独立核验](results/server_a4d7a00/TRAINING_VERIFICATION.json)
- [90模型Test独立重载核验](results/server_a4d7a00/TEST_VERIFICATION.json)
- [统计复算](results/server_a4d7a00/STATISTICAL_VERIFICATION.json)
- [本地标准库复核](results/server_a4d7a00/LOCAL_VERIFICATION.json)
- [运行脚本原路径与SHA256](results/server_a4d7a00/SNAPSHOT_MAP.json)
- [权重指纹（300份权重保留服务器）](results/server_a4d7a00/WEIGHTS_MANIFEST.json)
- [发布文件原字节映射](results/server_a4d7a00/PUBLICATION_MANIFEST.json)

原ResNet主体不变；无Stage1；源2660 Train/65 Val，目标每类K Train/5 Val、Test465条。目标三条件逐步Episode、新11类头、归一化、400次更新、SGD配置相同。Frozen同时冻结骨干参数和BN缓冲；Fine全参数与训练BN适配。源13类头丢弃，每个种子的源最佳骨干在三个K上复用。源域标签和预训练计算是额外资源，不声称总标注或总计算等量。

服务器训练固定在提交`a4d7a008d7e7597a864624f63f4185d91fada144`，复用原uv环境，没有下载包。10源模型+90目标模型的训练窗口合计约211.31秒；核验、Test、报告与同步另计。当前Test已用于历史探索，统计是固定Test下抽样/训练波动的条件推断。

本目录与已有少样本、固定Test、历史跨域实验独立。Git保留服务器文本和代码快照；完整非权重归档在服务器工作目录和本地工作目录父层，包含NPZ分数、Episode和bootstrap数组：

`resnet1d-skl-abl-transfer-v1-a4d7a00-server-results.tar.gz`

- 大小：7,534,395字节；1242个文件。
- SHA256：`4955d8a98d7c2376db3e1f7d127fb62cb12c79a6b4a847706d0ea4f29b199bf7`。
- 权重路径：`/home/wang/projects/会议论文/Conference-Paper-ccsae-prototype-diagnostics-v1/outputs/resnet_transfer_v1/server_a4d7a00/`。
- 完整归档不含PT、原始Data.zip或环境；300份initial/best/last权重均在服务器，100份best SHA与冻结清单相符。

按SNAPSHOT_MAP还原脚本相对路径，在训练提交上运行；源数据和旧固定Test审计仍是必要输入。快照只调用旧模块的只读数据/模型/资源监测工具，不调用旧训练主入口。现有完成目录不可覆盖；复现实验必须先另设输出路径。本地复核脚本仅使用标准库，不做本地模型推理或训练。
