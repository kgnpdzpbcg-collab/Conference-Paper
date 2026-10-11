# ResNet1D：同P1数据及分类协议的独立对照

[完整服务器报告](../../docs/RESNET1D_P1_CLASSIFICATION_PROTOCOL_SERVER_RESULTS.md) · [对应CCSAE-FC/P1规模结果](../ccsae_prototype_shot_scale/README.md)

**采用用户选定协议：保留原ResNet，从随机初始化开始，不做Stage1。数据及分类训练与CCSAE-FC对齐；CCSAE保留10轮Stage1，这一方法差异必须随结果报告。没有Test评估。**

| 电路 | K | CCSAE-FC Val F1 (%) | CCSAE-Proto P1 Val F1 (%) | ResNet1D Val F1 (%) |
|---|---:|---:|---:|---:|
| SKL | 10 | 59.76 ± 9.32 | 65.28 ± 5.74 | 58.59 ± 3.58 |
| SKL | 20 | 59.74 ± 1.87 | 70.27 ± 3.43 | 61.09 ± 4.31 |
| SKL | 40 | 61.78 ± 4.97 | 70.79 ± 5.65 | 61.49 ± 4.30 |
| ABL | 10 | 64.98 ± 8.05 | 70.20 ± 10.42 | 62.25 ± 3.64 |
| ABL | 20 | 61.94 ± 3.14 | 68.86 ± 7.03 | 61.20 ± 3.79 |
| ABL | 40 | 62.83 ± 2.76 | 68.14 ± 12.33 | 63.30 ± 0.80 |

均值 ± 样本标准差，n=3。ResNet与FC均值接近，18组中17组低于Proto；不声称显著性或纯Backbone因果结论。

## 冻结与核验

- 18次训练：SKL/ABL×K10/20/40×seed42/43/44；固定5 Val/类，当前P1数据池、逐类Train、标准化和400步实际Episode全部一致。
- 使用原ResNet-18式1D结构，通道32/64/128/256、256维特征、原线性头；原源码SHA256与历史基线一致。
- Stage1=0，从随机初始化开始；Stage2固定40×10更新、SGD momentum0.9、clip5；SKL lr0.0022，ABL lr0.0102。无调度、早停、增强或BN重估。
- 四进程、每进程四CPU线程，复用原uv环境；训练窗口53.217秒，未重建环境或安装依赖。
- 18次初始化、最后及最佳checkpoint核验通过：同种子跨K初始化一致，400步采样重放一致，BN训练计数400，最佳权重Val预测与存档逐条一致，独立混淆矩阵指标一致。
- 1100个旧实验/数据/环境文件前后字节一致，已安装包清单一致，旧CCSAE/P1/ResNet未改写。

## 文件与边界

`results/server_a4d7a00/`保存SUMMARY/PER_SEED_RESULTS、18个配置/曲线/Val预测、真实Episode指纹、独立验证、运行环境、保护清单及资源记录。provenance/splits逐条对应当前P1，不对应历史Data_deduplicated协议。

`runtime_source/`保留服务器脚本原字节及原路径层次；新增文件位于server_experiment/resnet_shot_scale，复用的common.py、资源采样入口和独立指标函数位于server_experiment。源码快照用于核查本次实际运行，不替换仓库主模型代码。

独立复现需把runtime_source中的server_experiment层次恢复到项目根，并固定CCSAE源码a4d7a00、原数据与已有P1只读参考结果；为audit.py的BASE和相关入口配置新的空输出目录。原存档路径拒绝覆盖。

276个完整服务器文件已通过本地原字节核对，指标及配对统计由Python标准库再次复算。LOCAL_SYNC_VERIFICATION.json记录同步检查；PUBLICATION_VERIFICATION.json记录服务器文本与Git归档路径/指纹对应。完整包SHA256：`e8a02f3eec40d9d9336844141f5201548732b8dea421952d7df5f0d2911fbdde`。

初始化/最佳/最后权重、Episode和分数NPZ保留服务器及本地同步包，不上传数据、模型权重或环境。固定小Val用于选模并汇总，非独立Test成绩；n=3且抽样/训练随机性耦合，精确去重不保证采集来源独立。历史ResNet使用不同划分、AdamW、完整Val和早停，其Test成绩不可与本表直接混用。
