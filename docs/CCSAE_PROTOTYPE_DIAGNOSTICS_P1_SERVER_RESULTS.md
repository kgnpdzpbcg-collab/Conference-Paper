# CCSAE 原型诊断 D0 与距离尺度 P1：服务器验证结果

实际源码提交：`a4d7a008d7e7597a864624f63f4185d91fada144`；分支 `research/ccsae-prototype-diagnostics-v1`。原 P0 训练提交 `7ef4042`。

**结果：距离尺度由除以嵌入维度 D 改为不除以 D 后，六组 P1 原型均优于各自 P0 原型；SKL 三组均优于 FC，ABL 两组优于 FC、一组略低。仅为固定小验证集结果，没有 Test 评估。**

## 1. 执行顺序与版本边界

按最新 MD：16 项上游测试通过 → 六组旧 Stage1 checkpoint 的 D0 → D0 有尺度/梯度证据后，两电路 seed42 正式预算预检 → 两组原型明显改善，补 seed43/44 → 独立核验全部六组。没有跳过门槛，也没有在 Test 选择方案。

本地及服务器新建独立诊断分支目录，旧 P0 checkout 仍在原目录且训练源码固定。复用已有 `analog-circuit-crossdomain/.venv`，Python 3.12.3 / torch 2.14.1+cu130 / RTX 5070 Ti，确定性 FP32、禁用 TF32/AMP、math SDP、每工作进程 4 CPU 线程。通过 PYTHONPATH 直接导入新源码，没有执行 pip install，没有创建环境或下载包。

## 2. D0：旧权重只读诊断

加载的是 `shared_stage1_last.pt`，不是训练后的 `proto_best.pt`。构造原 P0 的 K=20 Train 和 5 Val/类，逐类样本和 Train/Val/Test 池哈希均核验一致。为重建这一划分，loader 会读取原包的官方 Train/Test 文件；没有对新协议的留出 Test 池生成模型预测。

| 电路 | seed | 固定 Stage1 全 K 原型 Val F1 (%) | 3-shot Val F1 均值±SD (%) | 投影梯度 none/dim | 卷积梯度 none/dim | 类内距离/最近类中心距离 | Support 漂移均值 | BN 相对 L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ABL | 42 | 46.16 | 36.45 ± 5.64 | 157.23 | 155.15 | 3.007 | 1.514 | 0.055 |
| ABL | 43 | 39.95 | 40.34 ± 5.79 | 20.57 | 16.66 | 4.062 | 1.972 | 0.062 |
| ABL | 44 | 37.83 | 37.07 ± 5.29 | 71.18 | 79.05 | 3.102 | 1.474 | 0.056 |
| SKL | 42 | 37.99 | 36.71 ± 5.13 | 106.49 | 67.53 | 1.852 | 0.891 | 0.127 |
| SKL | 43 | 47.74 | 38.64 ± 4.68 | 81.37 | 97.94 | 1.679 | 0.801 | 0.091 |
| SKL | 44 | 44.18 | 37.65 ± 4.86 | 48.24 | 31.16 | 1.947 | 0.873 | 0.210 |

32 次 Support 抽样的 SD 是同一固定 checkpoint 内的抽样波动；它与后面的三种子 SD 不是同一种不确定性。Support 漂移按最近异类中心间距归一，BN 为同一 Episode batch 下 train/eval 特征的代理比较。几何统计按类分别计算再平均，类内/类间比值均值不等于两个均值之比。

全部六组 dim/none 固定特征预测一致，且与 FP64 cdist 参考一致；仅改推理分数不能提高 F1。原 dim 的 Episode CE 接近 ln(13) 或 ln(11)，softmax 近均匀，梯度路径正常但相对 none 更小。投影梯度倍率约 20–157，卷积约 17–155；它们是两个参数张量的未裁剪梯度，不能当作整个模型优化步长倍率。

类中心重叠和 3-shot 原型漂移也存在，因而不能只凭 D0 断言尺度是唯一原因。D0 模型参数和 BN buffers 运行后逐字节恢复；确定性复算一致；独立直接 FP64 平方距离和 sklearn 指标复算一致。

## 3. P1：严格单因素对照

唯一算法因素：原型 logits 从 `−||z−p||²/(D·τ)` 改为 `−||z−p||²/τ`，τ 固定为 1，SKL D=1000、ABL D=600。FC 的训练及判别公式不变。

其他条件：K=20、Val=5/类、split/val seed=20261009、sample/training seed=42/43/44；Stage1 10 epoch、batch64、signed_real MSE+SupCon；Stage2 40 epoch×10 step；每类 3 Support+2 Query；SGD momentum0.9、clip5，SKL lr0.0022/λ0.0017，ABL lr0.0102/λ0.0059。全部 K Train 生成验证原型，Val Macro-F1 选最佳 epoch，平局保留最早。

本轮重新执行 Stage1 与 FC，用于核验与 P0 的严格复现。六组 Stage1 参数/buffer、FC 最佳参数/buffer、FC 全训练曲线及实际 Episode 样本流均与 P0 完全一致。FC checkpoint 文件含新增配置字段，因此比较的是 state_dict 指纹，不能要求整个 checkpoint 文件字节相同。

| 电路 | FC Val Macro-F1 (%) | P0 Proto dim (%) | P1 Proto none (%) | P1−FC (pp) | P1−P0 Proto (pp) |
|---|---:|---:|---:|---:|---:|
| SKL | 59.74 ± 1.87 | 46.99 ± 3.10 | 70.27 ± 3.43 | 10.54 ± 4.43 | 23.28 ± 3.70 |
| ABL | 61.94 ± 3.14 | 42.98 ± 1.21 | 68.86 ± 7.03 | 6.92 ± 7.15 | 25.88 ± 8.15 |

均值 ± 样本标准差（ddof=1，n=3）；差值为同种子配对差值的均值与 SD。

| 电路 | seed | FC (%) | P0 Proto (%) | P1 Proto (%) | P1−FC (pp) | P1 最佳 epoch |
|---|---:|---:|---:|---:|---:|---:|
| ABL | 42 | 58.40 | 42.95 | 65.67 | +7.26 | 39 |
| ABL | 43 | 64.39 | 44.20 | 63.99 | -0.40 | 40 |
| ABL | 44 | 63.03 | 41.79 | 76.92 | +13.89 | 39 |
| SKL | 42 | 61.43 | 44.14 | 66.90 | +5.47 | 29 |
| SKL | 43 | 57.74 | 50.29 | 70.15 | +12.41 | 39 |
| SKL | 44 | 60.04 | 46.55 | 73.77 | +13.72 | 27 |

ABL seed43 比 FC 低约 0.40 pp；ABL 的 P1 三种子 SD 为 7.03 pp。不能写成“每组全面优于 FC”或已验证稳定 Test 泛化。ABL 三组最佳 epoch 均在 39–40，本轮按预设预算结束，没有因结果追加训练轮数。

## 4. 时间与资源

D0 六组四进程诊断窗口 4.190 秒；seed42 两进程训练 20.186 秒；seed43/44 四进程训练 35.224 秒。P1 两批训练窗口合计 55.410 秒，不含同步、核验、报告生成。没有测量整批串行正式运行，不推算正式并发加速比。

| 阶段 | 平均 GPU 利用率 | 峰值显存 MiB |
|---|---:|---:|
| seed42_precheck | 81.15% | 1175 |
| seeds43_44_followup | 88.23% | 2346 |

资源每秒采样，窗口包括子进程启动、加载与保存。原始采样和时间戳保存于各批次 JSON。

## 5. 核验、结论和限制

16 项测试通过；6 组 D0 和 6 组 P1 配对独立核验通过。170 个旧结果、数据及环境配置文件指纹与执行前一致，已安装包版本清单一致。六组 P1 checkpoint 重载后的验证预测与保存结果完全一致，独立混淆矩阵复算 F1/Accuracy/Balanced Accuracy 一致，Support/Query 互斥且按类平衡，未训练 decoder 和 Proto classifier 保持原值。

单因素对照支持“距离分数尺度明显影响本协议下的原型训练优化结果”。它没有证明尺度是所有差距的唯一原因，也没有证明普通原型在所有数据、训练预算或真实部署条件下优于 FC。D0 的几何和 BN 代理检查仍不能单独建立因果归因。

只评估固定每类 5 条 Val，n=3且采样/训练随机种子耦合；没有 Test 评估。P0 新划分与历史 Data_deduplicated 不同，不能混用旧 CNN/ResNet/full-data/few-shot 成绩。原始物理单位、完整故障列索引、独立板卡/批次来源未被本轮解决；本轮多分类不依赖 C0=健康的假设。数据曾用于先前探索的限制持续存在。

本轮完成文档预定的 D0/P1，停止追加模块与参量搜索。若继续研究，应先冻结主协议和候选配置，再单独决定下一阶段评估；不因某个结果追加 Test 筛选。

## 6. 原始证据与复现

D0：`outputs/proto_diagnostics/server_a4d7a00/`；P1：`outputs/proto_scale_p1/server_a4d7a00/`。所有完整权重与 Episode NPZ 留在服务器及本地同步包，GitHub 只归档 JSON/CSV/日志/源代码快照，禁止上传 Data.zip 或模型权重。

复用实际 Python 环境，以新项目根为工作目录：

```bash
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
../analog-circuit-crossdomain/.venv/bin/python -m pytest -q
../analog-circuit-crossdomain/.venv/bin/python -m ccsae.proto_diagnostics --data ../Data.zip --run-dir ../Conference-Paper-ccsae-prototype-k20/outputs/server_proto_k20_7ef4042/skl/k20_sample42_train42 --seed 42 --device cuda --repeats 32 --output outputs/proto_diagnostics/fresh/skl42.json
```

实际并发封装与验证脚本见 `server_experiment/` 的运行源快照。为复现 FP32 数值，实际封装还调用 common.configure() 禁用 TF32/AMP 和非 math SDP；仅复制 CLI 命令不保证默认后端完全一致。新输出必须另选空目录，正式权重不能从 GitHub 文本归档取得。

关键证据：D0 VERIFICATION.json、P1 各批 verification.json、COMPLETION_VERIFICATION.json、SOURCE_MANIFEST.json、PER_SEED_RESULTS.csv、全部 Train/Val 曲线和固定最佳预测。
