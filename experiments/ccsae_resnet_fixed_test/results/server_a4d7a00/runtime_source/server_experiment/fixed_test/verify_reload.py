"""修正核验入口的上下文边界，不改变正式评估或原核验逻辑。

首次核验给整个verify加inference_mode，导致模型参数在该上下文中创建；
正式评估先常规创建模型，再仅在infer中禁用梯度。诊断已确认差异来源。
此入口移除外层装饰器，保留infer内部inference_mode，逐位核验正式输出。
原evaluate.py和首次54模型预测、权重清单均不改写。
"""
import evaluate

if __name__ == '__main__':
    # torch装饰器保留__wrapped__，此处只调整模型创建的上下文边界。
    # 未修改模型、scaler、距离、批大小、Test、验证条件或指标计算。
    evaluate.verify.__wrapped__()
