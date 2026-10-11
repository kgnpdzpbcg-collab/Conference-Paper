"""从逐样本预测计算固定类别的指标，支持 11 类和 13 类，零分母按零处理。"""

import numpy as np


def classification_metrics(labels, prediction, classes: int) -> dict:
    """行是真实类、列是预测类；Macro-F1 始终平均全部类别。

    模型没有预测某一类时，该类精确率和 F1 为零，不能跳过该类抬高宏平均。
    Balanced Accuracy 是各类召回率的算术平均，不受原类别比例直接支配。
    """
    labels, prediction = np.asarray(labels), np.asarray(prediction)
    confusion = np.bincount(labels * classes + prediction, minlength=classes ** 2).reshape(classes, classes)
    tp = np.diag(confusion).astype(float)
    support, predicted = confusion.sum(1), confusion.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(classes), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(classes), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(classes), where=precision + recall > 0)
    return {"accuracy": float(tp.sum() / len(labels)), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall.mean()), "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(), "per_class_f1": f1.tolist(),
            "support": support.tolist(), "confusion_matrix": confusion.tolist()}
