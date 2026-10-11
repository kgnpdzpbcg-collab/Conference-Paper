"""普通 1D-CNN 和 ResNet1D，统一输入为 [批大小, 1, 1000] 的波形。

输出为各类别的未归一化分类分数，不在模型内部执行 softmax。
ResNet1D 采用 ResNet-18 的基本残差块布局，但将二维卷积换为一维卷积，
起始宽度为 32 通道；它是本轮固定基线，不宣称复现某篇论文的全部设置。
"""

from torch import nn


class CNN1D(nn.Module):
    """三个普通卷积块，通道 32/64/128、卷积核 7/5/3，随后全局池化分类。

    每个块使用批归一化、ReLU 和二倍最大池化。末端全局均值池化控制参数量，
    两层分类头使用 0.2 dropout；模型没有残差连接或注意力模块。
    """

    def __init__(self, classes: int):
        super().__init__()
        blocks, previous = [], 1
        for channels, kernel in ((32, 7), (64, 5), (128, 3)):
            blocks.extend([nn.Conv1d(previous, channels, kernel, padding=kernel // 2, bias=False),
                           nn.BatchNorm1d(channels), nn.ReLU(), nn.MaxPool1d(2)])
            previous = channels
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                                        nn.Linear(64, classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock1D(nn.Module):
    """两个 3 点卷积组成的基本残差块。

    改变通道数或时间长度时，捷径使用 1 点卷积和相同的步长来对齐维度；
    其他情况下捷径为恒等映射，使主分支输出能够逐元素相加。
    """

    def __init__(self, previous: int, channels: int, stride: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(previous, channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm1d(channels))
        self.shortcut = nn.Identity() if previous == channels and stride == 1 else nn.Sequential(
            nn.Conv1d(previous, channels, 1, stride=stride, bias=False), nn.BatchNorm1d(channels))
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.main(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    """四个阶段各两个残差块，通道 32/64/128/256，共 18 个带权重的主路径层。

    卷积入口为 7 点卷积、步长 2，再进行三点最大池化。后三阶段的第一个
    残差块以步长 2 降采样；全局均值池化后由一个线性层输出类别分数。
    """

    def __init__(self, classes: int):
        super().__init__()
        layers = [nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False),
                  nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(3, stride=2, padding=1)]
        previous = 32
        for stage, channels in enumerate((32, 64, 128, 256)):
            layers.append(BasicBlock1D(previous, channels, 1 if stage == 0 else 2))
            layers.append(BasicBlock1D(channels, channels, 1))
            previous = channels
        layers.extend([nn.AdaptiveAvgPool1d(1), nn.Flatten()])
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(256, classes)

    def forward(self, x):
        return self.classifier(self.features(x))


def make_model(name: str, classes: int) -> nn.Module:
    """按冻结实验名称构造模型；不依据任何实验成绩自动改变结构。"""
    return {"cnn1d": CNN1D, "resnet1d": ResNet1D}[name](classes)
