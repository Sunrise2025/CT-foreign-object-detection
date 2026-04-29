# Scaling module is adapted from here
# https://github.com/ahendriksen/msd_pytorch/blob/master/msd_pytorch/msd_model.py
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

from .import InputError


def scaling_module(num_channels):
    c = nn.Conv2d(num_channels, num_channels, 1)
    c.bias.requires_grad = False
    c.weight.requires_grad = False
    scaling_module_set_scale(c, 1.0)
    scaling_module_set_bias(c, 0.0)

    return c


def scaling_module_set_scale(sm, s):
    c_out, c_in = sm.weight.shape[:2]
    assert c_out == c_in
    sm.weight.data.zero_()
    for i in range(c_out):
        sm.weight.data[i, i] = s


def scaling_module_set_bias(sm, bias):
    sm.bias.data.fill_(bias)


class NNmodel(nn.Module):
    def  __init__(self, c_in, c_out, nn_type):
        super(NNmodel, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.nn_type = nn_type
        self.scaling = scaling_module(c_in)
        self.mean = None
        self.std = None
        self.set_classification_nn(c_in, c_out, nn_type)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        #交叉熵损失函数CrossEntropyLoss， 确保损失函数在 GPU 上
        self.init_optimizer(self.net)
        self.to('cuda')
        # 在模型设置完成后，将整个网络移到 GPU
        # self.criterion.cuda()
    # 将损失函数（criterion）移动到 GPU

    def set_classification_nn(self, c_in, c_out, nn_type, sample_type=None):
        if nn_type == "resnet50":
            self.classifier = models.resnet50(weights=None)
            self.classifier.conv1 = nn.Conv2d(c_in, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # 修改 ResNet50 的第一层卷积层，将其输入通道数修改为 `c_in`（通常为 1 或 3，取决于输入图像的通道数），
            # 将卷积核大小设置为 (7, 7)，步长为 (2, 2)，并且填充为 (3, 3) 来保持输入尺寸。
            # `bias=False` 表示不使用偏置项。
            self.classifier.fc = nn.Linear(in_features=2048, out_features=c_out, bias=True)
            # 修改 ResNet50 的最后一个全连接层，调整输入特征数量为 2048（ResNet50 的输出特征数），
            # 输出特征数量修改为 `c_out`，即分类的类别数3。
            # `bias=True` 表示使用偏置项。
        elif nn_type == "efficientnetb4":
            self.classifier = models.efficientnet_b4(weights=None)
            self.classifier.features[0][0] = nn.Conv2d(c_in, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.classifier.classifier[1] = nn.Linear(in_features=1792, out_features=c_out, bias=True)
        else:
            raise InputError("Unknown sample type. Got {}".format(sample_type))
        self.net = nn.Sequential(self.scaling, self.classifier)  # 将网络移动到 GPU
        self.net.cuda()
        self.init_optimizer(self.classifier)

    def init_optimizer(self, trainable_classifier):
        self.optimizer = torch.optim.SGD(trainable_classifier.parameters(), lr=0.001, momentum=0.9)
#学习率被设定为 0.001，并且使用的是随机梯度下降（SGD）优化器，带有动量参数 momentum=0.9。因此，模型的学习率默认是固定的，并不会在训练过程中自动调整。


    def set_normalization(self, dl):
        mean = square = 0

        for (inp, tg) in dl:
            mean += inp.mean()
            square += inp.pow(2).mean()

        mean /= len(dl)
        square /= len(dl)

        std = np.sqrt(square - mean ** 2)

        scaling_module_set_scale(self.scaling, 1 / std)
        scaling_module_set_bias(self.scaling, -mean / std)

        self.mean = mean
        self.std = std
    def _classify(self, inp):
        inp = inp.to('cuda')
        out = self.net(inp)
        return out

    def _forward(self, inp, tg):
        inp = inp.to('cuda')
        tg = tg.to('cuda').long()  # 确保目标标签是 LongTensor 类型
        out = self.net(inp)
        loss = self.criterion(out, tg)
        return loss

    def _learn(self, inp, tg):
        loss = self._forward(inp, tg)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def _train(self, dl):
        self.train()  # Set model to training mode
        avg_loss = 0
        for (inp, tg) in dl:
            inp = inp.to('cuda')
            tg = tg.to('cuda')
            avg_loss += self._learn(inp, tg).item()
        avg_loss /= len(dl)
        return avg_loss

    def _validate(self, dl):
        self.eval()  # Set model to evaluation mode
        avg_loss = 0
        for (inp, tg) in dl:
            inp = inp.to('cuda')
            tg = tg.to('cuda')
            avg_loss += self._forward(inp, tg).item()
        avg_loss /= len(dl)
        return avg_loss

    def save(self, path, epoch, batch_size):
        init_seed = torch.initial_seed()
        state = {
            "epoch": int(epoch),
            "random_seed": int(init_seed),
            "batch_size": int(batch_size),
            "state_dict": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # ✅ 新增（关键）
            "mean": float(self.mean) if self.mean is not None else None,
            "std": float(self.std) if self.std is not None else None,
        }
        torch.save(state, path)

    def test(self, dl):
        self.eval()  # Set model to evaluation mode
        conf_mat = np.zeros((self.c_out, self.c_out))
        y_true = []
        y_pred = []

        with torch.no_grad():
            for (inp, tg) in dl:
                out = self._classify(inp)
                for i in range(out.size(0)):
                    out_max = torch.argmax(out[i, :], dim=0)
                    gt = tg[i].item()
                    prediction = out_max.item()
                    y_true.append(gt)
                    y_pred.append(prediction)
                    conf_mat[gt, prediction] += 1
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        return (conf_mat, y_pred, y_true)

    def load(self, path):
        state = torch.load(path, weights_only=True)
        self.net.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.to('cuda')  # Ensure the entire model is moved to GPU
        epoch = state["epoch"]

        return epoch

    def classify(self, inp):
        inp = inp.to('cuda')
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            out = self._classify(inp)
        return out


