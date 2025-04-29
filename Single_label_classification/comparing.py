import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable
import torch
from torch import Tensor
from torch.nn import functional as F
from sklearn.metrics import roc_curve, auc
import math
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch.utils.data import TensorDataset, DataLoader
from skimage import io
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import os
from skimage import io
from PIL import Image, ImageEnhance, ImageOps
from other_code.images_preprocessing import *
from torchvision.models import ResNet
from scipy import stats

base_dir = "/path/Internal_dataset/All_data"
train_label_dir = os.path.join(base_dir, 'train_labels')
train_dir = os.path.join(base_dir, 'train_images')
test_label_dir = os.path.join(base_dir, 'test_labels')
test_dir = os.path.join(base_dir, 'test_images')
train_label_files = [f for f in os.listdir(train_label_dir) if f.endswith('.txt')]
train_images = [f for f in os.listdir(train_dir) if f.endswith('.png')]
test_label_files = [f for f in os.listdir(test_label_dir) if f.endswith('.txt')]
test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
matched_data1 = []
matched_data2 = []
for label_file in train_label_files:
    prefix = label_file.rsplit('.', 1)[0]
    image_file = f"{prefix}.png"
    if image_file in train_images:
        labels = []
        with open(os.path.join(train_label_dir, label_file), 'r') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line and stripped_line.isdigit() or '.' in stripped_line:
                    labels.append(float(stripped_line))
        matched_data1.append((os.path.join(train_dir, image_file), labels))
for label_file in test_label_files:
    prefix = label_file.rsplit('.', 1)[0]
    image_file = f"{prefix}.png"
    if image_file in test_images:
        labels = []
        with open(os.path.join(test_label_dir, label_file), 'r') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line and stripped_line.isdigit() or '.' in stripped_line:
                    labels.append(float(stripped_line))
        matched_data2.append((os.path.join(test_dir, image_file), labels))

train_data = matched_data1[:]
test_data = matched_data2[:]
cases1 = []
cases2 = []
for image_path, labels in train_data:
    case1 = {}
    train_image = io.imread(image_path)
    adjusted_train_image = adjust_image(train_image, exposure_factor=0.8, saturation_factor=1.1, brightness_factor=1.1,
                                        contrast_factor=1.1, shadow_factor=1.1, hue_factor=35, kelvin_temperature=8000)
    case1['image'] = adjusted_train_image
    case1['images'] = train_image
    case1['label'] = labels
    cases1.append(case1)
for image_path, labels in test_data:
    case2 = {}
    test_image = io.imread(image_path)
    adjusted_test_image = adjust_image(test_image, exposure_factor=0.8, saturation_factor=1.1, brightness_factor=1.1,
                                       contrast_factor=1.1, shadow_factor=1.1, hue_factor=35, kelvin_temperature=8000)
    case2['image'] = adjusted_test_image
    case2['images'] = test_image
    case2['label'] = labels
    cases2.append(case2)
train_label0 = torch.tensor([case1['label'][0] for case1 in cases1], dtype=torch.long)
test_label0 = torch.tensor([case2['label'][0] for case2 in cases2], dtype=torch.long)
train_label1 = torch.tensor([case1['label'][1] for case1 in cases1], dtype=torch.long)
test_label1 = torch.tensor([case2['label'][1] for case2 in cases2], dtype=torch.long)
train_label2 = torch.tensor([case1['label'][2] for case1 in cases1], dtype=torch.long)
test_label2 = torch.tensor([case2['label'][2] for case2 in cases2], dtype=torch.long)
train_label3 = torch.tensor([case1['label'][3] for case1 in cases1], dtype=torch.long)
test_label3 = torch.tensor([case2['label'][3] for case2 in cases2], dtype=torch.long)
train_label4 = torch.tensor([case1['label'][4] for case1 in cases1], dtype=torch.long)
test_label4 = torch.tensor([case2['label'][4] for case2 in cases2], dtype=torch.long)
train_label5 = torch.tensor([case1['label'][5] for case1 in cases1], dtype=torch.long)
test_label5 = torch.tensor([case2['label'][5] for case2 in cases2], dtype=torch.long)
train_label6 = torch.tensor([case1['label'][6] for case1 in cases1], dtype=torch.long)
test_label6 = torch.tensor([case2['label'][6] for case2 in cases2], dtype=torch.long)
train_label7 = torch.tensor([case1['label'][7] for case1 in cases1], dtype=torch.long)
test_label7 = torch.tensor([case2['label'][7] for case2 in cases2], dtype=torch.long)
train_label8 = torch.tensor([case1['label'][8] for case1 in cases1], dtype=torch.long)
test_label8 = torch.tensor([case2['label'][8] for case2 in cases2], dtype=torch.long)
train_label00 = nn.functional.one_hot(train_label0, num_classes=3).float()
train_label01 = nn.functional.one_hot(train_label1, num_classes=5).float()
train_label02 = nn.functional.one_hot(train_label2, num_classes=2).float()
train_label03 = nn.functional.one_hot(train_label3, num_classes=4).float()
train_label04 = nn.functional.one_hot(train_label4, num_classes=2).float()
train_label05 = nn.functional.one_hot(train_label5, num_classes=2).float()
train_label06 = nn.functional.one_hot(train_label6, num_classes=2).float()
train_label07 = nn.functional.one_hot(train_label7, num_classes=2).float()
train_label08 = nn.functional.one_hot(train_label8, num_classes=2).float()
test_label00 = nn.functional.one_hot(test_label0, num_classes=3).float()
test_label01 = nn.functional.one_hot(test_label1, num_classes=5).float()
test_label02 = nn.functional.one_hot(test_label2, num_classes=2).float()
test_label03 = nn.functional.one_hot(test_label3, num_classes=4).float()
test_label04 = nn.functional.one_hot(test_label4, num_classes=2).float()
test_label05 = nn.functional.one_hot(test_label5, num_classes=2).float()
test_label06 = nn.functional.one_hot(test_label6, num_classes=2).float()
test_label07 = nn.functional.one_hot(test_label7, num_classes=2).float()
test_label08 = nn.functional.one_hot(test_label8, num_classes=2).float()
train_images1 = torch.tensor([case1['image'] for case1 in cases1], dtype=torch.float32)
test_images1 = torch.tensor([case2['image'] for case2 in cases2], dtype=torch.float32)
train_images1 = train_images1.permute(0, 3, 2, 1)
test_images1 = test_images1.permute(0, 3, 2, 1)
train_images2 = torch.tensor([case1['images'] for case1 in cases1], dtype=torch.float32) / 255
test_images2 = torch.tensor([case2['images'] for case2 in cases2], dtype=torch.float32) / 255
train_images2 = train_images2.permute(0, 3, 2, 1)
test_images2 = test_images2.permute(0, 3, 2, 1)


class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeAndExcitation, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


# 深度可分离卷积块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super(DepthwiseSeparableConv, self).__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.ReLU(inplace=True)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels * expansion, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.ReLU(inplace=True)
        )
        self.se = SqueezeAndExcitation(in_channels * expansion)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.skip = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.skip:
            return self.pointwise(self.se(self.depthwise(self.expand(x)))) + x
        else:
            return self.pointwise(self.se(self.depthwise(self.expand(x))))


# EfficientNet-B0
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB0, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(32, 16, stride=1, expansion=1),
            DepthwiseSeparableConv(16, 24, stride=2, expansion=6),
            DepthwiseSeparableConv(24, 40, stride=2, expansion=6),
            DepthwiseSeparableConv(40, 80, stride=2, expansion=6),
            DepthwiseSeparableConv(80, 112, stride=1, expansion=6),
            DepthwiseSeparableConv(112, 192, stride=2, expansion=6),
            DepthwiseSeparableConv(192, 320, stride=1, expansion=6)
        )
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


#   Inception 结构模板
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        #   将输入特征矩阵分别输入到四个分支
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        #   将输出放入一个列表中
        outputs = [branch1, branch2, branch3, branch4]
        #   通过torch.cat合并四个输出，合并维度为1，即按照通道维度合并
        return torch.cat(outputs, 1)


#   InceptionAux 辅助分类器模板
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(1536, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        #   aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14

        x = self.averagePool(x)
        #   aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4

        x = self.conv(x)
        #   N x 128 x 4 x 4

        #   特征矩阵展平，从channel维度开始展平
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        #   N x 2048

        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        #   N x 1024

        x = self.fc2(x)
        #   N x num_classes
        return x


#   定义GoogLeNet网络
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weight=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)  # ceil_mode=True 计算为小数时，向上取整
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        #   辅助分类器
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        #   AdaptiveAvgPool2d 自适应全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:  # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class residual_block(nn.Module):
    def __init__(self, input_channels, output_channels, conv1_1=True, strides=1):
        super(residual_block, self).__init__()
        self.conv1_1 = conv1_1
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)

    def forward(self, x):
        Y = self.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv1_1 is True:
            x = self.conv3(x)
        Y += x
        return self.relu(Y)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, is_b2=False):
        super(Res_block, self).__init__()
        self.inchannel = in_channels
        self.outhannel = out_channels
        self.num_blocks = num_blocks
        self.is_b2 = is_b2
        res = []
        for i in range(num_blocks):
            if i == 0 and self.is_b2 is not True:
                res.append(residual_block(in_channels, out_channels, conv1_1=True, strides=2))
            else:
                res.append(residual_block(out_channels, out_channels, conv1_1=False, strides=1))
        self.seq = nn.Sequential(
            *res
        )

    def forward(self, x):
        return self.seq(x)


class Resnet_18(nn.Module):
    def __init__(self, inputchannel):
        super(Resnet_18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(inputchannel, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(num_features=64, eps=1e-5),
            nn.MaxPool2d(3, stride=1, padding=1)
        )
        self.net = nn.Sequential(
            self.b1,
            Res_block(64, 64, 2, is_b2=True),
            Res_block(64, 128, 2),
            Res_block(128, 256, 2),

            Res_block(256, 512, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.net(x)

    def show_net(self):
        return self.net


class BasicBlock(nn.Module):
    """搭建BasicBlock模块"""
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # 使用BN层是不需要使用bias的，bias最后会抵消掉
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN层, BN层放在conv层和relu层中间使用
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    # 前向传播
    def forward(self, X):
        identity = X
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.downsample is not None:  # 保证原始输入X的size与主分支卷积后的输出size叠加时维度相同
            identity = self.downsample(X)

        return self.relu(Y + identity)


class BottleNeck(nn.Module):
    """搭建BottleNeck模块"""
    # BottleNeck模块最终输出out_channel是Residual模块输入in_channel的size的4倍(Residual模块输入为64)，shortcut分支in_channel
    # 为Residual的输入64，因此需要在shortcut分支上将Residual模块的in_channel扩张4倍，使之与原始输入图片X的size一致
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BottleNeck, self).__init__()

        # 默认原始输入为256，经过7x7层和3x3层之后BottleNeck的输入降至64
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN层, BN层放在conv层和relu层中间使用
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)  # Residual中第三层out_channel扩张到in_channel的4倍

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    # 前向传播
    def forward(self, X):
        identity = X

        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))

        if self.downsample is not None:  # 保证原始输入X的size与主分支卷积后的输出size叠加时维度相同
            identity = self.downsample(X)

        return self.relu(Y + identity)


class ResNet1(nn.Module):
    """搭建ResNet-layer通用框架"""

    # num_classes是训练集的分类个数，include_top是在ResNet的基础上搭建更加复杂的网络时用到，此处用不到
    def __init__(self, residual, num_residuals, num_classes=1000, include_top=True):
        super(ResNet1, self).__init__()

        self.out_channel = 64  # 输出通道数(即卷积核个数)，会生成与设定的输出通道数相同的卷积核个数
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, self.out_channel, kernel_size=7, stride=2, padding=3,
                               bias=False)  # 3表示输入特征图像的RGB通道数为3，即图片数据的输入通道为3
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self.residual_block(residual, 64, num_residuals[0])
        self.conv3 = self.residual_block(residual, 128, num_residuals[1], stride=2)
        self.conv4 = self.residual_block(residual, 256, num_residuals[2], stride=2)
        self.conv5 = self.residual_block(residual, 512, num_residuals[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output_size = (1, 1)
            self.fc = nn.Linear(512 * residual.expansion, num_classes)

        # 对conv层进行初始化操作
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def residual_block(self, residual, channel, num_residuals, stride=1):
        downsample = None

        # 用在每个conv_x组块的第一层的shortcut分支上，此时上个conv_x输出out_channel与本conv_x所要求的输入in_channel通道数不同，
        # 所以用downsample调整进行升维，使输出out_channel调整到本conv_x后续处理所要求的维度。
        # 同时stride=2进行下采样减小尺寸size，(注：conv2时没有进行下采样，conv3-5进行下采样，size=56、28、14、7)。
        if stride != 1 or self.out_channel != channel * residual.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.out_channel, channel * residual.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * residual.expansion))

        block = []  # block列表保存某个conv_x组块里for循环生成的所有层
        # 添加每一个conv_x组块里的第一层，第一层决定此组块是否需要下采样(后续层不需要)
        block.append(residual(self.out_channel, channel, downsample=downsample, stride=stride))
        self.out_channel = channel * residual.expansion  # 输出通道out_channel扩张

        for _ in range(1, num_residuals):
            block.append(residual(self.out_channel, channel))

        # 非关键字参数的特征是一个星号*加上参数名，比如*number，定义后，number可以接收任意数量的参数，并将它们储存在一个tuple中
        return nn.Sequential(*block)

    # 前向传播
    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.maxpool(Y)
        Y = self.conv5(self.conv4(self.conv3(self.conv2(Y))))

        if self.include_top:
            Y = self.avgpool(Y)
            Y = torch.flatten(Y, 1)
            Y = self.fc(Y)

        return Y


# 构建ResNet-34模型
def resnet34(num_classes=1000, include_top=True):
    return ResNet1(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


# 构建ResNet-50模型
def resnet50(num_classes=1000, include_top=True):
    return ResNet1(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，输入BCHW -> 输出 B*C*1*1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 可以看到channel得被reduction整除，否则可能出问题
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 得到B*C*1*1,然后转成B*C，才能送入到FC层中。
        y = self.fc(y).view(b, c, 1, 1)  # 得到B*C的向量，C个值就表示C个通道的权重。把B*C变为B*C*1*1是为了与四维的x运算。
        return x * y.expand_as(x)  # 先把B*C*1*1变成B*C*H*W大小，其中每个通道上的H*W个值都相等。*表示对应位置相乘。


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        # 参数列表里的 * 星号，标志着位置参数的就此终结，之后的那些参数，都只能以关键字形式来指定。
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        # 参数列表里的 * 星号，标志着位置参数的就此终结，之后的那些参数，都只能以关键字形式来指定。
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes=1_000):
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes=1_000):
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes=1_000, pretrained=False):
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class VGG(nn.Module):  # 定义VGG网络
    def __init__(self, features, num_classes=2, init_weights=False):  # num_classed 为分类的个数
        super(VGG, self).__init__()
        self.features = features  # 特征提取层通过make_features 创建
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # dropout 随机失活
            nn.Linear(512 * 8 * 5, 2048),  # 特征提取最后的size是（512*7*7）
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),  # 原论文中，线性层全都是是4096，分为1000类
            nn.ReLU(True),  # 最后的分类不能有dropout
            nn.Linear(2048, num_classes)
        )
        if init_weights:  # 初始化权重
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # 特征提取层
        x = torch.flatten(x, start_dim=1)  # ddata维度为（batch_size,512，7，7），从第二个维度开始flatten
        x = self.classifier(x)  # 分类层
        return x

    def _initialize_weights(self):  # 随机初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)  # 初始化权重
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # bias 为 0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)  # 高斯初始化线性层参数
                nn.init.constant_(m.bias, 0)  # bias 为0


def make_features(cfg: list):  # 生成特征提取层，就是VGG前面的卷积池化层
    layers = []  # 保存每一层网络结构
    in_channels = 3  # 输入图片的深度channels，起始输入是RGB 3 通道的
    for v in cfg:  # 遍历配置列表 cfgs
        if v == "M":  # M 代表最大池化层，VGG中max pooling的size=2，stride = 2
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # M 代表最大池化层
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            BatchNorm2d = nn.BatchNorm2d(v)  # 数字代表卷积核的个数==输出的channels
            layers += [conv2d, BatchNorm2d, nn.ReLU(inplace=True)]  # 添加卷积层
            in_channels = v  # 输出的channels == 下次输入的channels
    return nn.Sequential(*layers)  # 解引用，将大的list里面的小list拿出来


# 特征提取层的 网络结构参数
cfgs = {  # 建立网络的字典文件，对应的key可以生成对应网络结构参数的value值
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # 数字代表卷积核的个数，M代表池化层
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vggx': [6, 'M', 12, 'M', 24, 24, 'M', 48, 48, 'M', 48, 48, 'M']
}


# 定义生成VGG 网络函数
def vgg(model_name="vgg16", **kwargs):  # 创建VGG网络，常用的为 VGG16 结构,如果不指定分类个数，默认是10

    cfg = cfgs[model_name]  # 先定义特征提取层的结构
    model = VGG(make_features(cfg), **kwargs)  # 将cfgs里面某个参数传给make_features，并且生成VGG net

    return model


class MMOE(nn.Module):
    def __init__(self, expert_dim, num_experts, num_tasks, gate_dim=32, classes=None):
        super(MMOE, self).__init__()
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.gate_dim = gate_dim
        self.classes = classes

        def make_layer(block, in_channels, out_channels, blocks, stride=1):
            downsample = None
            if stride != 1 or in_channels != out_channels * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )

            layers = []
            layers.append(block(in_channels, out_channels, stride, downsample))
            in_channels = out_channels * block.expansion
            for _ in range(1, blocks):
                layers.append(block(in_channels, out_channels))
            return nn.Sequential(*layers)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 12, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(12),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(12, 12, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(12),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(12, 24, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(24, 24, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Conv2d(24, 48, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Conv2d(48, 96, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Flatten(),
                nn.Linear(1920, 1500),
                nn.ReLU(),
                nn.Linear(1500, expert_dim)
            ) for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 12, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(12),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(12, 12, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(12),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(12, 24, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(24, 24, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Conv2d(24, 48, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Conv2d(48, 96, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Flatten(),
                nn.Linear(1920, 1500),
                nn.ReLU(),
                nn.Linear(1500, gate_dim),
                nn.ReLU(),
                nn.Linear(gate_dim, num_experts, bias=False)
            ) for _ in range(num_tasks)
        ])
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, 800),
                nn.ReLU(),
                nn.Linear(800, classes[i])
            ) for i in range(num_tasks)
        ])

    def forward(self, x_input):
        expert_outputs = [expert(x_input) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=2)
        gate_outputs = []
        for gate in self.gates:
            gate_scores = gate(x_input)
            gate_scores = F.softmax(gate_scores, dim=-1)
            weighted_experts = torch.bmm(expert_outputs, gate_scores.unsqueeze(2)).squeeze(2)
            gate_outputs.append(weighted_experts)
        task_outputs = [head(gate_output) for head, gate_output in zip(self.task_heads, gate_outputs)]
        return task_outputs


def confidence_level(auc_value, positive_samples, negative_samples):
    Q1 = auc_value / (2 - auc_value)
    Q2 = (2 * auc_value ** 2) / (1 + auc_value)
    standard_error = math.sqrt((auc_value * (1 - auc_value) +
                                (positive_samples - 1) * (Q1 - auc_value ** 2) +
                                (negative_samples - 1) * (Q2 - auc_value ** 2))
                               / (positive_samples * negative_samples))
    z_score = 1.96
    lower_bound = auc_value - z_score * standard_error
    upper_bound = auc_value + z_score * standard_error
    return lower_bound,upper_bound


net1 = EfficientNetB0(num_classes=2)
net2 = GoogLeNet(num_classes=2, aux_logits=False, init_weight=True)
net3 = Resnet_18(inputchannel=3)
net4 = resnet50(num_classes=2)
net5 = se_resnet18(num_classes=2)
net6 = se_resnet50(num_classes=2)
net7 = vgg(model_name='vgg11', num_classes=2, init_weights=False)
net8 = vgg(model_name='vgg16', num_classes=2, init_weights=False)
net9 = MMOE(expert_dim=1000, num_experts=9, num_tasks=9, gate_dim=1000, classes=[3, 5, 2, 4, 2, 2, 2, 2, 2])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels_indices1 = torch.argmax(test_label08, dim=1)
test_dataset = TensorDataset(test_images1, labels_indices1)
test_loader = DataLoader(test_dataset, batch_size=42, shuffle=False)
test_dataset1 = TensorDataset(test_images2, test_label00, test_label01, test_label02, test_label03, test_label04,
                              test_label05, test_label06, test_label07, labels_indices1)
test_loader1 = DataLoader(test_dataset1, batch_size=36, shuffle=False)
model_classes = [net1, net2, net3, net4, net5, net6, net7, net8, net9]
state_dict_paths = ['EfficientNet.pth', 'GoogleNet.pth', 'ResNet18.pth', 'ResNet50.pth', 'SE_ResNet18.pth',
                    'SE_ResNet50.pth', 'VGG11.pth', 'VGG16.pth', 'MMOE_Expert_VGG_7.pth']
model_names = ['EfficientNet', 'GoogleNet', 'ResNet18', 'ResNet50', 'SE_ResNet18', 'SE_ResNet50', 'VGG11', 'VGG16',
               'MMOE']
colors = ['crimson', 'orange', 'gold', 'mediumseagreen', 'steelblue', 'mediumpurple', 'blue', 'red', 'darkred']

net1.eval()
with torch.no_grad():
    plt.figure()
    for k in range(len(colors)):
        y_preds, y_labels, = [], []
        if k == 8:
            for images, label0, label1, label2, label3, label4, label5, label6, label7, label8 in test_loader1:
                images, label0, label1, label2, label3, label4, label5, label6, label7, label8 = images.to(
                    device), label0.to(device), label1.to(device), label2.to(device), label3.to(device), label4.to(
                    device), label5.to(device), label6.to(device), label7.to(device), label8.to(device)
                net = model_classes[k]
                net.load_state_dict(torch.load(state_dict_paths[k]))
                net.to(device)
                outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8 = net(images)
                predict = torch.softmax(outputs8, dim=1)
                predicted1 = torch.zeros(size=(len(label8),))
                score = predict[:, 1].cpu().numpy()
                y_preds += predict.tolist()
                y_labels += score.tolist()
        else:
            for images, label8 in test_loader:
                images, label8 = images.to(device), label8.to(device)
                net = model_classes[k]
                net.load_state_dict(torch.load(state_dict_paths[k]))
                net.to(device)
                outputs8 = net(images)
                predict = torch.softmax(outputs8, dim=1)
                predicted1 = torch.zeros(size=(len(label8),))
                score = predict[:, 1].cpu().numpy()
                y_preds += predict.tolist()
                y_labels += score.tolist()
        y_true = np.array(labels_indices1)
        num_ones = np.count_nonzero(y_true)
        num_zeros = len(y_true) - num_ones
        fpr1, tpr1, _ = roc_curve(y_true, y_labels)
        fpr = fpr1
        tpr = tpr1
        roc_auc = auc(fpr, tpr)
        ci_low, ci_high = confidence_level(roc_auc,num_ones,num_zeros)
        plt.plot(fpr, tpr, color=colors[k], lw=1, label=f'{model_names[k]} AUC={roc_auc:.3f} (95% CI={ci_low:.3f}~{ci_high:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title(f'ROC')
    plt.legend(loc="lower right",fontsize=6)
    plt.savefig("各模型ROC曲线与AUC对比图")
