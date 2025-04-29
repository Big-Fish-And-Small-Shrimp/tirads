import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import torch
import math
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch.utils.data import TensorDataset, DataLoader
from skimage import io
import pandas as pd
from other_code.images_preprocessing import *

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
    case1['label'] = labels
    cases1.append(case1)
for image_path, labels in test_data:
    case2 = {}
    test_image = io.imread(image_path)
    adjusted_test_image = adjust_image(test_image, exposure_factor=0.8, saturation_factor=1.1, brightness_factor=1.1,
                                       contrast_factor=1.1, shadow_factor=1.1, hue_factor=35, kelvin_temperature=8000)
    case2['image'] = adjusted_test_image
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
train_images1 = torch.tensor([case1['image'] for case1 in cases1], dtype=torch.float32) / 255
test_images1 = torch.tensor([case2['image'] for case2 in cases2], dtype=torch.float32) / 255
train_images1 = train_images1.permute(0, 3, 2, 1)
test_images1 = test_images1.permute(0, 3, 2, 1)


class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


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


net = MMOE(expert_dim=1200, num_experts=9, num_tasks=9, gate_dim=1200, classes=[3, 5, 2, 4, 2, 2, 2, 2, 2])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = TensorDataset(train_images1, train_label0, train_label01, train_label02, train_label03, train_label04,
                              train_label05, train_label06, train_label07, train_label08)  #将数据进行封装
train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True)


def train(net, train_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    min_loss = float('inf')
    min_loss_state_dict = None

    for epoch in range(num_epochs):
        total_loss = 0
        net.train()
        with tqdm(train_iter, desc=f'Epoch {epoch + 1}/{10}', unit='batch') as tepoch:
            for images, label0, label1, label2, label3, label4, label5, label6, label7, label8 in tepoch:
                images, label0, label1, label2, label3, label4, label5, label6, label7, label8 = images.to(
                    device), label0.to(device), label1.to(device), label2.to(device), label3.to(device), label4.to(
                    device), label5.to(device), label6.to(device), label7.to(device), label8.to(device)
                optimizer.zero_grad()
                outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8 = net(images)
                loss = criterion(outputs0, label0) * 0.04 + criterion(outputs1, label1) * 0.05 + criterion(outputs2,
                                                                                                           label2) * 0.01 + criterion(
                    outputs3, label3) * 0.04 + criterion(outputs4, label4) * 0.01 + criterion(outputs5,
                                                                                              label5) * 0.04 + criterion(
                    outputs6, label6) * 0.01 + criterion(outputs7, label7) * 0.05 + criterion(outputs8, label8)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                avg_loss = total_loss / len(train_iter)
                tepoch.set_postfix(loss=f'{avg_loss:.7f}')
            if avg_loss < min_loss:
                min_loss = avg_loss
                min_loss_state_dict = net.state_dict()
    torch.save(min_loss_state_dict, 'MMOE_Expert_VGG1.pth')


# train(net, train_loader, 0, lr=0.0001, device=device)
torch.cuda.empty_cache()
labels_indices1 = torch.argmax(test_label08, dim=1)
test_dataset = TensorDataset(test_images1, test_label00, test_label01, test_label02, test_label03, test_label04,
                             test_label05, test_label06, test_label07, labels_indices1)
test_loader = DataLoader(test_dataset, batch_size=36, shuffle=False)
net.eval()
total = 0
correct1 = 0
a = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
batch_size = 36
P, R, F1, acc, TPR, FPR, TP, TN, FP, FN, S = torch.zeros(size=(len(a),)), torch.zeros(size=(len(a),)), torch.zeros(
    size=(len(a),)), torch.zeros(size=(len(a),)), torch.zeros(size=(len(a),)), torch.zeros(size=(len(a),)), torch.zeros(
    size=(len(a), math.ceil(len(labels_indices1) / batch_size))), torch.zeros(
    size=(len(a), math.ceil(len(labels_indices1) / batch_size))), torch.zeros(
    size=(len(a), math.ceil(len(labels_indices1) / batch_size))), torch.zeros(
    size=(len(a), math.ceil(len(labels_indices1) / batch_size))), torch.zeros(size=(len(a),))
tp, tn, fp, fn = torch.zeros(size=(len(a),)), torch.zeros(size=(len(a),)), torch.zeros(size=(len(a),)), torch.zeros(
    size=(len(a),))
esp = 1e-6

with torch.no_grad():
    for k in range(len(a)):  # 假设a是某种迭代次数或数据集分割的索引
        t, f, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0
        for images, label0, label1, label2, label3, label4, label5, label6, label7, label8 in test_loader:
            images, label0, label1, label2, label3, label4, label5, label6, label7, label8 = images.to(
                device), label0.to(device), label1.to(device), label2.to(device), label3.to(device), label4.to(
                device), label5.to(device), label6.to(device), label7.to(device), label8.to(device)
            net.load_state_dict(torch.load('MMOE_Expert_VGG1.pth'))  # 通常，加载模型状态字典应该在循环外部
            net.to(device)
            outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8 = net(images)
            predict = torch.softmax(outputs7, dim=1)
            predict0 = torch.argmax(predict, dim=1)
            label00 = torch.argmax(label7, dim=1)

            # 遍历批次中的每个样本
            for i in range(len(label0)):
                true_label = label00[i].item()  # 将张量转换为整数
                pred_label = predict0[i].item()
                if pred_label == true_label:
                    t += 1
                else:
                    f += 1
                # if (pred_label == 2 and true_label == 2) or (pred_label == 3 and true_label == 3):
                if pred_label == 1 and true_label == 1:
                    tp += 1
                # elif (pred_label == 0 or pred_label == 1) and true_label in [2,3]:
                elif pred_label == 0 and true_label == 1:
                    fn += 1
                # elif pred_label == true_label and true_label in [0, 1]:
                elif pred_label == 0 and true_label == 0:
                    tn += 1  # 注意这里的逻辑可能需要调整，因为tp和tn的定义可能重叠
                else:
                    fp += 1

        accuracy = t / (t + f)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 避免除以零
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"acc为{accuracy}, tn{tn}, tp{tp}, fn{fn}, fp{fp}, re为{recall}, sp为{specificity}")
        #     predict = torch.softmax(outputs0, dim=1)
        #     predicted1 = torch.zeros(size=(len(label8),))
        #     score = predict[:, 1].cpu().numpy()
        #     y_preds += predict.tolist()
        #     y_labels += score.tolist()
        #     # for k in range(len(a)):
        #     q = 0
        #     label07 = torch.argmax(label0, dim=1)
        #     for i in range(len(label0)):
        #         b = predict[i]
        #         predict1 = torch.max(b)
        #         index1 = torch.argmax(b)
        #         if index1 == 1:
        #             predicted1[i] = 0
        #         elif index1 == 0:
        #             predicted1[i] = 0
        #         elif index1 == 2:
        #             predicted1[i] = 0
        #         # elif index1 == 3:
        #         #     predicted1[i] = 1
        #         # elif index1 == 4:
        #         #     predicted1[i] = 1
        #         # predicted1[i] = torch.argmax(predict[i])
        #         if predicted1[i] == 0 and label07[i] == 0:
        #             TP[k, j] += 1
        #         if predicted1[i] == 1 and label07[i] == 1:
        #             TN[k, j] += 1
        #         if predicted1[i] == 0 and label07[i] == 1:
        #             FP[k, j] += 1
        #         if predicted1[i] == 1 and label07[i] == 0:
        #             FN[k, j] += 1
        #         q = i + 1
        #     if q == len(label0):
        #         j = +1
        #         continue
        #     else:
        #         pass
        # tp[k] = torch.sum(TP[k])
        # fp[k] = torch.sum(FP[k])
        # tn[k] = torch.sum(TN[k])
        # fn[k] = torch.sum(FN[k])
        # P[k] = tp[k] / (tp[k] + fp[k] + esp)
        # R[k] = tp[k] / (tp[k] + fn[k] + esp)
        # S[k] = tn[k] / (tn[k] + fp[k] + esp)
        # F1[k] = 2 * P[k] * R[k] / (P[k] + R[k] + esp)
        # acc[k] = (tp[k] + tn[k]) / (tp[k] + tn[k] + fp[k] + fn[k] + esp)
        # TPR[k] = tp[k] / (tp[k] + fn[k] + esp)
        # FPR[k] = fp[k] / (fp[k] + tn[k] + esp)
        # print(f"阈值为：{a[k]},TP为：{tp[k]},TN为：{tn[k]},FP为：{fp[k]},FN为:{fn[k]},精度为：{P[k]},召回率为：{R[k]},F1值为：{F1[k]},准确率为：{acc[k]},真阳性率为:{TPR[k]},伪阴性率为:{FPR[k]},特异性率为:{S[k]}\n")
