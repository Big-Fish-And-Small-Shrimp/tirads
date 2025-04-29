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
    case1['image'] = train_image
    case1['label'] = labels
    cases1.append(case1)
for image_path, labels in test_data:
    case2 = {}
    test_image = io.imread(image_path)
    case2['image'] = test_image
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
train_label03 = nn.functional.one_hot(train_label3,num_classes=4).float()
train_label04 = nn.functional.one_hot(train_label4, num_classes=2).float()
train_label05 = nn.functional.one_hot(train_label5, num_classes=2).float()
train_label06 = nn.functional.one_hot(train_label6, num_classes=2).float()
train_label07 = nn.functional.one_hot(train_label7, num_classes=2).float()
train_label08 = nn.functional.one_hot(train_label8, num_classes=2).float()
test_label00 = nn.functional.one_hot(test_label0, num_classes=3).float()
test_label01 = nn.functional.one_hot(test_label1, num_classes=5).float()
test_label02 = nn.functional.one_hot(test_label2, num_classes=2).float()
test_label03 = nn.functional.one_hot(test_label3,num_classes=4).float()
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
    return lower_bound, upper_bound


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
0

net1 = MMOE(expert_dim=1000, num_experts=9, num_tasks=9, gate_dim=1000, classes=[3, 5, 2, 4, 2, 2, 2, 2, 2])
net2 = MMOE(expert_dim=1000, num_experts=9, num_tasks=9, gate_dim=1000, classes=[3, 5, 2, 4, 2, 2, 2, 2, 2])
net3 = MMOE(expert_dim=1000, num_experts=9, num_tasks=9, gate_dim=1000, classes=[3, 5, 2, 4, 2, 2, 2, 2, 2])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = TensorDataset(train_images1, train_label0, train_label01, train_label02, train_label03, train_label04, train_label05,train_label06, train_label07, train_label08)  #将数据进行封装
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
                images, label0, label1, label2, label3, label4, label5, label6, label7, label8 = images.to(device), label0.to(device), label1.to(device), label2.to(device),label3.to(device), label4.to(device), label5.to(device), label6.to(device), label7.to(device), label8.to(device)
                optimizer.zero_grad()
                outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8 = net(images)
                loss = criterion(outputs0, label0) * 0.04 + criterion(outputs1, label1) * 0.05 + criterion(outputs2, label2) * 0.01 + criterion(outputs3,label3)*0.04 + criterion(outputs4, label4) * 0.01 + criterion(outputs5, label5) * 0.04 + criterion(outputs6,label6) * 0.01 + criterion(outputs7,label7) * 0.05 + criterion(outputs8, label8)
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
test_dataset = TensorDataset(test_images1, test_label00, test_label01, test_label02, test_label03, test_label04, test_label05, test_label06, test_label07, labels_indices1)
test_loader = DataLoader(test_dataset, batch_size=36, shuffle=False)
net1.eval()
net2.eval()
total = 0
correct1 = 0
a = [0.5]
batch_size = 36
P, R, F1, acc, TPR, FPR, TP, TN, FP, FN ,S= torch.zeros(size=(len(a),)), torch.zeros(size=(len(a),)), torch.zeros(
    size=(len(a),)), torch.zeros(size=(len(a),)), torch.zeros(size=(len(a),)), torch.zeros(size=(len(a),)), torch.zeros(
    size=(len(a), math.ceil(len(labels_indices1) / batch_size))), torch.zeros(
    size=(len(a), math.ceil(len(labels_indices1) / batch_size))), torch.zeros(
    size=(len(a), math.ceil(len(labels_indices1) / batch_size))), torch.zeros(
    size=(len(a), math.ceil(len(labels_indices1) / batch_size))),torch.zeros(size=(len(a),))
tp, tn, fp, fn = torch.zeros(size=(len(a),)), torch.zeros(size=(len(a),)), torch.zeros(size=(len(a),)), torch.zeros(
    size=(len(a),))
esp = 1e-6

with torch.no_grad():
    for k in range(len(a)):
        j = 0
        y_preds, y_preds1,y_preds2, y_labels ,y_labels1,y_labels2=[],[], [], [], [], []
        for images, label0, label1, label2, label3, label4, label5, label6, label7, label8 in test_loader:
            images, label0, label1, label2, label3, label4, label5, label6, label7, label8 = images.to(device), label0.to(device), label1.to(device), label2.to(device), label3.to(device), label4.to(device), label5.to(device), label6.to(device), label7.to(device), label8.to(device)
            net1.load_state_dict(torch.load('1.pth'))
            net2.load_state_dict(torch.load("2.pth"))
            net3.load_state_dict(torch.load('3.pth'))
            net1.to(device)
            net2.to(device)
            net3.to(device)
            outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8 = net1(images)
            outputs9, outputs10, outputs11, outputs12, outputs13, outputs14, outputs15, outputs16, outputs17 = net2(images)
            outputs18, outputs19, outputs20, outputs21, outputs22, outputs23, outputs24, outputs25, outputs26 = net3(images)
            predict = torch.softmax(outputs8, dim=1)
            predictr = torch.softmax(outputs17, dim=1)
            predictrr = torch.softmax(outputs26,dim=1)
            predicted1 = torch.zeros(size=(len(label8),))
            score = predict[:, 1].cpu().numpy()
            score2 = predictr[:,1].cpu().numpy()
            score3 = predictrr[:,1].cpu().numpy()
            y_preds += predict.tolist()
            y_preds1 += predictr.tolist()
            y_preds2 += predictrr.tolist()
            y_labels += score.tolist()
            y_labels1 += score2.tolist()
            y_labels2 += score3.tolist()
        if k == 0:
            y_true = np.array(labels_indices1)
            num_ones = np.count_nonzero(y_true)
            num_zeros = len(y_true) - num_ones
            fpr1, tpr1, _ = roc_curve(y_true, y_labels)
            fpr2, tpr2, _ = roc_curve(y_true, y_labels1)
            fpr3, tpr3, _ = roc_curve(y_true, y_labels2)
            roc_auc1 = auc(fpr1, tpr1)
            roc_auc2 = auc(fpr2, tpr2)
            roc_auc3 = auc(fpr3, tpr3)
            ci_low1, ci_high1 = confidence_level(roc_auc1, num_ones, num_zeros)
            ci_low2, ci_high2 = confidence_level(roc_auc2, num_ones, num_zeros)
            ci_low3, ci_high3 = confidence_level(roc_auc3, num_ones, num_zeros)
            plt.figure()
            plt.plot(fpr1, tpr1, color='deeppink', lw=1, label=f'Single B-mode MMOE AUC=0.903 (95% CI=0.890~0.916)')
            plt.plot(fpr2, tpr2, color='indigo', lw=1, label=f'multimode MMOE AUC={roc_auc2:.3f} (95% CI={ci_low2:.3f}~{ci_high2:.3f})')
            plt.plot(fpr3, tpr3, color='skyblue', lw=1, label=f'Single color Doppler MMOE AUC={roc_auc3:.3f} (95% CI={ci_low3:.3f}~{ci_high3:.3f})')
            plt.plot([0, 1], [0, 1], color='lightblue', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('1-Specificity')
            plt.ylabel('Sensitivity')
            plt.title(f'Model contrast ROC')
            plt.legend(loc="lower right", fontsize=6)
            plt.savefig(f"Model_contrast.png")


# 将预测结果转换为numpy数组
y_labels1 = np.array(y_labels1)
y_labels2 = np.array(y_labels2)
y_labels = np.array(y_labels)
y_true = test_label8.numpy()

from scipy.stats import chi2

# 将概率预测转换为二分类标签（阈值0.5）
threshold = 0.5
y_pred1_labels = (np.array(y_labels1) >= threshold).astype(int)
y_pred2_labels = (np.array(y_labels2) >= threshold).astype(int)
y_pred3_labels = (np.array(y_labels) >= threshold).astype(int)


# 定义McNemar检验函数
def mcnemar_test(y_true, pred1, pred2):
    # 构建列联表
    b = np.sum((pred1 == y_true) & (pred2 != y_true))  # B: Model1对，Model2错
    c = np.sum((pred1 != y_true) & (pred2 == y_true))  # C: Model1错，Model2对

    # 计算统计量（带连续性校正）
    chi_squared = ((abs(b - c) - 1) ** 2) / (b + c + 1e-8)  # 防止除以零
    p_value = 1 - chi2.cdf(chi_squared, df=1)
    return chi_squared, p_value


# 定义要比较的模型对（标签列表）
model_pairs_mcnemar = [
    ('Single color Doppler MMOE', y_pred2_labels, 'Single B-mode MMOE', y_pred3_labels),
    ('multimode MMOE', y_pred1_labels,'Single B-mode MMOE', y_pred3_labels)
]
# 执行McNemar检验并打印结果
print("\nMcNemar检验结果（分类准确率比较）：")
for name1, pred1, name2, pred2 in model_pairs_mcnemar:
    chi_sq, p_val = mcnemar_test(y_true, pred1, pred2)
    print(f"\n{name1} vs {name2}:")
    print(f"卡方统计量 = {chi_sq:.3f}, P值 = {p_val:.5f}")
    if p_val < 0.05:
        print("结论：准确率差异显著（p < 0.05）")
    else:
        print("结论：准确率差异不显著")

# 定义评估函数
def evaluate_model(y_true, y_pred, threshold=0.5):
    y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
    TP = np.sum((y_pred_binary == 1) & (y_true == 1))
    TN = np.sum((y_pred_binary == 0) & (y_true == 0))
    FP = np.sum((y_pred_binary == 1) & (y_true == 0))
    FN = np.sum((y_pred_binary == 0) & (y_true == 1))

    sensitivity = TP / (TP + FN + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    accuracy = (TP + TN) / len(y_true)
    return TP, TN, FP, FN, sensitivity, specificity, accuracy

# 评估所有模型
models = [
    (y_labels1, "multimodal MMOE"),
    (y_labels2, "Single color Doppler MMOE"),
    (y_labels, "Single B-mode MMOE")
]

results = []
for y_pred, name in models:
    tp, tn, fp, fn, sens, spec, acc = evaluate_model(y_true, y_pred)
    results.append((name, tp, tn, fp, fn, sens, spec, acc))

# 打印结果
print("\n模型评估结果：")
print(
    f"{'模型':<15} | {'TP':<5} | {'TN':<5} | {'FP':<5} | {'FN':<5} | {'Sensitivity':<12} | {'Specificity':<12} | {'Accuracy':<10}")
print("-" * 80)
for name, tp, tn, fp, fn, sens, spec, acc in results:
    print(
        f"{name:<15} | {tp:<5} | {tn:<5} | {fp:<5} | {fn:<5} | {sens * 100:.2f}% | {spec * 100:.2f}% | {acc * 100:.2f}%")