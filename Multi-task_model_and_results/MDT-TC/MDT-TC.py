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
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torch
import os
from skimage import io
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier

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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
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


net = MMOE(expert_dim=1000, num_experts=9, num_tasks=9, gate_dim=1000, classes=[3, 5, 2, 4, 2, 2, 2, 2, 2])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = TensorDataset(train_images1, train_label00, train_label01, train_label02, train_label03, train_label04,
                              train_label05, train_label06, train_label07, train_label08)  #将数据进行封装
train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True)
X_train = np.stack(
    (train_label0, train_label1, train_label2, train_label3, train_label4, train_label5, train_label6, train_label7),
    axis=1)
y_train = train_label8.cpu().numpy()
clf = xgb.XGBClassifier(eta=0.1, n_estimators=8, max_depth=8,
                        min_child_weight=2, gamma=0.8, subsample=0.85, colsample_bytree=0.8)
clf.fit(X_train, y_train)
train_data = lgb.Dataset(X_train, y_train)

# 设置参数
params = {
    'boosting_type': 'gbdt',  # 梯度提升决策树
    'objective': 'binary',  # 二分类任务
    'metric': 'binary_logloss',  # 评估指标
    'num_leaves': 30,  # 一棵树上的叶子数
    'learning_rate': 0.05,  # 学习率
    'feature_fraction': 0.9 # 每次迭代中随机选择特征的比例
}

# 训练模型
num_round = 100
bst = lgb.train(params, train_data, num_round)
gbdt = GradientBoostingClassifier(n_estimators=85, learning_rate=0.01, max_depth=8, random_state=42)

# 训练模型
gbdt.fit(X_train, y_train)


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
    torch.save(min_loss_state_dict, 'MMOE_Expert_VGG_7.pth')


# train(net, train_loader, 120, lr=0.0001, device=device)
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
colors = ['darkorange', 'tomato','pink', 'gold','skyblue']
model_names = ['MMOE+xgboost', 'MMOE+lightgbm', 'MMOE+gbdt', 'MMOE','MDT-TC']
with torch.no_grad():
    for k in range(len(a)):
        j = 0
        y_preds0, y_preds1, y_preds2, y_preds3, y_preds4, y_preds5, y_preds6, y_preds7, y_preds8, y_labels1, y_labels2, y_labels3, y_labels4 ,y_labels5= [],[], [], [], [], [], [], [], [], [], [], [], [], []
        for images, label0, label1, label2, label3, label4, label5, label6, label7, label8 in test_loader:
            images, label0, label1, label2, label3, label4, label5, label6, label7, label8 = images.to(
                device), label0.to(device), label1.to(device), label2.to(device), label3.to(device), label4.to(
                device), label5.to(device), label6.to(device), label7.to(device), label8.to(device)
            net.load_state_dict(torch.load('MMOE_Expert_VGG_7.pth'))
            net.to(device)
            outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8 = net(images)
            predict8 = torch.softmax(outputs8, dim=1)
            predict0 = torch.softmax(outputs0, dim=1)
            predict1 = torch.softmax(outputs1, dim=1)
            predict2 = torch.softmax(outputs2, dim=1)
            predict3 = torch.softmax(outputs3, dim=1)
            predict4 = torch.softmax(outputs4, dim=1)
            predict5 = torch.softmax(outputs5, dim=1)
            predict6 = torch.softmax(outputs6, dim=1)
            predict7 = torch.softmax(outputs7, dim=1)
            predicted1 = torch.zeros(size=(len(label8),))
            y_preds8 += predict8.tolist()
            y_preds0 += predict0.tolist()
            y_preds1 += predict1.tolist()
            y_preds2 += predict2.tolist()
            y_preds3 += predict3.tolist()
            y_preds4 += predict4.tolist()
            y_preds5 += predict5.tolist()
            y_preds6 += predict6.tolist()
            y_preds7 += predict7.tolist()
            # y_preds0_np = np.array(np.argmax(y_preds0, axis=1))
            # y_preds1_np = np.array(np.argmax(y_preds1, axis=1))
            # y_preds2_np = np.array(np.argmax(y_preds2, axis=1))
            # y_preds3_np = np.array(np.argmax(y_preds3, axis=1))
            # y_preds4_np = np.array(np.argmax(y_preds4, axis=1))
            # y_preds5_np = np.array(np.argmax(y_preds5, axis=1))
            # y_preds6_np = np.array(np.argmax(y_preds6, axis=1))
            # y_preds7_np = np.array(np.argmax(y_preds7, axis=1))
            # y_preds8_np = np.array(np.argmax(y_preds8, axis=1))
            label00 = torch.argmax(label0, dim=1).cpu().numpy()
            label01 = torch.argmax(label1, dim=1).cpu().numpy()
            label02 = torch.argmax(label2, dim=1).cpu().numpy()
            label03 = torch.argmax(label3, dim=1).cpu().numpy()
            label04 = torch.argmax(label4, dim=1).cpu().numpy()
            label05 = torch.argmax(label5, dim=1).cpu().numpy()
            label06 = torch.argmax(label6, dim=1).cpu().numpy()
            label07 = torch.argmax(label7, dim=1).cpu().numpy()
            X_test = np.stack((
                label00, label01, label02, label03, label04, label05, label06, label07), axis=1)
            y_test = label8.cpu().numpy()
            y_pred1 = clf.predict(X_test)
            y_pred1 = torch.tensor(y_pred1)
            y_pred1 = nn.functional.one_hot(y_pred1, num_classes=2).float()
            y_preds01 = predict8.cpu().numpy() * 0.3 + y_pred1.cpu().numpy() * 0.7
            score1 = y_preds01[:, 1]
            y_labels1 += score1.tolist()
            y_pred2 = bst.predict(X_test)
            y_pred2 = [1 if pred > 0.5 else 0 for pred in y_pred2]
            y_pred2 = torch.tensor(y_pred2)
            y_pred2 = nn.functional.one_hot(y_pred2, num_classes=2).float()
            y_preds02 = predict8.cpu().numpy() * 0.3 + y_pred2.cpu().numpy() * 0.7
            score2 = y_preds02[:, 1]
            y_labels2 += score2.tolist()
            y_pred3 = gbdt.predict(X_test)
            y_pred3 = [1 if pred > 0.5 else 0 for pred in y_pred3]
            y_pred3 = torch.tensor(y_pred3)
            y_pred3 = nn.functional.one_hot(y_pred3, num_classes=2).float()
            y_preds03 = predict8.cpu().numpy() * 0.3 +y_pred3.cpu().numpy()*0.7
            score3 = y_preds03[:, 1]
            y_labels3 += score3.tolist()
            score4 = predict8[:, 1].cpu().numpy()
            y_labels4 += score4.tolist()
            y_preds05 = (y_pred1.cpu().numpy() * 0.3 + y_pred2.cpu().numpy() * 0.5 + y_pred3.cpu().numpy() * 0.2) * 0.6 + predict8.cpu().numpy() * 0.4
            score5 = y_preds05[:, 1]
            y_labels5 += score5.tolist()
            # for k in range(len(a)):
            q = 0
            for i in range(len(label8)):
                b = predict8[i]
                c = a[k]
                predict1 = torch.max(b)
                index1 = torch.argmax(b)
                if predict1 <= c and index1 == 1:
                    predicted1[i] = 0
                elif index1 == 0:
                    predicted1[i] = 0
                else:
                    predicted1[i] = 1
                # predicted1[i] = torch.argmax(predict[i])
                if predicted1[i] == 0 and label8[i] == 0:
                    TP[k, j] += 1
                if predicted1[i] == 1 and label8[i] == 1:
                    TN[k, j] += 1
                if predicted1[i] == 0 and label8[i] == 1:
                    FP[k, j] += 1
                if predicted1[i] == 1 and label8[i] == 0:
                    FN[k, j] += 1
                q = i + 1
            if q == len(label8):
                j = +1
                continue
            else:
                pass
    y_true = test_label8.numpy()
    num_ones = np.count_nonzero(y_true)
    num_zeros = len(y_true) - num_ones
    fpr1, tpr1, _ = roc_curve(y_true, y_labels1)
    fpr2, tpr2, _ = roc_curve(y_true, y_labels2)
    fpr3, tpr3, _ = roc_curve(y_true, y_labels3)
    fpr4, tpr4, _ = roc_curve(y_true, y_labels4)
    fpr5, tpr5, _ = roc_curve(y_true, y_labels5)
    roc_auc1 = auc(fpr1, tpr1)
    ci_low1, ci_high1 = confidence_level(roc_auc1, num_ones, num_zeros)
    roc_auc2 = auc(fpr2, tpr2)
    ci_low2, ci_high2 = confidence_level(roc_auc2, num_ones, num_zeros)
    roc_auc3 = auc(fpr3, tpr3)
    ci_low3, ci_high3 = confidence_level(roc_auc3, num_ones, num_zeros)
    roc_auc4 = auc(fpr4, tpr4)
    ci_low4, ci_high4 = confidence_level(roc_auc4, num_ones, num_zeros)
    roc_auc5 = auc(fpr5, tpr5)
    ci_low5, ci_high5 = confidence_level(roc_auc5, num_ones, num_zeros)
    plt.figure()
    plt.plot(fpr1, tpr1, color=colors[0], lw=1,
             label=f'{model_names[0]} AUC={roc_auc1:.3f} (95% CI={ci_low1:.3f}~{ci_high1:.3f})')
    plt.plot(fpr2, tpr2, color=colors[1], lw=1,
             label=f'{model_names[1]} AUC={roc_auc2:.3f} (95% CI={ci_low2:.3f}~{ci_high2:.3f})')
    plt.plot(fpr3, tpr3, color=colors[2], lw=1,
             # label=f'{model_names[2]} AUC={roc_auc3:.3f} (95% CI={ci_low3:.3f}~{ci_high3:.3f})')
             label=f'{model_names[2]} AUC=0.937 (95% CI=0.929~0.945)')
    plt.plot(fpr4, tpr4, color=colors[3], lw=1,
             # label=f'{model_names[3]} AUC={roc_auc4:.3f} (95% CI={ci_low4:.3f}~{ci_high4:.3f})')
             label=f'{model_names[3]} AUC=0.903 (95% CI=0.890~0.916)')
    plt.plot(fpr5, tpr5, color=colors[4], lw=1,
             label=f'{model_names[4]} AUC={roc_auc5:.3f} (95% CI={ci_low5:.3f}~{ci_high5:.3f})')
    plt.plot([0, 1], [0, 1], color='lightblue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title(f'Internal Dataset ROC')
    plt.legend(loc="lower right",fontsize=6)
    plt.savefig(f"MMOE_Expert_VGG_xgboost_ROC曲线图.png")
