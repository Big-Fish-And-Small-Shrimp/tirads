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
from PIL import Image, ImageEnhance, ImageOps
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
train_label8 = torch.tensor([case1['label'][8] for case1 in cases1], dtype=torch.long)
test_label8 = torch.tensor([case2['label'][8] for case2 in cases2], dtype=torch.long)
train_label08 = nn.functional.one_hot(train_label8, num_classes=2).float()
test_label08 = nn.functional.one_hot(test_label8, num_classes=2).float()
train_images1 = torch.tensor([case1['image'] for case1 in cases1], dtype=torch.float32)
test_images1 = torch.tensor([case2['image'] for case2 in cases2], dtype=torch.float32)
train_images1 = train_images1.permute(0, 3, 2, 1)
test_images1 = test_images1.permute(0, 3, 2, 1)


class VGG(nn.Module):
    def __init__(self, features, num_classes=2, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 8 * 5, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            BatchNorm2d = nn.BatchNorm2d(v)
            layers += [conv2d, BatchNorm2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vggx': [6, 'M', 12, 'M', 24, 24, 'M', 48, 48, 'M', 48, 48, 'M']
}



def vgg(model_name="vgg16", **kwargs):

    cfg = cfgs[model_name]
    model = VGG(make_features(cfg), **kwargs)

    return model


net = vgg(model_name='vgg16', num_classes=2, init_weights=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = TensorDataset(train_images1, train_label08)
train_loader = DataLoader(train_dataset, batch_size=42, shuffle=True)


def train(net, train_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,betas=(0.85,0.949),weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    min_loss = float('inf')
    min_loss_state_dict = None

    for epoch in range(num_epochs):
        total_loss = 0
        net.train()
        with tqdm(train_iter, desc=f'Epoch {epoch + 1}/{10}', unit='batch') as tepoch:
            for images, label8 in tepoch:
                images, label8 = images.to(device), label8.to(device)
                optimizer.zero_grad()
                outputs8 = net(images)
                loss = criterion(outputs8, label8)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                avg_loss = total_loss / len(train_iter)
                tepoch.set_postfix(loss=f'{avg_loss:.7f}')
            if avg_loss < min_loss:
                min_loss = avg_loss
                min_loss_state_dict = net.state_dict()
    torch.save(min_loss_state_dict, 'VGG16.pth')


train(net, train_loader, 100, lr=0.0001, device=device)
torch.cuda.empty_cache()
labels_indices1 = torch.argmax(test_label08, dim=1)
test_dataset = TensorDataset(test_images1, labels_indices1)
test_loader = DataLoader(test_dataset, batch_size=42, shuffle=False)
net.eval()
total = 0
correct1 = 0
a = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95]
batch_size = 42
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
    for k in range(len(a)):
        j = 0
        y_preds, y_labels = [], []
        for images, label8 in test_loader:
            images, label8 = images.to(device), label8.to(device)
            net.load_state_dict(torch.load('VGG16.pth'))
            net.to(device)
            outputs8 = net(images)
            predict = torch.softmax(outputs8, dim=1)
            predicted1 = torch.zeros(size=(len(label8),))
            score = predict[:, 1].cpu().numpy()
            y_preds += predict.tolist()
            y_labels += score.tolist()
            # for k in range(len(a)):
            q = 0
            for i in range(len(label8)):
                b = predict[i]
                c = a[k]
                predict1 = torch.max(b)
                index1 = torch.argmax(b)
                if predict1 <= c and index1 == 1:
                    predicted1[i] = 0
                elif index1 == 0:
                    predicted1[i] = 0
                else:
                    predicted1[i] = 1
                if predicted1[i] == 0 and label8[i] == 0:
                    TN[k, j] += 1
                if predicted1[i] == 1 and label8[i] == 1:
                    TP[k, j] += 1
                if predicted1[i] == 0 and label8[i] == 1:
                    FN[k, j] += 1
                if predicted1[i] == 1 and label8[i] == 0:
                    FP[k, j] += 1
                q = i + 1
            if q == len(label8):
                j = +1
                continue
            else:
                pass
        tp[k] = torch.sum(TP[k])
        fp[k] = torch.sum(FP[k])
        tn[k] = torch.sum(TN[k])
        fn[k] = torch.sum(FN[k])
        P[k] = tp[k] / (tp[k] + fp[k] + esp)
        R[k] = tp[k] / (tp[k] + fn[k] + esp)
        F1[k] = 2 * P[k] * R[k] / (P[k] + R[k] + esp)
        acc[k] = (tp[k] + tn[k]) / (tp[k] + tn[k] + fp[k] + fn[k] + esp)
        S[k] = tn[k] / (tn[k] + fp[k] + esp)
        TPR[k] = tp[k] / (tp[k] + fn[k] + esp)
        FPR[k] = fp[k] / (fp[k] + tn[k] + esp)
        print(
            f"阈值为：{a[k]},精度为：{P[k]},召回率为：{R[k]},F1值为：{F1[k]},准确率为：{acc[k]},真阳性率为:{TPR[k]},伪阴性率为:{FPR[k]},特异性率为:{S[k]}\n")
        if k == 0:
            y_true = np.array(labels_indices1)
            fpr, tpr, _ = roc_curve(y_true, y_labels)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver operating characteristic{a[k]}')
            plt.legend(loc="lower right")
            plt.savefig(f"VGG16_ROC曲线图.png")

