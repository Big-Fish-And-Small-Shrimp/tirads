import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.preprocessing import LabelEncoder
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.preprocessing import LabelEncoder

base_dir = "/path/Internal_dataset/All_data"
base_dirs = "/path/Internal_dataset/All_data1"
train_label_dir = os.path.join(base_dir, 'train_labels')
train_dir = os.path.join(base_dir, 'train_images')
test_label_dir = os.path.join(base_dirs, 'test_labels')
test_dir = os.path.join(base_dirs, 'test_images')
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
    'feature_fraction': 0.8  # 每次迭代中随机选择特征的比例
}

# 训练模型
num_round = 100
bst = lgb.train(params, train_data, num_round)
gbdt = GradientBoostingClassifier(n_estimators=70, learning_rate=0.01, max_depth=8, random_state=42)

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
colors = ['deeppink', 'indigo', 'turquoise', 'salmon']
model_names = ['MMOE+xgboost', 'MMOE+lightgbm', 'MMOE+GBDT', 'MMOE']
net.load_state_dict(torch.load('1.pth', map_location=device))
net.to(device)
net.eval()
A1 = ['Cystic or almost completely cystic or Spongiform', 'Mixed cystic and solid', 'Solid or almost completety solid']
A2 = ['Anechoic', 'Hyperechoic or isoechoic', 'Mixed cystic and solid echo', 'Hypoechoic', 'Very hypoechoic']
A3 = ['Wider-than-tall', 'Tall-than-wide']
A4 = ['Smooth', 'III-defined', 'Lobulated or irregular', 'Extra-thyroidal extension']
A5 = ['No', 'Yes']
A6 = ['No', 'Yes']
A7 = ['No', 'Yes']
A8 = ['No', 'Yes']
A9 = ['benign', 'malignancy']
all_outputs = []
with torch.no_grad():
    y_preds0, y_preds1, y_preds2, y_preds3, y_preds4, y_preds5, y_preds6, y_preds7, y_preds8= [], [], [], [], [], [], [], [],[]
    y_preds9 = np.array([])
    for images, label0, label1, label2, label3, label4, label5, label6, label7, label8 in test_loader:
        images, label0, label1, label2, label3, label4, label5, label6, label7, label8 = images.to(
            device), label0.to(device), label1.to(device), label2.to(device), label3.to(device), label4.to(
            device), label5.to(device), label6.to(device), label7.to(device), label8.to(device)
        net.load_state_dict(torch.load('1.pth'))
        net.to(device)
        outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8 = net(images)


        def get_predictions(output, classes):
            probs = torch.softmax(output, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            return [classes[p] for p in preds]

        def value(output0, output1, output2, output3, output4, output5, output6, output7, output8):
            probs0 = torch.softmax(output0, dim=1).cpu().numpy()
            probs1 = torch.softmax(output1, dim=1).cpu().numpy()
            probs2 = torch.softmax(output2, dim=1).cpu().numpy()
            probs3 = torch.softmax(output3, dim=1).cpu().numpy()
            probs4 = torch.softmax(output4, dim=1).cpu().numpy()
            probs5 = torch.softmax(output5, dim=1).cpu().numpy()
            probs6 = torch.softmax(output6, dim=1).cpu().numpy()
            probs7 = torch.softmax(output7, dim=1).cpu().numpy()
            probs8 = torch.softmax(output8, dim=1).cpu().numpy()
            preds0 = np.argmax(probs0, axis=1)
            preds1 = np.argmax(probs1, axis=1)
            preds2 = np.argmax(probs2, axis=1)
            preds3 = np.argmax(probs3, axis=1)
            preds4 = np.argmax(probs4, axis=1)
            preds5 = np.argmax(probs5, axis=1)
            preds6 = np.argmax(probs6, axis=1)
            preds7 = np.argmax(probs7, axis=1)
            preds8 = np.argmax(probs8, axis=1)
            values=0.53*preds0+0.82*preds1+0.36*preds2+0.27*preds3+0.49*preds4+0.19*preds5+0.16*preds6+0.10*preds7+preds8
            return values

        # 获取所有预测结果
        y_preds0 += get_predictions(outputs0, A1)
        y_preds1 += get_predictions(outputs1, A2)
        y_preds2 += get_predictions(outputs2, A3)
        y_preds3 += get_predictions(outputs3, A4)
        y_preds4 += get_predictions(outputs4, A5)
        y_preds5 += get_predictions(outputs5, A6)
        y_preds6 += get_predictions(outputs6, A7)
        y_preds7 += get_predictions(outputs7, A8)
        y_preds8 += get_predictions(outputs8, A9)
        batch_values = value(outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8)
        y_preds9 = np.concatenate((y_preds9, batch_values))

# 合并所有batch的数据
data = {
    'AI_Diagnosis': y_preds8,
    'COM': y_preds0,
    'ECH': y_preds1,
    'SHA': y_preds2,
    'MAR': y_preds3,
    'ELCA': y_preds4,
    'MAC': y_preds5,
    'PCA': y_preds6,
    'PEF': y_preds7,
    'value': y_preds9
}

# 创建DataFrame
df = pd.DataFrame(data)
numeric_cols = ["value"]
categorical_cols = [col for col in df.columns if col not in numeric_cols+ ["AI_Diagnosis"]]
le = LabelEncoder()
df_encoded = df[numeric_cols].copy()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df[col])

# 运行UMAP
umap = UMAP(n_components=2, random_state=42)
embedding = umap.fit_transform(df_encoded)

# 3. 绘制多面板UMAP图
features_to_plot = categorical_cols # 8个特征
n_cols = 2
n_rows = int(np.ceil(len(features_to_plot) / n_cols))

plt.figure(figsize=(20, 5 * n_rows))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1],
        hue=df[feature], palette="viridis", alpha=0.7
    )
    plt.title(f"UMAP colored by {feature}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# 4. 计算质心距离 (AI诊断 vs 各特征)
def calculate_centroid_distances(embedding, df, target_col="AI_Diagnosis"):
    results = {}
    target_centroid = embedding[df[target_col] == "malignancy"].mean(axis=0)  # 以恶性诊断为参考

    for feature in features_to_plot:
        # 计算每个特征类别的质心
        categories = df[feature].unique()
        distances = []
        for cat in categories:
            mask = (df[feature] == cat)
            cat_centroid = embedding[mask].mean(axis=0)
            distance = np.linalg.norm(cat_centroid - target_centroid)
            distances.append(distance)
        mean_distance = np.mean(distances)
        results[feature] = mean_distance

    return pd.Series(results).sort_values(ascending=False)


distance_results = calculate_centroid_distances(embedding, df)
print("\nMean distances from AI Diagnosis centroid:")
print(distance_results.round(3))
