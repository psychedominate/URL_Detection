import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
web_data = pd.read_excel('D:/Desktop/experiment/data/website.xlsx')

# 选择特征列和标签列
feature_cols = ['length','split','special', 'rate','max_num', 'change']
X = web_data[feature_cols].values
y = web_data['label'].values

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 计算类别权重以处理不平衡数据
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = torch.FloatTensor(class_weights)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 创建数据加载器
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 定义DNN模型
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=2):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 防止过拟合

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

# 初始化模型
input_size = X_train.shape[1]
model = DNN(input_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    train_accuracy = correct / total
    train_accuracies.append(train_accuracy)

    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_labels)
            test_running_loss += test_loss.item() * test_inputs.size(0)

            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()

    test_epoch_loss = test_running_loss / len(test_loader.dataset)
    test_losses.append(test_epoch_loss)
    test_accuracy = test_correct / test_total
    test_accuracies.append(test_accuracy)

    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Train Loss: {epoch_loss:.4f}, '
          f'Train Accuracy: {train_accuracy:.4f}, '
          f'Test Loss: {test_epoch_loss:.4f}, '
          f'Test Accuracy: {test_accuracy:.4f}')

# 绘制训练和测试准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Train', color='blue')
plt.plot(test_accuracies, label='Test', color='orange')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# 绘制训练和测试损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train', color='blue')
plt.plot(test_losses, label='Test', color='orange')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# 评估模型
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    y_prob = torch.softmax(outputs, dim=1)[:, 1].numpy()
    y_pred = predicted.numpy()

# 输出评估指标
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)

print("DNN模型的评价指标")
print(f"测试集准确率: {accuracy:.4f}")
print(f"测试集精确率: {precision:.4f}")
print(f"测试集召回率: {recall:.4f}")
print(f"测试集F1分数: {f1:.4f}")

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["预测负类", "预测正类"],
            yticklabels=["实际负类", "实际正类"])
plt.title("混淆矩阵")
plt.tight_layout()
plt.show()

# 绘制ROC曲线
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
roc_auc = metrics.auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 保存预测结果
result_df = pd.DataFrame({
    '实际标签': y_test,
    '预测标签': y_pred,
    '预测概率': y_prob
})
# result_df.to_excel('DNN_prediction_results.xlsx', index=False)
# print("预测结果已保存至 DNN_prediction_results.xlsx")