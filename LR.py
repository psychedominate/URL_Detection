import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
# D:/Desktop/experiment/data/website.xlsx
# D:/Desktop/feature_output.xlsx
web_data = pd.read_excel('D:/Desktop/feature_output.xlsx')  

# 选择特征列和标签列
feature_cols = ['length', 'split', 'special', 'rate', 'max_num', 'change']
X = web_data[feature_cols]
y = web_data['label']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 交叉验证评估模型
model = LogisticRegression(random_state=42, class_weight='balanced')
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"交叉验证准确率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # 获取正类的概率

# 输出评估指标
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)

print("逻辑回归模型的评价指标")
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

# 绘制特征重要性
plt.figure(figsize=(10, 6))
coef = model.coef_[0]
feature_importance = pd.Series(coef, index=feature_cols).sort_values(ascending=False)
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title("特征重要性（逻辑回归系数）")
plt.xlabel("系数值")
plt.tight_layout()
plt.show()

# 绘制ROC曲线
fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
roc_auc = metrics.auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率 (FPR)')
plt.ylabel('真正例率 (TPR)')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 保存预测结果
result_df = pd.DataFrame({
    '实际标签': y_test,
    '预测标签': y_pred,
    '预测概率': y_prob
})
# result_df.to_excel('LR_prediction_results.xlsx', index=False)
# print("预测结果已保存至 LR_prediction_results.xlsx")