import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

def train_decision_tree(input_file):
    # 读取特征数据
    try:
        df = pd.read_excel(input_file)
        print(f"成功加载数据: {input_file}，共{len(df)}条记录")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 检查数据完整性
    required_columns = ['length', 'split', 'special', 'rate', 'max_num', 'change', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"数据缺少必要列: {', '.join(missing_columns)}")
        return
    
    # 特征和标签分离
    X = df[['length', 'split', 'special', 'rate', 'max_num', 'change']]
    y = df['label']
    
    # 数据标准化（决策树不需要标准化，但可作为其他模型对比）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集: {len(X_train)}条，测试集: {len(X_test)}条")
    print(f"标签分布 - 训练集: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
    print(f"标签分布 - 测试集: 0={sum(y_test==0)}, 1={sum(y_test==1)}")
    
    # 初始化决策树模型
    model = DecisionTreeClassifier(
        criterion='gini',         # 评价指标：基尼不纯度
        max_depth=5,              # 树的最大深度
        min_samples_split=10,     # 最小分裂样本数
        min_samples_leaf=5,       # 最小叶子节点样本数
        random_state=42
    )
    
    # 模型训练
    model.fit(X_train, y_train)
    
    # 交叉验证（5折）
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"\n交叉验证结果 (F1): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # 在测试集上的预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # 正类概率
    
    # 输出评价指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("随机森林模型的评价指标")
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"测试集精确率: {precision:.4f}")
    print(f"测试集召回率: {recall:.4f}")
    print(f"测试集F1分数: {f1:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        '特征': X.columns,
        '重要性': model.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    print("\n=== 特征重要性 ===")
    print(feature_importance)
    
    # 绘制混淆矩阵热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['预测白名单', '预测黑名单'], 
                yticklabels=['实际白名单', '实际黑名单'])
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('实际标签')
    plt.show()

    # 绘制特征重要性柱状图
    plt.figure(figsize=(8, 6))
    sns.barplot(x='重要性', y='特征', data=feature_importance)
    plt.title('特征重要性')
    plt.show()

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.show()

    # 绘制预测概率分布
    plt.figure(figsize=(8, 6))
    sns.histplot(y_prob[y_test==0], bins=20, color='green', alpha=0.5, label='白名单')
    sns.histplot(y_prob[y_test==1], bins=20, color='red', alpha=0.5, label='黑名单')
    plt.title('预测概率分布')
    plt.xlabel('预测为黑名单的概率')
    plt.ylabel('样本数量')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, X_test, y_test

if __name__ == '__main__':
    input_file = 'D:/Desktop/experiment/data/website.xlsx'
    model, X_test, y_test = train_decision_tree(input_file)