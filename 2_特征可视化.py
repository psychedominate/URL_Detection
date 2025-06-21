import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # 引入Seaborn提升可视化效果

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_feature_distribution(df, feature_name):
    """
    绘制单个特征在黑白名单中的分布对比图（直方图 + 密度曲线）
    :param df: 包含所有特征和标签的数据集
    :param feature_name: 特征名称
    """
    # 分离黑白名单数据
    black_list = df[df['label'] == 1][feature_name]
    white_list = df[df['label'] == 0][feature_name]

    # 创建画布
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图和密度曲线（黑底白字为黑名单，绿底白字为白名单）
    sns.histplot(black_list, bins=20, kde=True, color='red', alpha=0.5, label='black_list (bad)')
    sns.histplot(white_list, bins=20, kde=True, color='green', alpha=0.5, label='white_list (good)')
    
    # 美化图表
    plt.title(f'特征分布对比：{feature_name}', fontsize=14)
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 读取整合后的特征文件（包含label=0和label=1）
    input_file = 'D:/Desktop/feature_output.xlsx'
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"错误：未找到文件 {input_file}")
        exit(1)
    
    # 确保label列存在且包含0和1
    if 'label' not in df.columns or set(df['label']) != {0, 1}:
        print("错误:数据中缺少有效标签(label应为0和1)")
        exit(1)
    
    # 定义特征名称列表（与特征提取结果一致）
    features = ['length', 'split', 'special', 'rate', 'max_num', 'change']
    
    # 依次绘制每个特征的分布对比图
    for feature in features:
        plot_feature_distribution(df, feature)