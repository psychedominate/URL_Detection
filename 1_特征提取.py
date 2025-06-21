import pandas as pd
import re
import openpyxl

def clean_url(url):
    """清洗URL,移除非法字符"""
    # 移除不可见的ASCII控制字符
    url = re.sub(r'[\x00-\x1F\x7F]', '', url)
    # 移除换行符、制表符等
    url = url.replace('\n', '').replace('\t', '').replace('\r', '')
    return url

def extract_url_features(df):
    # 清洗URL列
    df['URL'] = df['URL'].apply(clean_url)
    
    # 1. 计算URL长度(不含点号)
    df['length'] = df['URL'].apply(lambda x: len(x) - x.count('.'))
    # 2. 点号数量
    df['split'] = df['URL'].apply(lambda x: x.count('.'))
    # 3. 特殊字符数量
    df['special'] = df['URL'].apply(lambda x: sum(1 for char in x if not char.isalnum()))
    # 4. 数字比例
    df['rate'] = df['URL'].apply(lambda x: sum(1 for char in x if char.isdigit()) / len(x) if len(x) != 0 else 0)
    # 5. 子域名最大连续数字长度
    df['max_num'] = df['URL'].apply(lambda x: max(sum(1 for char in sub if char.isdigit()) for sub in x.split('.')) if x else 0)
    # 6. 字符类型变化频率
    df['change'] = df['URL'].apply(lambda x: sum(1 for i in range(len(x)-1) if x[i].isdigit() != x[i+1].isdigit()) / len(x) if len(x) != 0 else 0)
    
    # 转换label:good→0，bad→1
    df['label'] = df['label'].map({'good': 0, 'bad': 1})
    return df

def save_features(df, output_file):
    try:
        df.to_excel(output_file, index=False)
        print(f"特征已保存至 {output_file}，包含 {len(df)} 条数据")
    except openpyxl.utils.exceptions.IllegalCharacterError as e:
        error_url = re.search(r"'(.*?)'", str(e)).group(1)
        print(f"检测到非法字符 URL:{error_url}，已自动清洗并重新保存")
        df['URL'] = df['URL'].apply(lambda u: u if u != error_url else clean_url(error_url))
        save_features(df, output_file)
    except Exception as e:
        print(f"保存失败:{str(e)},尝试保存为CSV格式")
        csv_output = output_file.replace('.xlsx', '.csv')
        df.to_csv(csv_output, index=False)
        print(f"已保存为CSV文件:{csv_output}")

if __name__ == '__main__':
    input_file = 'D:/Desktop/data.csv'
    try:
        raw_data = pd.read_csv(input_file, header=0, names=['URL', 'label'])
    except FileNotFoundError:
        print(f"错误:未找到文件 {input_file}")
        exit(1)
    
    if not set(raw_data['label']).issubset({'good', 'bad'}):
        print("错误:label必须为'good'或'bad'")
        exit(1)
    
    feature_df = extract_url_features(raw_data)
    output_file = 'D:/Desktop/feature_output.xlsx'
    save_features(feature_df, output_file)
    
    print("\n标签分布:")
    print(f"good(0):{len(feature_df[feature_df['label']==0])} 条")
    print(f"bad(1):{len(feature_df[feature_df['label']==1])} 条")