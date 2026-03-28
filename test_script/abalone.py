import pandas as pd
import numpy as np
import os

def bin_abalone_age(input_path, output_path):
    print(f"正在处理: {os.path.basename(input_path)}")
    
    # 1. 读取数据 (假设此时数据已经是没有表头的格式)
    # Abalone 原始 8 列特征 + 1 列 Rings (索引为 8)
    df = pd.read_csv(input_path, header=None, sep=',')
    
    # 获取 Rings 列 (最后一列)
    rings = df.iloc[:, -1].astype(int)
    
    # 2. 定义分箱逻辑
    # 1-8 环 -> 0 (Infant)
    # 9-10 环 -> 1 (Young)
    # 11-15 环 -> 2 (Adult)
    # 16+ 环 -> 3 (Old)
    bins = [0, 8, 10, 15, 100] 
    labels = [0, 1, 2, 3]
    
    age_groups = pd.cut(rings, bins=bins, labels=labels)
    
    # 3. 新增一列到数据中
    # 我们把原来的 Rings 保留，在最后再加一列作为聚类的 Ground Truth
    df['age_group'] = age_groups
    
    # 4. 保存为新的 data 文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, header=False, index=False)
    
    print(f"✅ 处理完成！")
    print(f"   - 原始行数: {len(df)}")
    print(f"   - 新文件保存至: {output_path}")
    print(f"   - 分布情况: \n{df['age_group'].value_counts().sort_index()}")

if __name__ == "__main__":
    # 设定你的路径
    input_file = r"d:\Program Files\py\k_prototype_data\abalone\abalone.data"
    output_file = r"d:\Program Files\py\k_prototype_data\abalone\abalone_binned.data"
    
    bin_abalone_age(input_file, output_file)