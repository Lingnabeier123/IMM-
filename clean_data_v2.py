import os
import numpy as np
import json
import pandas as pd

# 配置路径（请确保与你的环境一致）
JSON_PATH = r"d:\Program Files\py\k_prototype_data\dataset_info.json"
OUTPUT_BASE = r"d:\Program Files\py\k_prototype_data_cleaned"

def load_json_config(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def fill_missing_values_precision(df, categorical_indices):
    """
    更稳健的清洗逻辑：防止字符串被误转为 0.0
    """
    missing_marks = ['?', '', 'nan', 'NaN', 'None', 'null']
    df = df.replace(missing_marks, np.nan)
    
    for col_idx in range(df.shape[1]):
        col_name = df.columns[col_idx]
        if col_idx in categorical_indices:
            # 离散属性：确保是字符串并填充众数
            mode_series = df[col_name].mode()
            fill_val = str(mode_series[0]) if not mode_series.empty else "Unknown"
            df[col_name] = df[col_name].fillna(fill_val).astype(str)
        else:
            # 连续属性：强转数值，失败变 NaN，填充均值
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            mean_val = df[col_name].mean()
            df[col_name] = df[col_name].fillna(mean_val if pd.notna(mean_val) else 0)
    return df

def load_and_clean_data_v5(file_path, original_cat_idx, remove_fields, label_idx):
    print(f"  正在处理: {os.path.basename(file_path)}")
    
    try:
        # 1. 读取原始数据
        df = pd.read_csv(
            file_path, sep=None, engine='python', header=None, 
            encoding='utf-8', encoding_errors='ignore', quoting=3 
        )
    except Exception as e:
        raise ValueError(f"读取失败: {e}")

    df = df.dropna(how='all').reset_index(drop=True)
    total_cols = df.shape[1]

    # --- 核心逻辑：重新排列索引 ---
    
    # 确定特征列：排除【标签列】和【剔除列】
    # 注意：label_idx 可能是 0（如肝炎数据），也可能是中间某列
    feature_cols_original = [
        c for c in range(total_cols) 
        if c != label_idx and c not in remove_fields
    ]
    
    # 计算新 X 中离散属性的下标
    new_cat_idx = []
    for new_i, old_idx in enumerate(feature_cols_original):
        if old_idx in original_cat_idx:
            new_cat_idx.append(new_i)
            
    # 2. 提取数据
    X_df = df.iloc[:, feature_cols_original].copy()
    y_ser = df.iloc[:, label_idx].copy()
    
    # 重置列名以便清洗函数处理
    X_df.columns = range(X_df.shape[1])

    # 3. 清洗特征
    X_cleaned = fill_missing_values_precision(X_df, new_cat_idx)
    
    # 4. 标签处理（统一转为整数编码）
    y_str = y_ser.astype(str).str.strip()
    y_encoded, _ = pd.factorize(y_str)
    
    print(f"  [完成] 原始列:{total_cols} -> 特征:{X_cleaned.shape[1]} | 标签列原索引:{label_idx}")
    
    return X_cleaned.values, y_encoded, new_cat_idx

def save_data_final(X, y, output_path):
    output_path = output_path.rsplit('.', 1)[0] + ".data"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 将 y 拼接到最后一列
    combined = np.column_stack([X, y])
    # 使用 %s 保持混合数据格式
    np.savetxt(output_path, combined, delimiter=',', fmt='%s', encoding='utf-8')
    return output_path

def clean_all_datasets():
    if not os.path.exists(JSON_PATH):
        print(f"错误：找不到配置文件 {JSON_PATH}")
        return

    datasets = load_json_config(JSON_PATH)
    new_datasets_info = []

    for info in datasets:
        try:
            path = info["数据集路径"]
            original_cat_idx = info["离散属性列index"]
            remove_fields = info.get("remove_field", [])
            label_idx = info.get("标签列index", -1) # 获取新字段
            
            if label_idx == -1:
                print(f"  ⚠️ 跳过 {path}: 缺少 '标签列index'")
                continue

            # 执行 V5 版清洗逻辑
            X, y, updated_cat_idx = load_and_clean_data_v5(
                path, original_cat_idx, remove_fields, label_idx
            )
            
            # 生成输出路径
            rel_path = os.path.relpath(path, r"d:\Program Files\py\k_prototype_data")
            target_path = os.path.join(OUTPUT_BASE, rel_path)
            final_path = save_data_final(X, y, target_path)
            
            # 记录新的元数据
            new_datasets_info.append({
                "数据集名称": os.path.basename(os.path.dirname(path)),
                "数据集路径": final_path,
                "总属性个数": X.shape[1],  # 这里的总属性仅指特征数，方便 K-proto 调用
                "离散属性列index": updated_cat_idx,
                "标签列index": X.shape[1], # 现在标签永远在最后一列
                "样本量": len(X)
            })
            print(f"  ✅ 成功保存至: {final_path}")

        except Exception as e:
            print(f"  ❌ 失败 {info.get('数据集路径','Unknown')}: {e}")
            import traceback
            traceback.print_exc()

    # 保存修正后的 JSON 供后续测试脚本直接读取
    new_json_path = os.path.join(OUTPUT_BASE, "dataset_info.json")
    with open(new_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_datasets_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✨ 全部处理完成！")
    print(f"新配置文件已生成: {new_json_path}")

if __name__ == "__main__":
    clean_all_datasets()