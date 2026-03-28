import json
import numpy as np
import os

JSON_PATH = r"d:\Program Files\py\k_prototype_data_cleaned\dataset_info.json"
OUTPUT_JSON = r"d:\Program Files\py\k_prototype_data_cleaned\dataset_info_fixed.json"
UNIQUE_RATIO_THRESHOLD=0.05
UNIQUE_COUNT_THRESHOLD=5
RANGE_THRESHOLD=20


def load_data(path):

    data = np.genfromtxt(
        path,
        delimiter=",",
        dtype=str,
        filling_values=""
    )

    return data[:, :-1]


def is_float(v):

    try:
        float(v)
        return True
    except:
        return False


def detect_column_type(col):

    n = len(col)

    # 是否包含字符串
    numeric_flags = [is_float(v) for v in col]

    if not all(numeric_flags):
        return "categorical"

    # 转成 float
    values = np.array(col, dtype=float)

    unique_count = len(np.unique(values))
    unique_ratio = unique_count / n

    # 唯一值比例判断
    if unique_ratio < UNIQUE_RATIO_THRESHOLD:
        return "categorical"

    # 整数离散判断
    if np.all(values == values.astype(int)):
        if unique_count < UNIQUE_COUNT_THRESHOLD and (values.max() - values.min()) < RANGE_THRESHOLD:
            return "categorical"

    return "numerical"


def detect_dataset(dataset_info):

    path = dataset_info["数据集路径"]

    print("\n检测:", path)

    X = load_data(path)

    n_features = X.shape[1]

    categorical_indices = []

    for i in range(n_features):

        col = X[:, i]

        col_type = detect_column_type(col)

        print(f"列 {i} -> {col_type}")

        if col_type == "categorical":
            categorical_indices.append(i)

    return categorical_indices


def main():

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        datasets = json.load(f)

    new_configs = []

    for dataset in datasets:

        try:

            categorical_indices = detect_dataset(dataset)

            dataset["离散属性列index"] = categorical_indices

            new_configs.append(dataset)

        except Exception as e:

            print("检测失败:", e)

            new_configs.append(dataset)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(new_configs, f, indent=2, ensure_ascii=False)

    print("\n修复完成")
    print("新JSON:", OUTPUT_JSON)


if __name__ == "__main__":
    main()