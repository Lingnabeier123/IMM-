import os
import json
import numpy as np
import random

JSON_PATH = r"d:\Program Files\py\k_prototype_data_cleaned\dataset_info.json"


def load_json_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(file_path):

    data = np.genfromtxt(
        file_path,
        delimiter=",",
        dtype=str,
        filling_values=""
    )

    X = data[:, :-1]
    y = data[:, -1]

    return X, y


def sample_values(column, n=5):

    if len(column) <= n:
        return column.tolist()

    idx = random.sample(range(len(column)), n)
    return column[idx].tolist()


def check_dataset(dataset_info):

    data_path = dataset_info["数据集路径"]
    categorical_indices = dataset_info["离散属性列index"]

    print("\n=================================================")
    print("数据集:", data_path)

    X, y = load_data(data_path)

    n_samples, n_features = X.shape

    print("样本数:", n_samples)
    print("特征数:", n_features)
    print("离散列:", categorical_indices)

    for i in range(n_features):

        col = X[:, i]

        samples = sample_values(col)

        if i in categorical_indices:

            print(f"\n列 {i} (离散列)")
            print("样本:", samples)

        else:

            print(f"\n列 {i} (连续列)")
            print("样本:", samples)

            # 尝试检测是否可转换为 float
            bad_values = []

            for v in samples:
                try:
                    float(v)
                except:
                    bad_values.append(v)

            if bad_values:
                print("⚠️ 发现非数值:", bad_values)


def main():

    datasets = load_json_config(JSON_PATH)

    print("共检测数据集:", len(datasets))

    for dataset in datasets:

        try:
            check_dataset(dataset)
        except Exception as e:
            print("检测失败:", e)


if __name__ == "__main__":
    main()