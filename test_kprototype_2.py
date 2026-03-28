import os
import json
import numpy as np
import pandas as pd
import traceback
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score

# 配置路径
JSON_PATH = r"d:\Program Files\py\k_prototype_data_cleaned\dataset_info.json"
OUTPUT_DIR = "output_kprototype"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 1. 核心工具：标签对齐与评价
# ==============================

def get_best_mapping(y_true, y_pred):
    y_true = np.array(y_true).astype(np.int64)
    y_pred = np.array(y_pred).astype(np.int64)
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    mapped_y_pred = np.array([mapping.get(label, label) for label in y_pred])
    return mapped_y_pred

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_mapped = get_best_mapping(y_true, y_pred)
    
    acc = np.sum(y_true == y_mapped) / len(y_true)
    f1 = f1_score(y_true, y_mapped, average="macro", zero_division=0)
    
    purity = 0
    clusters = np.unique(y_pred)
    for c in clusters:
        mask = (y_pred == c)
        if np.any(mask):
            purity += pd.Series(y_true[mask]).value_counts().max()
    purity /= len(y_true)
    
    return acc, f1, purity

# ==============================
# 2. 数据加载
# ==============================

def load_data_file(file_path, categorical_indices):
    data = np.genfromtxt(file_path, delimiter=",", dtype=str, filling_values="")
    X_raw = data[:, :-1]
    y_raw = data[:, -1]

    n_samples, n_features = X_raw.shape
    X = np.empty((n_samples, n_features), dtype=object)
    cat_set = set(categorical_indices)

    for j in range(n_features):
        col = X_raw[:, j]
        if j in cat_set:
            X[:, j] = col
        else:
            def to_float(v):
                try: return float(v)
                except: return 0.0
            X[:, j] = [to_float(v) for v in col]

    unique_labels = np.unique(y_raw)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y_raw])
    return X, y

# ==============================
# 3. 画图与表格渲染
# ==============================

def _draw_base_table(dataset_names, columns, cell_text, title, save_path, header_color):
    fig_height = len(dataset_names) * 0.4 + 2
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('tight')
    ax.axis('off')

    colors = ["#f2f2f2" if i % 2 == 0 else "#ffffff" for i in range(len(dataset_names))]
    
    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center', loc='center',
                     cellColours=[[c]*len(columns) for c in colors])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(header_color)

    plt.title(title, y=0.98, fontsize=14, weight='bold')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def draw_performance_table(results_dict, save_name="final_comparison_results.png"):
    names = list(results_dict.keys())
    columns = ('Dataset', 'KProto Acc', 'Ours Acc', 'KProto F1', 'Ours F1', 'KProto Purity', 'Ours Purity')
    cell_text = []
    for name in names:
        r = results_dict[name]
        cell_text.append([
            name,
            f"{r['kproto_acc']:.4f}", f"{r['tree_acc']:.4f}",
            f"{r['kproto_f1']:.4f}", f"{r['tree_f1']:.4f}",
            f"{r['kproto_purity']:.4f}", f"{r['tree_purity']:.4f}"
        ])
    _draw_base_table(names, columns, cell_text, "Algorithm Performance Comparison (Avg of 10 Runs)", 
                     os.path.join(OUTPUT_DIR, save_name), '#40466e')

def draw_leaf_structure_table(results_dict, save_name="tree_structure_results.png"):
    names = list(results_dict.keys())
    columns = ('Dataset', 'Ours Max Depth', 'Ours Avg Leaf Depth', 'Ours Leaf Count', 'Real Clusters Count')
    cell_text = []
    for name in names:
        r = results_dict[name]
        cell_text.append([
            name,
            f"{r['tree_depth']:.1f}", 
            f"{r['tree_avg_depth']:.2f}",
            f"{r['tree_leaves']:.1f}",
            f"{r['real_k']:.0f}"
        ])
    _draw_base_table(names, columns, cell_text, "IMM Tree Structural Analysis (Avg of 10 Runs)", 
                     os.path.join(OUTPUT_DIR, save_name), '#2e8b57')

# ==============================
# 4. 主实验逻辑
# ==============================

def process_dataset(dataset_info, times=10):
    data_path = dataset_info["数据集路径"]
    categorical_indices = dataset_info["离散属性列index"]
    
    from kmodes.kprototypes import KPrototypes
    from k_prototype_tree_v1 import (
        iterative_mistake_minimization, predict, 
        max_depth, avg_leaf_depth, count_leaves
    )

    X, y = load_data_file(data_path, categorical_indices)
    k = len(np.unique(y))
    
    run_results = []

    for i in range(times):
        # K-Prototypes Baseline (不设置固定随机种子以计算平均值)
        kp = KPrototypes(n_clusters=k, init="Cao", n_init=1, random_state=None)
        y_kp = kp.fit_predict(X, categorical=categorical_indices)
        kp_acc, kp_f1, kp_pur = calculate_metrics(y, y_kp)
        
        # IMM Tree
        tree, _, _ = iterative_mistake_minimization(X, k, categorical_indices)
        y_tree = predict(tree, X)
        t_acc, t_f1, t_pur = calculate_metrics(y, y_tree)
        
        run_results.append({
            "kproto_acc": kp_acc, "kproto_f1": kp_f1, "kproto_purity": kp_pur,
            "tree_acc": t_acc, "tree_f1": t_f1, "tree_purity": t_pur,
            "tree_depth": max_depth(tree),
            "tree_avg_depth": avg_leaf_depth(tree),
            "tree_leaves": count_leaves(tree)
        })

    # 计算 10 次运行的平均值
    avg_metrics = pd.DataFrame(run_results).mean().to_dict()
    avg_metrics["real_k"] = k
    
    return avg_metrics

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        datasets_config = json.load(f)

    all_results = {}
    for entry in datasets_config:
        name = os.path.basename(os.path.dirname(entry["数据集路径"]))
        print(f"正在处理: {name} (运行 10 次取平均值)...")
        try:
            res = process_dataset(entry, times=10)
            all_results[name] = res
            print(f"  [OK] AVG ACC: {res['tree_acc']:.4f} | AVG F1: {res['tree_f1']:.4f}")
        except Exception:
            print(f"  [Error] {name} 失败")
            traceback.print_exc()

    if all_results:
        draw_performance_table(all_results)
        draw_leaf_structure_table(all_results)
    
    pd.DataFrame(all_results).T.to_csv(os.path.join(OUTPUT_DIR, "detailed_avg_results.csv"))
    print(f"\n所有实验已完成。结果保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()