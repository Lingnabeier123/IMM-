from typing import Any
import numpy as np
from kmodes.kprototypes import KPrototypes


class TreeNode:
    def __init__(self):
        self.condition = None
        self.cluster = None
        self.left = None
        self.right = None
        self.dimension = None
        self.threshold = None
        self.is_categorical = None


def count_leaves(tree: TreeNode) -> int:
    if tree is None:
        return 0
    if tree.cluster is not None:
        return 1
    left_leaves = count_leaves(tree.left) if tree.left else 0
    right_leaves = count_leaves(tree.right) if tree.right else 0
    return left_leaves + right_leaves


def max_depth(tree: TreeNode) -> int:
    if tree is None:
        return 0
    if tree.cluster is not None:
        return 1
    left_depth = max_depth(tree.left) if tree.left else 0
    right_depth = max_depth(tree.right) if tree.right else 0
    return 1 + max(left_depth, right_depth)


def avg_leaf_depth(tree: TreeNode) -> float:
    def collect_leaf_depths(n: TreeNode, depth: int, depths: list):
        if n is None:
            return
        if n.cluster is not None:
            depths.append(depth)
            return
        collect_leaf_depths(n.left, depth + 1, depths)
        collect_leaf_depths(n.right, depth + 1, depths)

    depths = []
    collect_leaf_depths(tree, 1, depths)
    if not depths:
        return 0.0
    return sum(depths) / len(depths)


def get_majority_label(y):
    if len(y) == 0:
        return None
    unique, counts = np.unique(y, return_counts=True)
    return unique[np.argmax(counts)]


def mistake_categorical(x: np.ndarray, mu: np.ndarray, i: int, theta: float):
    return 1 if (x[i] == theta) != (mu[i] == theta) else 0


def mistake_numerical(x: np.ndarray, mu: np.ndarray, i: int, theta: float):
    return 1 if (x[i] <= theta) != (mu[i] <= theta) else 0


def get_candidate_thetas_categorical(X: np.ndarray, active_centers: list, centroids: np.ndarray, dim: int):
    center_values = {centroids[cid][dim] for cid in active_centers}
    if len(center_values) < 2:
        return []
    return list(center_values)

def get_candidate_thetas_numerical(X: np.ndarray, active_centers: list, centroids: np.ndarray, dim: int):
    center_values = set()

    for cid in active_centers:
        val = centroids[cid][dim]
        center_values.add(val)

    if len(center_values) < 2:
        return []

    try:
        sorted_values = sorted(center_values, key=lambda x: float(x))
    except Exception as e:
        print("\n========== 排序失败 ==========")
        print(f"列编号: {dim}")
        print(f"centroid values: {center_values}")
        print(f"value types: {[type(v) for v in center_values]}")
        print("错误:", e)
        print("==============================\n")
        return []

    thetas = []

    for i in range(len(sorted_values) - 1):
        try:
            v1 = float(sorted_values[i])
            v2 = float(sorted_values[i + 1])

            theta = (v1 + v2) / 2
            thetas.append(theta)

        except Exception as e:
            print("\n========== theta计算失败 ==========")
            print(f"列编号: {dim}")
            print(f"value1: {sorted_values[i]} type={type(sorted_values[i])}")
            print(f"value2: {sorted_values[i+1]} type={type(sorted_values[i+1])}")
            print("centroid column values:")

            for cid in active_centers:
                val = centroids[cid][dim]
                print(f"  cluster {cid}: {val} type={type(val)}")

            print("错误:", e)
            print("=================================\n")

            continue

    return thetas
# def get_candidate_thetas_numerical(X: np.ndarray, active_centers: list, centroids: np.ndarray, dim: int):
#     center_values = set()
#     for cid in active_centers:
#         center_values.add(centroids[cid][dim])
    
#     if len(center_values) < 2:
#         return []
    
#     sorted_values = sorted(center_values)
#     thetas = []
#     for i in range(len(sorted_values) - 1):
#         theta = (sorted_values[i] + sorted_values[i + 1]) / 2
#         thetas.append(theta)
    
#     return thetas


def build_tree(X: np.ndarray, y: np.ndarray, centroids: np.ndarray, active_centers: list, categorical_indices: list):
    if len(active_centers) == 1:
        leaf = TreeNode()
        leaf.cluster = active_centers[0]
        return leaf

    d = X.shape[1] if len(X) > 0 else centroids.shape[1]
    m = len(X)
    
    candidate_splits = []

    for i in range(d):
        is_categorical = i in categorical_indices
        
        if is_categorical:
            valid_thetas = get_candidate_thetas_categorical(X, active_centers, centroids, i)
            mistake_func = mistake_categorical
        else:
            valid_thetas = get_candidate_thetas_numerical(X, active_centers, centroids, i)
            mistake_func = mistake_numerical
        
        for theta in valid_thetas:
            if is_categorical:
                all_centers_match = True
                for cid in active_centers:
                    if centroids[cid][i] != theta:
                        all_centers_match = False
                        break
                if all_centers_match:
                    continue
            else:
                # if not isinstance(theta, (int, float)):
                #      continue
                try:
                    all_centers_le = True
                    all_centers_gt = True
                    for cid in active_centers:
                        val = centroids[cid][i]

                        # 尝试转float
                        val = float(val)

                        if val <= theta:
                            all_centers_gt = False
                            break
                        if val > theta:
                            all_centers_le = False
                            break

                except Exception as e:
                    print("\n========== 数值列错误 ==========")
                    print(f"列编号: {i}")
                    print(f"theta: {theta} type={type(theta)}")

                    print("centroid column values:")
                    for cid in active_centers:
                        v = centroids[cid][i]
                        print(f" cluster {cid}: {v} type={type(v)}")

                    print("错误:", e)
                    print("该列应该被标记为 categorical")
                    print("================================\n")

                    continue
                if all_centers_le or all_centers_gt:
                    continue

            total_mistake = 0
            if m > 0:
                for j in range(m):
                    total_mistake += mistake_func(X[j], centroids[y[j]], i, theta)
            
            if m > 0:
                if is_categorical:
                    L_count = np.sum(X[:, i] == theta)
                else:
                    L_count = np.sum(X[:, i] <= theta)
                R_count = m - L_count
                balance_score = min(L_count, R_count)
            else:
                balance_score = 0
                
            candidate_splits.append((total_mistake, balance_score, i, theta, is_categorical))
    
    if not candidate_splits:
        leaf = TreeNode()
        leaf.cluster = active_centers[0]
        return leaf

    candidate_splits.sort(key=lambda x: (x[0], -x[1]))
    best_mistake, _, best_i, best_theta, best_is_categorical = candidate_splits[0]
    
    L_active = []
    R_active = []
    
    for cid in active_centers:
        if best_is_categorical:
            if centroids[cid][best_i] == best_theta:
                L_active.append(cid)
            else:
                R_active.append(cid)
        else:
            if centroids[cid][best_i] <= best_theta:
                L_active.append(cid)
            else:
                R_active.append(cid)
    
    L_X, L_y = [], []
    R_X, R_y = [], []
    
    if m > 0:
        M_indices = set()
        for j in range(m):
            if best_is_categorical:
                if mistake_categorical(X[j], centroids[y[j]], best_i, best_theta) == 1:
                    M_indices.add(j)
            else:
                if mistake_numerical(X[j], centroids[y[j]], best_i, best_theta) == 1:
                    M_indices.add(j)
        
        for j in range(m):
            if j not in M_indices:
                if best_is_categorical:
                    if X[j][best_i] == best_theta:
                        L_X.append(X[j])
                        L_y.append(y[j])
                    else:
                        R_X.append(X[j])
                        R_y.append(y[j])
                else:
                    if X[j][best_i] <= best_theta:
                        L_X.append(X[j])
                        L_y.append(y[j])
                    else:
                        R_X.append(X[j])
                        R_y.append(y[j])
    
    L_X = np.array(L_X) if len(L_X) > 0 else np.array([]).reshape(0, d)
    L_y = np.array(L_y) if len(L_y) > 0 else np.array([])
    R_X = np.array(R_X) if len(R_X) > 0 else np.array([]).reshape(0, d)
    R_y = np.array(R_y) if len(R_y) > 0 else np.array([])

    node = TreeNode()
    node.dimension = best_i
    node.threshold = best_theta
    node.is_categorical = best_is_categorical
    
    if best_is_categorical:
        node.condition = f"x_{best_i} == {best_theta}"
    else:
        node.condition = f"x_{best_i} <= {best_theta:.4f}"
    
    node.left = build_tree(L_X, L_y, centroids, L_active, categorical_indices)
    node.right = build_tree(R_X, R_y, centroids, R_active, categorical_indices)
    
    return node


def iterative_mistake_minimization(X: np.ndarray, k: int, categorical_indices: list):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    
    # 1. 运行 KPrototypes
    kproto = KPrototypes(n_clusters=k, init='Cao', n_init=5, random_state=42)
    y_kproto_pred = kproto.fit_predict(X, categorical=categorical_indices)

    # 2. 获取原始排序的质心（连续在前，离散在后）
    centroids_raw = kproto.cluster_centroids_
    num_features = X.shape[1] - len(categorical_indices)
    centroids_num = centroids_raw[:, :num_features].astype(float)
    centroids_cat = centroids_raw[:, num_features:]

    # --- 方案1核心：将质心列顺序复原为原始 X 的顺序 ---
    # 使用 object 类型以兼容 float 和 string/int 混合
    centroids = np.empty((k, X.shape[1]), dtype=object) 
    
    all_idx = list(range(X.shape[1]))
    cat_set = set(categorical_indices)
    num_idx = [i for i in all_idx if i not in cat_set]
    
    # 把连续特征塞回它原本在 X 中的位置
    for i, orig_col in enumerate(num_idx):
        centroids[:, orig_col] = centroids_num[:, i]
        
    # 把离散特征塞回它原本在 X 中的位置
    for i, orig_col in enumerate(categorical_indices):
        centroids[:, orig_col] = centroids_cat[:, i]
    # ------------------------------------------------

    # 3. 直接使用原始的 X, 还原后的 centroids, 和原始的 categorical_indices 建树
    initial_active_centers = list(range(k))
    tree_root = build_tree(X, y_kproto_pred, centroids, initial_active_centers, categorical_indices)
    
    return tree_root, centroids, y_kproto_pred
    
def predict(tree, X):
    def predict_single(node, x):
        if node.cluster is not None:
            return node.cluster
        
        if node.is_categorical:
            if x[node.dimension] == node.threshold:
                if node.left:
                    return predict_single(node.left, x)
            else:
                if node.right:
                    return predict_single(node.right, x)
        else:
            if x[node.dimension] <= node.threshold:
                if node.left:
                    return predict_single(node.left, x)
            else:
                if node.right:
                    return predict_single(node.right, x)
        
        return None
    
    predictions = []
    for x in X:
        pred = predict_single(tree, x)
        predictions.append(pred)
    
    return np.array(predictions)


def print_tree(node, depth=0, leaf_counter=None):
    if leaf_counter is None:
        leaf_counter = {}
    
    indent = "  " * depth
    
    if node.cluster is not None:
        print(f"{indent}Leaf: cluster = {node.cluster}")
        leaf_counter[node.cluster] = leaf_counter.get(node.cluster, 0) + 1
    else:
        print(f"{indent}Node: {node.condition}")
        if node.left:
            print(f"{indent}Left:")
            print_tree(node.left, depth + 1, leaf_counter)
        if node.right:
            print(f"{indent}Right:")
            print_tree(node.right, depth + 1, leaf_counter)
    
    if depth == 0:
        print("\n各簇叶子个数统计:")
        for cluster, count in leaf_counter.items():
            print(f"簇 {cluster}: {count} 个叶子")
        return leaf_counter


def main():
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    
    print("创建示例数据...")
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)
    X_discrete = np.round(X).astype(int)
    
    print(f"数据集形状: {X_discrete.shape}")
    print(f"数据样例:\n{X_discrete[:5]}")
    
    k = 3
    categorical_indices = []
    
    print(f"\n运行迭代错误最小化算法 (k={k})...")
    tree_root, centroids, y_pred = iterative_mistake_minimization(X_discrete, k, categorical_indices)
    
    print(f"\n聚类中心:\n{centroids}")
    
    print("\n决策树结构:")
    print_tree(tree_root)
    
    print("\n进行预测...")
    predictions = predict(tree_root, X_discrete)
    
    from sklearn.metrics import accuracy_score
    simple_accuracy = np.mean(predictions == y_pred)
    print(f"简单预测准确率: {simple_accuracy:.4f}")
    
    print("\n生成可视化结果...")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='viridis')
    plt.title('原始数据')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    
    plt.subplot(132)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.7, marker='X')
    plt.title('K-prototypes 聚类结果')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    
    plt.subplot(133)
    plt.scatter(X[:, 0], X[:, 1], c=predictions, s=50, cmap='viridis')
    plt.title('决策树预测结果')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    
    plt.tight_layout()
    plt.savefig('k_prototype_results.png')
    print("可视化结果已保存为 'k_prototype_results.png'")


if __name__ == "__main__":
    main()
