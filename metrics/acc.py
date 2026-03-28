import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import f1_score

def clustering_accuracy(y_true, y_pred):
    """
    计算聚类准确率 (Clustering Accuracy)。
    使用匈牙利算法 (Hungarian Algorithm) 寻找最佳标签映射。
    """
    # 1. 强制转换为整数类型，防止浮点数索引报错
    y_true = np.array(y_true).astype(np.int64)
    y_pred = np.array(y_pred).astype(np.int64)
    
    # 确保形状一致
    assert y_pred.size == y_true.size
    
    # 2. 计算混淆矩阵 (Confusion Matrix)
    # 注意：如果标签值非常大（如10000），建议先使用 LabelEncoder 编码
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_true[i], y_pred[i]] += 1
    
    # 3. 使用匈牙利算法寻找最佳匹配
    # linear_sum_assignment 旨在最小化成本，因此我们传入 (最大值 - 当前矩阵)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    # 4. 计算准确率
    # w[row_ind, col_ind] 提取了最佳匹配对角线上的元素
    match_count = w[row_ind, col_ind].sum()
    
    return float(match_count) / y_pred.size

def calculate_purity(y_true, y_pred):
    """
    计算聚类纯度 (Purity)。
    """
    # 1. 计算列联表 (行=真实标签, 列=预测标签)
    c_matrix = contingency_matrix(y_true, y_pred)
    
    # 2. 对每个簇（列），找到最大的真实类样本数
    # axis=0 表示沿着列方向求最大值
    return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)

def calculate_fscore(y_true, y_pred, average='macro'):
    """
    计算对齐后的 F-score。
    """
    # 1. 强制转换为整数类型
    y_true = np.array(y_true).astype(np.int64)
    y_pred = np.array(y_pred).astype(np.int64)
    
    assert y_pred.size == y_true.size
    
    # 2. 确定矩阵维度
    D = max(y_pred.max(), y_true.max()) + 1
    
    # 3. 构建混淆矩阵
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_true[i], y_pred[i]] += 1
        
    # 4. 使用匈牙利算法寻找最佳匹配
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    # 5. 生成映射字典: {预测标签 -> 真实标签}
    # col_ind 是预测标签索引，row_ind 是对应的真实标签索引
    map_dict = {c: r for r, c in zip(row_ind, col_ind)}
    
    # 6. 对齐预测标签
    # map_dict.get(label, label): 如果某个预测簇没有被匹配（极少见），保持原标签
    y_pred_aligned = np.array([map_dict.get(label, label) for label in y_pred])
    
    # 7. 计算 F-score
    return f1_score(y_true, y_pred_aligned, average=average)

# ==========================================
# 测试 Demo
# ==========================================
if __name__ == "__main__":
    # 模拟数据
    # 真实标签: [0, 0, 1, 1, 2, 2]
    # 聚类结果: [1, 1, 0, 0, 2, 2] (注意：0和1的标签互换了)
    y_true_sample = [0, 0, 1, 1, 2, 2]
    y_pred_sample = [1, 1, 0, 0, 2, 2] 

    print("真实标签:", y_true_sample)
    print("预测标签:", y_pred_sample)
    print("-" * 30)

    # 1. 计算 Accuracy
    acc = clustering_accuracy(y_true_sample, y_pred_sample)
    print(f"Accuracy: {acc:.4f}") # 应该是 1.0

    # 2. 计算 Purity
    purity = calculate_purity(y_true_sample, y_pred_sample)
    print(f"Purity:   {purity:.4f}") # 应该是 1.0

    # 3. 计算 F-score
    f_score = calculate_fscore(y_true_sample, y_pred_sample, average='macro')
    print(f"F-score:  {f_score:.4f}") # 应该是 1.0