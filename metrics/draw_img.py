import matplotlib.pyplot as plt

def draw_two_curves(data_dict1, data_dict2=None, label1='曲线1', label2='曲线2', save_path=None):
    """
    输入两个 dict，dict[dataset_name]=acc，
    横轴为 dataset_name，纵轴为 acc，
    每个 dict 绘制一条曲线，画在同一张图上。
    """
    # 提取横轴标签和对应的 acc 值
    datasets = list(data_dict1.keys())
    acc1 = [data_dict1[d] for d in datasets]
    if data_dict2:
        acc2 = [data_dict2[d] for d in datasets]
    else:
        acc2=None

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(datasets, acc1, marker='o', label=label1)
    if acc2:
        plt.plot(datasets, acc2, marker='s', label=label2)

    # 图表美化
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison Across Datasets')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    import os

    save_dir=os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
