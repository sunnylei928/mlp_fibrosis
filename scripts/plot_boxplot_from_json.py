"""
从 repeated_kfold_details.json 绘制箱线图
无需重新运行训练，直接使用保存的JSON文件
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_json_results(json_path):
    """加载JSON结果文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_boxplot(all_results, save_path):
    """
    绘制箱线图，展示各loss函数在不同指标上的分布

    Args:
        all_results: {repeat: {fold: {loss: {metric: value}}}}
        save_path: 保存路径
    """
    # 获取所有loss名称
    loss_names = list(all_results["repeat_0"]["0"].keys())
    metrics = ['accuracy', 'qwk', 'mae']
    metric_labels = ['Accuracy', 'QWK', 'MAE']

    # 准备数据：每个loss的每个指标的所有fold值
    plot_data = []
    for loss_name in loss_names:
        for metric_idx, metric in enumerate(metrics):
            values = []
            for repeat_key in all_results:
                fold_data = all_results[repeat_key]
                for f in fold_data:
                    values.append(fold_data[f][loss_name][metric])
            plot_data.append(values)

    # 创建图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        # 准备该指标的数据
        data_for_metric = []
        labels_for_metric = []
        for loss_idx, loss_name in enumerate(loss_names):
            data_idx = loss_idx * len(metrics) + metrics.index(metric)
            data_for_metric.append(plot_data[data_idx])
            labels_for_metric.append(loss_name)

        # 绘制箱线图
        bp = ax.boxplot(data_for_metric, tick_labels=labels_for_metric,
                        patch_artist=True, widths=0.6,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        # 添加均值点
        for i, data in enumerate(data_for_metric):
            mean_val = np.mean(data)
            ax.plot(i + 1, mean_val, 'go', markersize=8, label='Mean' if i == 0 else "")

        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Loss Function', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.set_ylim(0, 1.05 if metric != 'mae' else None)

        # 设置x轴刻度标签旋转（与comparison图一致）
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

        # 添加样本量标注
        for i, data in enumerate(data_for_metric):
            ax.text(i + 1, ax.get_ylim()[1] - 0.05, f"n={len(data)}",
                   ha='center', fontsize=8, color='gray')

    plt.suptitle('Repeated K-Fold Cross-Validation: Distribution Analysis (Box Plot)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"箱线图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='从JSON文件绘制箱线图')
    parser.add_argument('--json', type=str,
                        default='/home/ubuntu/lq/MLP_results/5fold_3/json/repeated_kfold_details.json',
                        help='JSON文件路径')
    parser.add_argument('--output', type=str,
                        default='repeated_kfold_boxplot.png',
                        help='输出图片路径')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.json):
        print(f"错误: 文件不存在 - {args.json}")
        print("\n请检查路径，或使用 --json 参数指定正确的文件路径")
        print("\n示例:")
        print("  python plot_boxplot_from_json.py --json path/to/repeated_kfold_details.json")
        return

    print(f"加载数据: {args.json}")
    all_results = load_json_results(args.json)

    # 检查数据结构
    print(f"Repeats: {len(all_results)}")
    print(f"Folds per repeat: {len(all_results['repeat_0'])}")
    print(f"Loss functions: {list(all_results['repeat_0']['0'].keys())}")

    plot_boxplot(all_results, args.output)


if __name__ == "__main__":
    main()
