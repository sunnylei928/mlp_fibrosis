"""
从 kfold_summary.json 绘制五折交叉验证条形图
显示均值 ± 标准差的形式
"""
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_json_results(json_path):
    """加载JSON结果文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_kfold_barchart(summary_data, save_path):
    """
    绘制五折交叉验证结果条形图（均值 ± 标准差）

    Args:
        summary_data: kfold_summary.json 内容
        save_path: 保存路径
    """
    # 使用系统通用的无衬线字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'liberation sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取所有loss名称
    loss_names = list(summary_data.keys())

    # 定义要绘制的指标
    metrics = ['accuracy_mean', 'adjacent_accuracy_mean', 'macro_f1_mean',
               'weighted_f1_mean', 'qwk_mean', 'mae_mean']
    metric_labels = ['Accuracy', 'Adjacent Accuracy', 'Macro F1',
                     'Weighted F1', 'QWK', 'MAE']

    # 创建子图：1行6列
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        # 提取数据
        means = []
        stds = []
        for loss_name in loss_names:
            means.append(summary_data[loss_name][metric])
            # 获取对应的标准差
            std_key = metric.replace('_mean', '_std')
            stds.append(summary_data[loss_name][std_key])

        # 绘制条形图
        x_pos = np.arange(len(loss_names))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color='steelblue', alpha=0.8, error_kw={'linewidth': 2})

        # 设置y轴范围
        if 'mae' not in metric:
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
            ax.set_ylim(top=max(means) + max(stds) + 0.1)

        # 设置标题和标签
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score' if 'mae' not in metric else 'Error',
                      fontsize=10, fontweight='bold')
        ax.set_xlabel('Loss Function', fontsize=10, fontweight='bold')

        # 设置x轴刻度
        ax.set_xticks(x_pos)
        ax.set_xticklabels(loss_names, rotation=45, ha='right', fontsize=9)

        # 添加网格
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

        # 找到最优bar的索引
        if 'mae' not in metric:
            best_idx = means.index(max(means))
        else:
            best_idx = means.index(min(means))

        # 在所有bar上显示数值，最优的用绿色突出
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            offset = 0.01 if 'mae' not in metric else (max(means) * 0.02)

            if i == best_idx:
                # 最优bar：绿色突出并显示数值
                bar.set_color('#2ecc71')
                bar.set_alpha(0.9)
                ax.text(bar.get_x() + bar.get_width()/2, height + std + offset,
                       f'{mean:.3f}', ha='center', va='bottom',
                       fontsize=8, fontweight='bold', color='#2ecc71')
            else:
                # 其他bar：显示数值，使用原色但降低透明度
                bar.set_color('steelblue')
                bar.set_alpha(0.7)
                ax.text(bar.get_x() + bar.get_width()/2, height + std + offset,
                       f'{mean:.3f}', ha='center', va='bottom',
                       fontsize=7, color='#555555')

    plt.suptitle('5-Fold Cross-Validation Results (Mean ± Std)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"条形图已保存: {save_path}")


def plot_combined_comparison(summary_data, save_path):
    """
    绘制综合对比图：所有指标的分组条形图

    Args:
        summary_data: kfold_summary.json 内容
        save_path: 保存路径
    """
    # 使用系统通用的无衬线字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'liberation sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    loss_names = list(summary_data.keys())
    metrics = ['accuracy_mean', 'adjacent_accuracy_mean', 'qwk_mean', 'mae_mean']
    metric_labels = ['Accuracy', 'Adjacent Accuracy', 'QWK', 'MAE']

    # 设置颜色
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        ax = axes[idx]

        means = []
        stds = []
        for loss_name in loss_names:
            means.append(summary_data[loss_name][metric])
            std_key = metric.replace('_mean', '_std')
            stds.append(summary_data[loss_name][std_key])

        x_pos = np.arange(len(loss_names))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=color, alpha=0.8, error_kw={'linewidth': 2})

        if 'mae' not in metric:
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
            ax.set_ylim(top=max(means) + max(stds) + 0.1)

        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score' if 'mae' not in metric else 'Error',
                      fontsize=10, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(loss_names, rotation=45, ha='right', fontsize=9)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

        # 找到最优bar的索引
        if 'mae' not in metric:
            best_idx = means.index(max(means))
        else:
            best_idx = means.index(min(means))

        # 在所有bar上显示数值，最优的用绿色突出
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            offset = 0.01 if 'mae' not in metric else (max(means) * 0.02)

            if i == best_idx:
                # 最优bar：绿色突出
                bar.set_color('#2ecc71')
                bar.set_alpha(0.9)
                ax.text(bar.get_x() + bar.get_width()/2, height + std + offset,
                       f'{mean:.3f}', ha='center', va='bottom',
                       fontsize=9, fontweight='bold', color='#2ecc71')
            else:
                # 其他bar：显示数值，使用原色但降低透明度
                bar.set_alpha(0.7)
                ax.text(bar.get_x() + bar.get_width()/2, height + std + offset,
                       f'{mean:.3f}', ha='center', va='bottom',
                       fontsize=8, color='#555555')

    plt.suptitle('5-Fold Cross-Validation: Performance Comparison',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"综合对比图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='从kfold_summary.json绘制条形图')
    parser.add_argument('--json', type=str,
                        default='/home/ubuntu/lq/MLP_results/20260528_054909/kfold_summary.json',
                        help='kfold_summary.json文件路径')
    parser.add_argument('--output', type=str, default='kfold_barchart.png',
                        help='输出图片路径')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.json):
        print(f"错误: 文件不存在 - {args.json}")
        print("\n请检查路径，或使用 --json 参数指定正确的文件路径")
        print("\n示例:")
        print("  python plot_kfold_barchart.py --json path/to/kfold_summary.json")
        return

    print(f"加载数据: {args.json}")
    summary_data = load_json_results(args.json)

    print(f"Loss functions: {list(summary_data.keys())}")
    print(f"Metrics available: {list(summary_data[list(summary_data.keys())[0]].keys())}")

    # 生成两种图
    base_name = args.output.replace('.png', '')
    plot_kfold_barchart(summary_data, f"{base_name}_detailed.png")
    plot_combined_comparison(summary_data, f"{base_name}_combined.png")


if __name__ == "__main__":
    main()
