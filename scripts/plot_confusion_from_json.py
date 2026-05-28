"""
从 repeated_kfold_predictions.json 绘制混淆矩阵
无需重新运行训练，直接使用保存的JSON文件
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix


def load_json_results(json_path):
    """加载JSON预测结果文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_confusion_matrices(all_predictions, save_path, n_classes=5, class_names=None):
    """
    绘制混淆矩阵对比图（聚合所有fold的结果）

    Args:
        all_predictions: {repeat_idx: {fold_idx: {loss_name: {"y_true": ..., "y_pred": ...}}}}
        save_path: 保存路径
        n_classes: 类别数量
        class_names: 类别名称
    """
    if class_names is None:
        class_names = [f'F{i}' for i in range(n_classes)]

    # 获取所有loss函数
    loss_names = list(all_predictions["repeat_0"]["0"].keys())
    n_losses = len(loss_names)

    # 计算聚合混淆矩阵
    aggregated_cms = {}

    for loss_name in loss_names:
        all_y_true = []
        all_y_pred = []

        # 聚合所有repeat和fold的预测结果
        for repeat_key in all_predictions:
            fold_data = all_predictions[repeat_key]
            for fold_idx in fold_data:
                if loss_name in fold_data[fold_idx]:
                    all_y_true.extend(fold_data[fold_idx][loss_name]['y_true'])
                    all_y_pred.extend(fold_data[fold_idx][loss_name]['y_pred'])

        # 计算混淆矩阵
        cm = confusion_matrix(all_y_true, all_y_pred, labels=range(n_classes))
        aggregated_cms[loss_name] = cm

    # 动态布局：根据loss数量调整
    n_cols = min(3, n_losses)
    n_rows = (n_losses + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # 处理单个子图的情况
    if n_losses == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, loss_name in enumerate(loss_names):
        ax = axes[idx]
        cm = aggregated_cms[loss_name]

        # 归一化（按行归一化，显示百分比）
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

        # 绘制热图
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=cm.max())

        # 添加数值标注
        for i in range(n_classes):
            for j in range(n_classes):
                # 同时显示绝对数量和百分比
                text_color = 'white' if cm_norm[i, j] > 50 else 'black'
                text = f'{cm[i, j]}\n({cm_norm[i, j]:.1f}%)'
                ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=9)

        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        ax.set_title(f'{loss_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Count', rotation=270, labelpad=15)

    # 隐藏未使用的子图
    for idx in range(n_losses, len(axes)):
        axes[idx].set_visible(False)

    # 获取总的评估次数
    n_repeats = len(all_predictions)
    n_folds = len(all_predictions["repeat_0"])
    total_evals = n_repeats * n_folds

    plt.suptitle(f'Confusion Matrices (Aggregated over {n_repeats} repeats × {n_folds} folds = {total_evals} evaluations)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"混淆矩阵已保存: {save_path}")


def plot_single_confusion_matrix(all_predictions, loss_name, save_path, n_classes=5, class_names=None):
    """
    绘制单个loss函数的混淆矩阵（适合论文主图）

    Args:
        all_predictions: 预测结果字典
        loss_name: 要绘制的loss名称
        save_path: 保存路径
        n_classes: 类别数量
        class_names: 类别名称
    """
    if class_names is None:
        class_names = [f'F{i}' for i in range(n_classes)]

    if loss_name not in all_predictions["repeat_0"]["0"]:
        print(f"错误: 找不到 loss 函数 '{loss_name}'")
        return

    # 聚合预测结果
    all_y_true = []
    all_y_pred = []

    for repeat_key in all_predictions:
        fold_data = all_predictions[repeat_key]
        for fold_idx in fold_data:
            if loss_name in fold_data[fold_idx]:
                all_y_true.extend(fold_data[fold_idx][loss_name]['y_true'])
                all_y_pred.extend(fold_data[fold_idx][loss_name]['y_pred'])

    # 计算混淆矩阵
    cm = confusion_matrix(all_y_true, all_y_pred, labels=range(n_classes))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

    # 绘制
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=cm.max())

    for i in range(n_classes):
        for j in range(n_classes):
            text_color = 'white' if cm_norm[i, j] > 50 else 'black'
            text = f'{cm[i, j]}\n({cm_norm[i, j]:.1f}%)'
            ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=11)

    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(f'{loss_name.upper()}: Confusion Matrix (Aggregated over {len(all_y_true)} samples)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count', rotation=270, labelpad=20, fontsize=11)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"单个混淆矩阵已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='从JSON文件绘制混淆矩阵')
    parser.add_argument('--json', type=str,
                        default='/home/ubuntu/lq/MLP_results/5fold_3/json/repeated_kfold_predictions.json',
                        help='JSON文件路径')
    parser.add_argument('--output', type=str,
                        default='confusion_matrices.png',
                        help='输出图片路径')
    parser.add_argument('--single', type=str, default=None,
                        help='只绘制单个loss函数的混淆矩阵（如: coral, ce）')
    parser.add_argument('--single-output', type=str,
                        default='confusion_matrix_single.png',
                        help='单个混淆矩阵输出路径')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.json):
        print(f"错误: 文件不存在 - {args.json}")
        print("\n请检查路径，或使用 --json 参数指定正确的文件路径")
        print("\n示例:")
        print("  python plot_confusion_from_json.py --json path/to/repeated_kfold_predictions.json")
        return

    print(f"加载数据: {args.json}")
    all_predictions = load_json_results(args.json)

    # 检查数据结构
    print(f"Repeats: {len(all_predictions)}")
    print(f"Folds per repeat: {len(all_predictions['repeat_0'])}")
    print(f"Loss functions: {list(all_predictions['repeat_0']['0'].keys())}")

    class_names = ['F0', 'F1', 'F2', 'F3', 'F4']

    # 绘制所有混淆矩阵
    plot_confusion_matrices(all_predictions, args.output, n_classes=5, class_names=class_names)

    # 如果指定了单个loss，单独绘制
    if args.single:
        plot_single_confusion_matrix(all_predictions, args.single, args.single_output,
                                     n_classes=5, class_names=class_names)


if __name__ == "__main__":
    main()
