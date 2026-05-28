import os
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# 使用系统通用的无衬线字体，兼容 Windows/Linux
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'liberation sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def create_versioned_dir(base_dir):
    """
    Create a timestamped subdirectory under base_dir for versioning results.
    Returns the full path to the version directory.
    Also writes a 'latest_run.txt' pointing to the latest version.
    """
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join(base_dir, timestamp)
    os.makedirs(version_dir, exist_ok=True)

    # Record latest version path
    latest_file = os.path.join(base_dir, "latest_run.txt")
    with open(latest_file, "w", encoding="utf-8") as f:
        f.write(version_dir)

    return version_dir


def save_run_metadata(config, model, loss_configs, save_dir):
    """
    Save hyperparameters, model architecture, and loss config for reproducibility.
    """
    # Config attributes (handle non-serializable types like torch.device)
    config_dict = {}
    for k, v in vars(config).items():
        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
            config_dict[k] = v
        else:
            config_dict[k] = str(v)

    # Loss configuration summary
    config_dict["loss_configs"] = {
        k: {"loss_type": v[0], "has_weights": v[1] is not None}
        for k, v in loss_configs.items()
    }

    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    # Model architecture (last model trained)
    with open(os.path.join(save_dir, "model_arch.txt"), "w", encoding="utf-8") as f:
        f.write(str(model))

    # Run timestamp
    meta = {
        "run_time": datetime.datetime.now().isoformat(),
        "version_dir": save_dir
    }
    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def plot_confusion_matrices(results_dict, label_encoder, save_dir):
    """
    Plot confusion matrices for all methods in a grid layout.
    results_dict: {loss_name: metrics_dict} where metrics_dict contains 'preds' and 'labels'.
    """
    os.makedirs(save_dir, exist_ok=True)

    n = len(results_dict)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    class_names = [label_encoder.classes_[i] for i in range(len(label_encoder.classes_))]

    for idx, (name, metrics) in enumerate(results_dict.items()):
        cm = confusion_matrix(metrics["labels"], metrics["preds"])
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)

        # Annotate each cell
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrices.png"), dpi=150)
    plt.close()

# 评估指标：测试集上的最终评估会输出四个核心指标：准确率 (Accuracy)、相邻准确率 (Adjacent Accuracy，即允许相差一个等级的容错准确率)
# 宏 F1 分数 (Macro F1) 和加权 F1 分数 (Weighted F1)。
def save_results(results_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable = {}
        for k, v in results_dict.items():
            serializable[k] = {m: float(v[m]) if isinstance(v[m], (np.floating, np.integer)) else v[m]
                               # 这里增加了 "qwk" 和 "mae"
                               for m in ["accuracy", "adjacent_accuracy", "macro_f1", "weighted_f1", "qwk", "mae"]}

            # 添加混淆矩阵数据
            if "preds" in v and "labels" in v:
                cm = confusion_matrix(v["labels"], v["preds"])
                serializable[k]["confusion_matrix"] = cm.tolist()
                # 添加预测结果（用于后续分析）
                serializable[k]["preds"] = v["preds"].tolist()
                serializable[k]["labels"] = v["labels"].tolist()

        json.dump(serializable, f, ensure_ascii=False, indent=2)


def plot_comparison(results_dict, histories, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Plot metrics bar chart
    # 将原本的 1 行 4 列改为 1 行 6 列，并加宽画布
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    metrics = ["accuracy", "adjacent_accuracy", "macro_f1", "weighted_f1", "qwk", "mae"]
    titles = ["Accuracy", "Adjacent Accuracy", "Macro F1", "Weighted F1", "QWK", "MAE"]

    for ax, metric, title in zip(axes, metrics, titles):
        names = list(results_dict.keys())
        values = [results_dict[n][metric] for n in names]
        bars = ax.bar(names, values, color='steelblue')

        # 特别处理：MAE 是误差，不能把上限锁死在 1.05，其他指标是 0~1 的分数
        if metric != "mae":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0) # MAE 的顶部自适应

        ax.set_title(title)
        ax.set_ylabel("Score" if metric != "mae" else "Error")
        for bar, val in zip(bars, values):
            # 动态调整 MAE 文字标签的纵向偏移量
            offset = 0.01 if metric != "mae" else (max(values)*0.02)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=9)

        # 改善 x 轴标签显示
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_comparison.png"), dpi=150)
    plt.close()

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, hist in histories.items():
        axes[0].plot(hist["train_loss"], label=f"{name} (train)")
        axes[0].plot(hist["val_loss"], label=f"{name} (val)", linestyle="--")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7)

    for name, hist in histories.items():
        axes[1].plot(hist["val_acc"], label=f"{name} acc")
        axes[1].plot(hist["val_adj_acc"], label=f"{name} adj_acc", linestyle="--")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
