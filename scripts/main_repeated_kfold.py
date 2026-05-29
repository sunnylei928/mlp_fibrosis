"""
多次五折交叉验证 - 完整实现
包含：数据划分、训练、评估、结果汇总、可视化
"""
import os
import sys
import json
import copy
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

# 导入项目模块
from config import Config
from core.dataset import FibrosisDataset
from core.model import MLPClassifier, CORALNet
from core.loss import get_loss
from training.train import train_model
from core.evaluate import evaluate_epoch
from training.utils import create_versioned_dir

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'liberation sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class Logger:
    """日志记录"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def close(self):
        self.log.close()


def log_model_params(config, loss_type, loss_kwargs, model_params, criterion=None):
    """记录模型和训练参数"""
    print(f"\n{'─' * 50}")
    print(f"  模型参数配置")
    print(f"{'─' * 50}")
    print(f"  Loss 类型: {loss_type}")
    print(f"  模型架构: {model_params.get('arch', 'MLP')}")
    print(f"  Hidden dims: {config.HIDDEN_DIMS}")
    print(f"  Dropout: {config.DROPOUT}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Weight decay: {config.WEIGHT_DECAY}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Random seed: {config.RANDOM_SEED}")
    print(f"  Loss 参数:")
    for key, value in loss_kwargs.items():
        if key not in ['device', 'num_classes']:
            print(f"    {key}: {value}")
    print(f"{'─' * 50}")


def load_data_full(config):
    """加载完整数据集"""
    df = pd.read_excel(config.DATA_PATH)

    # 数据预处理
    df["HA"] = df["HA"].astype(str).str.replace(r"\.\.", ".", regex=True)
    df["HA"] = pd.to_numeric(df["HA"], errors="coerce")
    if df["HA"].isnull().sum() > 0:
        df["HA"] = df["HA"].fillna(df["HA"].median())

    # 编码标签
    le = LabelEncoder()
    df[config.TARGET_COL] = le.fit_transform(df[config.TARGET_COL])

    # 独热编码
    df = pd.get_dummies(df, columns=config.CATEGORICAL_COLS, drop_first=False)

    # 准备特征和标签
    feature_cols = [c for c in df.columns
                    if c not in config.DROP_COLS + [config.TARGET_COL]]
    X = df[feature_cols].values.astype(np.float32)
    y = df[config.TARGET_COL].values.astype(np.int64)

    return X, y, le


def run_single_repeat(config, loss_configs, X, y, le, repeat_idx, seed, save_dir=None):
    """
    运行单次五折交叉验证

    Returns:
        fold_results: {fold_idx: {loss_name: metrics}}
        fold_predictions: {fold_idx: {loss_name: {"y_true": ..., "y_pred": ...}}}
    """
    print(f"\n{'='*80}")
    print(f"  Repeat {repeat_idx + 1} (seed={seed})")
    print(f"{'='*80}")

    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 创建五折划分
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    fold_results = {}
    fold_predictions = {}

    # 遍历每一折
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/5 ---")
        print(f"训练: {len(train_idx)}, 测试: {len(test_idx)}")

        # 划分数据
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 从训练集划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=seed
        )

        print(f"最终: 训练={len(X_train)}, 验证={len(X_val)}, 测试={len(X_test)}")
        print(f"标签分布: {np.bincount(y_train)}")

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # 创建数据加载器
        train_loader = DataLoader(FibrosisDataset(X_train, y_train),
                                  batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(FibrosisDataset(X_val, y_val),
                                batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(FibrosisDataset(X_test, y_test),
                                 batch_size=config.BATCH_SIZE, shuffle=False)

        input_dim = X_train.shape[1]
        num_classes = len(le.classes_)

        # 计算类别权重（sqrt 平滑版本，sklearn 风格）
        class_counts = np.bincount(y_train)
        # 使用 sqrt 平滑：weight = 1/sqrt(count)，避免权重过大
        class_weights = torch.FloatTensor(1.0 / np.sqrt(class_counts + 1e-6))
        # 归一化到 [1, max]，保证最小权重为1
        class_weights = class_weights / class_weights.min()
        class_weights = class_weights.to(config.DEVICE)

        fold_results[fold_idx] = {}
        fold_predictions[fold_idx] = {}

        # 遍历每个 loss
        for loss_name, (loss_type, weight) in loss_configs.items():
            print(f"\n{loss_name}:")

            # 根据损失类型选择架构
            if loss_type == 'coral':
                # CORALNet + CORAL loss (原始架构)
                model = CORALNet(input_dim, config.HIDDEN_DIMS, num_classes, config.DROPOUT).to(config.DEVICE)
            elif loss_type == 'mlp_coral':
                # MLP + CORAL loss (简化架构)
                model = MLPClassifier(input_dim, config.HIDDEN_DIMS, num_classes,
                                      config.DROPOUT, ordinal_head=True).to(config.DEVICE)
            else:
                # 标准分类损失
                model = MLPClassifier(input_dim, config.HIDDEN_DIMS, num_classes,
                                      config.DROPOUT, ordinal_head=False).to(config.DEVICE)

            # 创建 loss
            loss_kwargs = {"num_classes": num_classes, "device": config.DEVICE}
            if loss_type == 'cdw_ce':
                loss_kwargs["alpha"] = 1.0
            elif loss_type == 'cdw_ada':
                loss_kwargs["lower_param"] = 0.75  # 下三角权重
                loss_kwargs["upper_param"] = 0.5   # 上三角权重
                loss_kwargs["alpha"] = 1.2         # 距离敏感度（>1加剧远距离惩罚）
            elif loss_type == 'cdw_exp':
                loss_kwargs["lower_param"] = 1.3   # 下三角基数（低估惩罚）
                loss_kwargs["upper_param"] = 1.2   # 上三角基数（高估惩罚）

            criterion = get_loss(loss_type, class_weights=weight, **loss_kwargs)

            # 记录参数
            model_params = {
                'arch': 'CORALNet' if loss_type == 'coral' else 'MLP',
                'input_dim': input_dim,
                'num_classes': num_classes
            }
            log_model_params(config, loss_type, loss_kwargs, model_params, criterion)

            optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

            # 训练
            history = train_model(
                model, train_loader, val_loader, criterion, optimizer, config,
                loss_name=f"{loss_name}_r{repeat_idx}_f{fold_idx}", save_dir=save_dir
            )

            # 测试并获取预测结果
            test_metrics = evaluate_epoch(model, test_loader, criterion, config.DEVICE)

            # 获取预测结果
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(config.DEVICE)
                    outputs = model(batch_x)
                    if loss_type in ['coral', 'mlp_coral']:
                        # CORAL: 使用 predict 方法
                        if hasattr(criterion, 'predict'):
                            preds = criterion.predict(outputs)
                        else:
                            # 回退: 扩展到5类
                            prob_k = torch.sigmoid(outputs)
                            batch_probs = torch.zeros(batch_x.size(0), num_classes, device=config.DEVICE)
                            batch_probs[:, 0] = 1 - prob_k[:, 0]
                            for i in range(1, prob_k.size(1)):
                                batch_probs[:, i] = prob_k[:, i-1] - prob_k[:, i]
                            batch_probs[:, -1] = prob_k[:, -1]
                            preds = torch.argmax(batch_probs, dim=1)
                    else:
                        preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())

            print(f"  Acc: {test_metrics['accuracy']:.4f}, QWK: {test_metrics['qwk']:.4f}, MAE: {test_metrics['mae']:.4f}")

            # 计算混淆矩阵
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(np.array(all_labels), np.array(all_preds))

            # 保存结果（包含混淆矩阵）
            fold_results[fold_idx][loss_name] = {
                'accuracy': test_metrics['accuracy'],
                'adjacent_accuracy': test_metrics['adjacent_accuracy'],
                'macro_f1': test_metrics['macro_f1'],
                'weighted_f1': test_metrics['weighted_f1'],
                'qwk': test_metrics['qwk'],
                'mae': test_metrics['mae'],
                'confusion_matrix': cm.tolist()  # 添加混淆矩阵数据
            }

            # 保存预测结果
            fold_predictions[fold_idx][loss_name] = {
                'y_true': np.array(all_labels),
                'y_pred': np.array(all_preds)
            }

    return fold_results, fold_predictions


def run_repeated_kfold_cv(config, loss_configs, X, y, le, n_repeats=3, seeds=None, save_dir=None):
    """
    运行多次五折交叉验证

    Args:
        config: 配置对象
        loss_configs: loss配置
        X, y, le: 数据和标签编码器
        n_repeats: 重复次数
        seeds: 随机种子列表
        save_dir: 保存目录（由main函数传入，避免重复创建）

    Returns:
        all_results: {repeat_idx: {fold_idx: {loss_name: metrics}}}
        all_predictions: {repeat_idx: {fold_idx: {loss_name: {"y_true": ..., "y_pred": ...}}}}
        summary: 汇总统计
        save_dir: 保存目录
    """
    if seeds is None:
        seeds = [42, 123, 456][:n_repeats]

    print(f"\n总样本数: {len(X)}")
    print(f"标签分布: {np.bincount(y)}")
    print(f"标签映射: {le.classes_}")

    all_results = {}
    all_predictions = {}
    repeat_summaries = []

    # 创建weights子目录
    weights_dir = os.path.join(save_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 运行多次五折交叉验证
    for repeat_idx in range(n_repeats):
        seed = seeds[repeat_idx]

        fold_results, fold_predictions = run_single_repeat(
            config, loss_configs, X, y, le, repeat_idx, seed, weights_dir
        )

        all_results[f"repeat_{repeat_idx}"] = fold_results
        all_predictions[f"repeat_{repeat_idx}"] = fold_predictions

        # 计算本次运行的汇总
        repeat_summary = compute_repeat_summary(fold_results)
        repeat_summaries.append(repeat_summary)

        # 打印本次运行结果
        print(f"\n--- Repeat {repeat_idx + 1} 汇总 ---")
        print_repeat_summary(repeat_summary, le)

    # 计算多层次汇总
    summary = compute_hierarchical_summary(all_results, n_repeats, le)

    return all_results, all_predictions, summary, save_dir


def compute_repeat_summary(fold_results):
    """计算单次运行的汇总统计"""
    summary = {}

    for loss_name in fold_results[0].keys():
        metrics = ['accuracy', 'adjacent_accuracy', 'macro_f1', 'weighted_f1', 'qwk', 'mae']

        summary[loss_name] = {}
        for metric in metrics:
            values = [fold_results[f][loss_name][metric] for f in fold_results]
            summary[loss_name][f"{metric}_mean"] = np.mean(values)
            summary[loss_name][f"{metric}_std"] = np.std(values)

    return summary


def compute_hierarchical_summary(all_results, n_repeats, le):
    """
    计算多层次统计汇总

    Returns:
        summary: {
            "within_repeat": {...},
            "between_repeat": {...},
            "overall": {...}
        }
    """
    summary = {
        "within_repeat": {},
        "between_repeat": {},
        "overall": {}
    }

    loss_names = list(all_results["repeat_0"][0].keys())

    for loss_name in loss_names:
        # 收集所有结果
        all_values = {}  # {metric: [all_values]}
        repeat_means = {}  # {metric: [repeat_means]}

        # 初始化字典，收集所有指标
        for metric in ['accuracy', 'adjacent_accuracy', 'macro_f1', 'weighted_f1', 'qwk', 'mae']:
            all_values[metric] = []
            repeat_means[metric] = []

            for repeat_key in all_results:
                fold_data = all_results[repeat_key]

                # 收集该 repeat 所有 fold 的值
                fold_values = [fold_data[f][loss_name][metric] for f in fold_data]
                all_values[metric].extend(fold_values)

                # 该 repeat 的均值
                repeat_means[metric].append(np.mean(fold_values))

        # 为每个指标计算统计量，并汇总到一个字典中
        summary["overall"][loss_name] = {}

        for metric in ['accuracy', 'adjacent_accuracy', 'macro_f1', 'weighted_f1', 'qwk', 'mae']:
            # 总体统计
            mean_val = np.mean(all_values[metric])
            std_val = np.std(all_values[metric], ddof=1)

            # 95% 置信区间
            if len(all_values[metric]) >= 2:
                ci_low, ci_high = stats.t.interval(0.95, len(all_values[metric]) - 1,
                                                   loc=mean_val, scale=std_val / np.sqrt(len(all_values[metric])))
            else:
                ci_low = ci_high = mean_val

            # 添加到汇总字典
            summary["overall"][loss_name].update({
                f"{metric}_mean": mean_val,
                f"{metric}_std": std_val,
                f"{metric}_ci_low": ci_low,
                f"{metric}_ci_high": ci_high,
                f"{metric}_n": len(all_values[metric])
            })

        # 运行内标准差
        within_stds = []
        for repeat_key in all_results:
            fold_data = all_results[repeat_key]
            for metric in ['accuracy', 'qwk', 'mae']:
                fold_values = [fold_data[f][loss_name][metric] for f in fold_data]
                within_stds.append(np.std(fold_values))

        summary["within_repeat"][loss_name] = {
            "fold_avg_std": np.mean(within_stds)
        }

        # 运行间标准差 - 为每个指标计算并汇总
        summary["between_repeat"][loss_name] = {}
        for metric in ['accuracy', 'qwk', 'mae']:
            summary["between_repeat"][loss_name].update({
                f"{metric}_std": np.std(repeat_means[metric])
            })

    return summary


def print_repeat_summary(summary, le):
    """打印单次运行汇总"""
    print(f"\n{'Loss':<12} {'Acc':>8} {'QWK':>8} {'MAE':>8}")
    print("-" * 40)

    for loss_name, metrics in summary.items():
        print(f"{loss_name:<12} {metrics['accuracy_mean']:>8.4f} "
              f"{metrics['qwk_mean']:>8.4f} {metrics['mae_mean']:>8.4f}")


def print_final_summary(summary, le):
    """打印最终汇总结果"""
    print(f"\n{'='*100}")
    print(f"  多次五折交叉验证最终结果")
    n_evals = summary["overall"][list(summary["overall"].keys())[0]]["accuracy_n"]
    print(f"  数据: {len(le.classes_)}类 × {n_evals}个评估点")
    print(f"{'='*100}")

    print(f"\n{'Loss':<12} {'Accuracy':>20} {'QWK':>20} {'MAE':>20}")
    print("-" * 75)

    for loss_name in list(summary["overall"].keys()):
        acc_mean = summary["overall"][loss_name]["accuracy_mean"]
        acc_std = summary["overall"][loss_name]["accuracy_std"]
        acc_ci_low = summary["overall"][loss_name]["accuracy_ci_low"]
        acc_ci_high = summary["overall"][loss_name]["accuracy_ci_high"]

        qwk_mean = summary["overall"][loss_name]["qwk_mean"]
        qwk_std = summary["overall"][loss_name]["qwk_std"]
        qwk_ci_low = summary["overall"][loss_name]["qwk_ci_low"]
        qwk_ci_high = summary["overall"][loss_name]["qwk_ci_high"]

        mae_mean = summary["overall"][loss_name]["mae_mean"]
        mae_std = summary["overall"][loss_name]["mae_std"]
        mae_ci_low = summary["overall"][loss_name]["mae_ci_low"]
        mae_ci_high = summary["overall"][loss_name]["mae_ci_high"]

        print(f"{loss_name:<12} {acc_mean:>7.4f}±{acc_std:<5.4f} [{acc_ci_low:.4f},{acc_ci_high:.4f}]  "
              f"{qwk_mean:>7.4f}±{qwk_std:<5.4f} [{qwk_ci_low:.4f},{qwk_ci_high:.4f}]  "
              f"{mae_mean:>7.4f}±{mae_std:<5.4f} [{mae_ci_low:.4f},{mae_ci_high:.4f}]")

    print(f"\n说明: 均值±标准差 [95%置信区间]")
    print(f"{'='*100}")


def plot_comparison(summary, save_dir):
    """绘制对比图"""
    os.makedirs(save_dir, exist_ok=True)

    loss_names = list(summary["overall"].keys())

    # 准备数据
    metrics = ['accuracy', 'qwk', 'mae']
    metric_labels = ['Accuracy', 'QWK', 'MAE']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        means = []
        stds = []
        cis_low = []
        cis_high = []

        for loss_name in loss_names:
            means.append(summary["overall"][loss_name][f"{metric}_mean"])
            stds.append(summary["overall"][loss_name][f"{metric}_std"])
            cis_low.append(summary["overall"][loss_name][f"{metric}_ci_low"])
            cis_high.append(summary["overall"][loss_name][f"{metric}_ci_high"])

        x_pos = np.arange(len(loss_names))

        # 绘制柱状图
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, error_kw={'linewidth': 2})

        # 添加误差条
        ax.errorbar(x_pos, cis_low, [means[i] - cis_low[i] for i in range(len(means))],
                   fmt='none', ecolor='red', alpha=0.5, linewidth=1)
        ax.errorbar(x_pos, cis_high, [cis_high[i] - means[i] for i in range(len(means))],
                   fmt='none', ecolor='red', alpha=0.5, linewidth=1)

        ax.set_ylabel(label, fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(loss_names, rotation=45, ha='right', fontsize=9)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(stds)*0.02,
                   f'{means[i]:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Repeated K-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'repeated_kfold_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n对比图已保存: {save_path}")


def plot_boxplot(all_results, summary, save_dir):
    """绘制箱线图，展示各loss函数在不同指标上的分布"""
    os.makedirs(save_dir, exist_ok=True)

    loss_names = list(summary["overall"].keys())
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

        # 设置x轴刻度标签旋转（与comparison图一致）
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

        # 添加均值点
        for i, data in enumerate(data_for_metric):
            mean_val = np.mean(data)
            ax.plot(i + 1, mean_val, 'go', markersize=8, label='Mean' if i == 0 else "")

        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Loss Function', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.set_ylim(0, 1.05 if metric != 'mae' else None)

        # 添加样本量标注
        for i, data in enumerate(data_for_metric):
            ax.text(i + 1, ax.get_ylim()[1] - 0.05, f"n={len(data)}",
                   ha='center', fontsize=8, color='gray')

    plt.suptitle('Repeated K-Fold Cross-Validation: Distribution Analysis (Box Plot)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'repeated_kfold_boxplot.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"箱线图已保存: {save_path}")


def plot_confusion_matrices(all_predictions, summary, save_dir, n_classes=5, class_names=None):
    """
    绘制混淆矩阵对比图（聚合所有fold的结果）

    Args:
        all_predictions: {repeat_idx: {fold_idx: {loss_name: {"y_true": ..., "y_pred": ...}}}}
        summary: 汇总统计
        save_dir: 保存目录
        n_classes: 类别数量
        class_names: 类别名称
    """
    from sklearn.metrics import confusion_matrix

    if class_names is None:
        class_names = [f'F{i}' for i in range(n_classes)]

    os.makedirs(save_dir, exist_ok=True)

    # 绘制所有loss函数
    loss_names = list(summary["overall"].keys())
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
        ax.set_title(f'{loss_name.upper()}', fontsize=12, fontweight='bold')
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

    plt.suptitle('Confusion Matrices (Aggregated over 3 repeats × 5 folds)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"混淆矩阵已保存: {save_path}")

    # 单独绘制最佳模型（CORAL）的混淆矩阵，用于论文主图
    if 'coral' in aggregated_cms:
        fig, ax = plt.subplots(figsize=(8, 7))

        cm = aggregated_cms['coral']
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=cm.max())

        for i in range(n_classes):
            for j in range(n_classes):
                text_color = 'white' if cm_norm[i, j] > 50 else 'black'
                text = f'{cm[i, j]}\n({cm_norm[i, j]:.1f}%)'
                ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=11)

        ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
        ax.set_title('CORAL: Confusion Matrix (Aggregated over 15 evaluations)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, fontsize=11)
        ax.set_yticklabels(class_names, fontsize=11)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Count', rotation=270, labelpad=20, fontsize=11)

        plt.tight_layout()

        save_path = os.path.join(save_dir, 'confusion_matrix_coral.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"CORAL混淆矩阵已保存: {save_path}")


def save_results(all_results, all_predictions, summary, save_dir):
    """保存所有结果"""
    # 保存汇总统计
    summary_path = os.path.join(save_dir, 'repeated_kfold_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        # 转换 numpy 类型为原生 Python 类型
        serializable_summary = {}
        for level in summary:
            serializable_summary[level] = {}
            for loss_name, metrics in summary[level].items():
                serializable_summary[level][loss_name] = {}
                for key, value in metrics.items():
                    if isinstance(value, (np.floating, np.integer)):
                        serializable_summary[level][loss_name][key] = float(value)
                    elif isinstance(value, np.ndarray):
                        serializable_summary[level][loss_name][key] = value.tolist()
                    else:
                        serializable_summary[level][loss_name][key] = value
        json.dump(serializable_summary, f, ensure_ascii=False, indent=2)

    # 保存详细结果
    details = {}
    for repeat_key, fold_data in all_results.items():
        details[repeat_key] = {}
        for fold_idx, loss_data in fold_data.items():
            details[repeat_key][fold_idx] = {}
            for loss_name, metrics in loss_data.items():
                details[repeat_key][fold_idx][loss_name] = metrics

    details_path = os.path.join(save_dir, 'repeated_kfold_details.json')
    with open(details_path, 'w', encoding='utf-8') as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    # 保存预测结果（用于绘制混淆矩阵）
    predictions = {}
    for repeat_key, fold_data in all_predictions.items():
        predictions[repeat_key] = {}
        for fold_idx, loss_data in fold_data.items():
            predictions[repeat_key][fold_idx] = {}
            for loss_name, pred_data in loss_data.items():
                predictions[repeat_key][fold_idx][loss_name] = {
                    'y_true': pred_data['y_true'].tolist(),
                    'y_pred': pred_data['y_pred'].tolist()
                }

    predictions_path = os.path.join(save_dir, 'repeated_kfold_predictions.json')
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"结果已保存:")
    print(f"  - {summary_path}")
    print(f"  - {details_path}")
    print(f"  - {predictions_path}")


def create_report(summary, save_dir):
    """创建文本报告"""
    report_path = os.path.join(save_dir, 'experiment_report.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 多次五折交叉验证实验报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 实验配置\n\n")
        f.write("- **折数**: 5\n")
        f.write("- **重复次数**: 3\n")
        f.write("- **随机种子**: [42, 123, 456]\n")
        f.write("- **总评估点**: 75 (5折 × 3次 × 5个模型)\n\n")

        f.write("## 结果汇总\n\n")
        f.write("### 主要指标 (均值 ± 标准差 [95%置信区间])\n\n")

        f.write("| Loss | Accuracy | QWK | MAE |\n")
        f.write("|------|----------|-----|-----|\n")

        for loss_name in summary["overall"].keys():
            acc = summary["overall"][loss_name]
            qwk = summary["overall"][loss_name]
            mae = summary["overall"][loss_name]

            f.write(f"| **{loss_name}** | "
                   f"{acc['accuracy_mean']:.4f}±{acc['accuracy_std']:.4f} [{acc['accuracy_ci_low']:.4f}, {acc['accuracy_ci_high']:.4f}] | "
                   f"{qwk['qwk_mean']:.4f}±{qwk['qwk_std']:.4f} [{qwk['qwk_ci_low']:.4f}, {qwk['qwk_ci_high']:.4f}] | "
                   f"{mae['mae_mean']:.4f}±{mae['mae_std']:.4f} [{mae['mae_ci_low']:.4f}, {mae['mae_ci_high']:.4f}] |\n")

        f.write("\n### 稳定性分析\n\n")
        f.write("| Loss | 五折变异 | 划分敏感度 |\n")
        f.write("|------|----------|------------|\n")

        for loss_name in summary["overall"].keys():
            within = summary["within_repeat"][loss_name]["fold_avg_std"]
            between = summary["between_repeat"][loss_name]["accuracy_std"]

            # 稳定性评级
            ratio = between / (within + 1e-8)
            if ratio < 0.8:
                stability = "✅ 优秀"
            elif ratio < 1.5:
                stability = "⚠️ 一般"
            else:
                stability = "❌ 较差"

            f.write(f"| {loss_name} | {within:.4f} | {between:.4f} {stability} |\n")

        f.write(f"\n*说明: 五折变异小 = 模型收敛好; 划分敏感度低 = 泛化能力强*\n\n")

    print(f"实验报告已保存: {report_path}")


def main():
    """主函数"""
    config = Config()

    # Loss 配置
    loss_configs = {
        "ce":            ("ce", None),
        "cdw_ce":        ("cdw_ce", None),
        "mse":           ("mse", None),
        "cdw_ada":       ("cdw_ada", None),  # 权重版本：下0.75，上0.5
        "cdw_exp":       ("cdw_exp", None),  # 指数版本：下1.3，上1.2
    }

    # 实验参数
    n_repeats = 3
    seeds = [42, 123, 456]

    # 创建输出目录
    version_dir = create_versioned_dir(config.OUTPUT_DIR)

    # 创建日志
    log_file = os.path.join(version_dir, "repeated_kfold.log")
    logger = Logger(log_file)
    sys.stdout = logger

    print(f"\n{'='*100}")
    print(f"  多次五折交叉验证实验")
    print(f"{'='*100}")
    print(f"版本目录: {version_dir}")
    print(f"日志文件: {log_file}")
    print(f"配置: {n_repeats}次重复 × 5折交叉验证")
    print(f"随机种子: {seeds}")

    # 加载数据（只加载一次，传入run_repeated_kfold_cv）
    X, y, le = load_data_full(config)

    # 运行多次五折交叉验证（传入version_dir避免重复创建目录）
    all_results, all_predictions, summary, save_dir = run_repeated_kfold_cv(
        config, loss_configs, X, y, le, n_repeats=n_repeats, seeds=seeds, save_dir=version_dir
    )

    # 创建子目录：json文件和图片分开存放
    json_dir = os.path.join(save_dir, "json")
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 打印最终汇总
    print_final_summary(summary, le)

    # 绘制对比图
    plot_comparison(summary, plots_dir)

    # 绘制箱线图
    plot_boxplot(all_results, summary, plots_dir)

    # 绘制混淆矩阵
    plot_confusion_matrices(all_predictions, summary, plots_dir, n_classes=5, class_names=['F0', 'F1', 'F2', 'F3', 'F4'])

    # 保存结果（JSON文件）
    save_results(all_results, all_predictions, summary, json_dir)

    # 创建报告
    create_report(summary, json_dir)

    print(f"\n{'='*100}")
    print(f"  实验完成！")
    print(f"  所有结果已保存至: {save_dir}")
    print(f"    - JSON文件: {json_dir}")
    print(f"    - 图片: {plots_dir}")
    print(f"{'='*100}")

    logger.close()

    # 恢复标准输出（关闭日志重定向后再生成图表）
    sys.stdout = sys.__stdout__

    # 自动绘制条形图（汇总所有repeat和fold的结果）
    print(f"\n{'='*100}")
    print("  生成汇总条形图...")
    print(f"{'='*100}")

    # 添加 scripts 目录到 Python 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from plot_kfold_barchart import plot_kfold_barchart, plot_combined_comparison

    # 使用 overall 部分绘制条形图（结构相同）
    summary_path = os.path.join(json_dir, 'repeated_kfold_summary.json')
    overall_summary = summary["overall"]  # 提取overall部分
    plot_kfold_barchart(overall_summary, os.path.join(plots_dir, "kfold_barchart_detailed.png"))
    plot_combined_comparison(overall_summary, os.path.join(plots_dir, "kfold_barchart_combined.png"))

    print(f"  条形图已保存至: {plots_dir}")
    print(f"    - kfold_barchart_detailed.png (6指标详细图)")
    print(f"    - kfold_barchart_combined.png (4指标综合图)")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
