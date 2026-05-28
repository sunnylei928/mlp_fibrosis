"""
五折交叉验证主程序
替换原有的单次训练为五折交叉验证
"""
import os
import sys
import copy
import json
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import Config
from core.dataset import FibrosisDataset
from core.model import MLPClassifier, CORALNet
from core.loss import get_loss
from training.train import train_model
from core.evaluate import evaluate_epoch
from training.utils import create_versioned_dir, plot_confusion_matrices


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


def load_data_full(config):
    """加载数据（不划分），返回 X, y, label_encoder"""
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


def run_kfold_cross_validation(config, loss_configs, k=5):
    """
    运行五折交叉验证

    Args:
        config: 配置对象
        loss_configs: loss 配置字典
        k: 折数
    """
    # 添加混淆矩阵导入
    from sklearn.metrics import confusion_matrix

    # 加载完整数据
    X, y, le = load_data_full(config)
    print(f"总样本数: {len(X)}")
    print(f"标签分布: {np.bincount(y)}")
    print(f"标签映射: {le.classes_}")

    # 创建五折分层交叉验证
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.RANDOM_SEED)

    # 存储所有结果
    all_results = {loss_name: [] for loss_name in loss_configs.keys()}

    # 遍历每一折
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*80}")
        print(f"  Fold {fold_idx + 1}/5")
        print(f"{'='*80}")
        print(f"训练集大小: {len(train_idx)}, 测试集大小: {len(test_idx)}")

        # 划分数据
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 从训练集再划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=config.RANDOM_SEED
        )

        print(f"最终划分: 训练={len(X_train)}, 验证={len(X_val)}, 测试={len(X_test)}")
        print(f"标签分布 - 训练: {np.bincount(y_train)}")

        # 标准化（基于当前折的训练集）
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

        # 计算类别权重
        class_counts = np.bincount(y_train)
        class_weights = torch.FloatTensor(1.0 / (class_counts + 1e-6))
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = class_weights.to(config.DEVICE)

        # 遍历每个 loss
        for loss_name, (loss_type, weight) in loss_configs.items():
            print(f"\n--- {loss_name} ---")

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
            elif loss_type == 'cdw_ce_margin':
                loss_kwargs["alpha"] = 1.0
                loss_kwargs["margin"] = 0.05
            elif loss_type == 'cdw_ce_cc':
                loss_kwargs["alpha"] = 0.75  # 临床成本矩阵距离惩罚（v4: 提高alpha增强惩罚强度）

            criterion = get_loss(loss_type, class_weights=weight, **loss_kwargs)
            optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

            # 训练
            history = train_model(
                model, train_loader, val_loader, criterion, optimizer, config,
                loss_name=f"{loss_name}_fold{fold_idx}", save_dir=None
            )

            # 测试
            test_metrics = evaluate_epoch(model, test_loader, criterion, config.DEVICE)

            # 计算混淆矩阵
            cm = confusion_matrix(test_metrics['labels'], test_metrics['preds'])
            test_metrics['confusion_matrix'] = cm

            print(f"Fold {fold_idx + 1} - Acc: {test_metrics['accuracy']:.4f}, "
                  f"QWK: {test_metrics['qwk']:.4f}, MAE: {test_metrics['mae']:.4f}")

            # 保存结果（包含混淆矩阵）
            all_results[loss_name].append(test_metrics)

    return all_results, le


def summarize_kfold_results(all_results):
    """汇总五折交叉验证结果"""
    print(f"\n{'='*80}")
    print("  五折交叉验证汇总结果 (均值 ± 标准差)")
    print(f"{'='*80}")

    print(f"\n{'Loss':<12} {'Accuracy':>22} {'AdjAcc':>20} {'QWK':>20} {'MAE':>20}")
    print("-" * 95)

    summary = {}
    for loss_name, fold_results in all_results.items():
        metrics_summary = {}
        for metric in ['accuracy', 'adjacent_accuracy', 'macro_f1', 'weighted_f1', 'qwk', 'mae']:
            values = [fold[metric] for fold in fold_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            metrics_summary[f"{metric}_mean"] = mean_val
            metrics_summary[f"{metric}_std"] = std_val

        summary[loss_name] = metrics_summary

        acc_mean = metrics_summary['accuracy_mean']
        acc_std = metrics_summary['accuracy_std']
        adjacc_mean = metrics_summary['adjacent_accuracy_mean']
        adjacc_std = metrics_summary['adjacent_accuracy_std']
        qwk_mean = metrics_summary['qwk_mean']
        qwk_std = metrics_summary['qwk_std']
        mae_mean = metrics_summary['mae_mean']
        mae_std = metrics_summary['mae_std']

        print(f"{loss_name:<12} {acc_mean:>7.4f} ± {acc_std:<9.4f} "
              f"{adjacc_mean:>7.4f} ± {adjacc_std:<9.4f} "
              f"{qwk_mean:>7.4f} ± {qwk_std:<9.4f} "
              f"{mae_mean:>7.4f} ± {mae_std:<9.4f}")

    print(f"{'='*80}")

    return summary


def main():
    config = Config()
    version_dir = create_versioned_dir(config.OUTPUT_DIR)

    # 创建日志
    log_file = os.path.join(version_dir, "kfold_training.log")
    logger = Logger(log_file)
    sys.stdout = logger

    print(f"\n{'='*80}")
    print(f"  五折交叉验证实验")
    print(f"{'='*80}")
    print(f"Version output dir: {version_dir}")
    print(f"Log file: {log_file}")
    print(f"K folds: 5")

    # Loss 配置
    loss_configs = {
        "ce":            ("ce", None),
        "cdw_ce":        ("cdw_ce", None),
        "cdw_ce_margin": ("cdw_ce_margin", None),
        "cdw_ce_cc":      ("cdw_ce_cc", None),  # Clinical Cost-Aware Distance (NEW)
        "mse":           ("mse", None),
        "mlp_coral":     ("mlp_coral", None),  # MLP + CORAL loss
        "coral":         ("coral", None),       # CORALNet + CORAL loss (原始)
    }

    # 运行五折交叉验证
    all_results, le = run_kfold_cross_validation(config, loss_configs, k=5)

    # 汇总结果
    summary = summarize_kfold_results(all_results)

    # 保存结果
    import json
    with open(os.path.join(version_dir, 'kfold_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 保存详细结果（包含混淆矩阵）
    detailed_results = {}
    for loss_name, fold_results in all_results.items():
        detailed_results[loss_name] = []
        for i, fold in enumerate(fold_results):
            fold_dict = {}
            for k, v in fold.items():
                if k in ['preds', 'labels']:
                    # 转换为列表
                    fold_dict[k] = v.tolist() if hasattr(v, 'tolist') else list(v)
                elif k == 'confusion_matrix':
                    # 混淆矩阵已经存在，直接保留
                    fold_dict[k] = v.tolist() if hasattr(v, 'tolist') else v
                elif isinstance(v, (np.floating, np.integer)):
                    fold_dict[k] = float(v)
                else:
                    fold_dict[k] = v
            fold_dict['fold'] = i
            detailed_results[loss_name].append(fold_dict)

    with open(os.path.join(version_dir, 'kfold_details.json'), 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    # 绘制混淆矩阵
    print(f"\n{'='*80}")
    print("生成混淆矩阵可视化...")
    print(f"{'='*80}")

    # 为每个 loss 汇总所有 fold 的预测结果
    combined_results = {}
    for loss_name, fold_results in all_results.items():
        all_preds = []
        all_labels = []
        for fold in fold_results:
            if 'preds' in fold and 'labels' in fold:
                all_preds.extend(fold['preds'])
                all_labels.extend(fold['labels'])
        if all_preds:
            combined_results[loss_name] = {
                'preds': np.array(all_preds),
                'labels': np.array(all_labels)
            }

    # 保存汇总混淆矩阵
    if combined_results:
        cm_save_dir = os.path.join(version_dir, "confusion_matrices")
        plot_confusion_matrices(combined_results, le, cm_save_dir)
        print(f"混淆矩阵已保存至: {cm_save_dir}")

    print(f"\n所有结果已保存至: {version_dir}")
    logger.close()

    # 恢复标准输出（关闭日志重定向后再生成图表）
    sys.stdout = sys.__stdout__

    # 自动绘制条形图
    print(f"\n{'='*80}")
    print("生成五折交叉验证条形图...")
    print(f"{'='*80}")

    # 添加 scripts 目录到 Python 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from plot_kfold_barchart import plot_kfold_barchart, plot_combined_comparison

    summary_path = os.path.join(version_dir, 'kfold_summary.json')
    plot_kfold_barchart(summary, os.path.join(version_dir, "kfold_barchart_detailed.png"))
    plot_combined_comparison(summary, os.path.join(version_dir, "kfold_barchart_combined.png"))

    print(f"条形图已保存至: {version_dir}")
    print(f"  - kfold_barchart_detailed.png (6指标详细图)")
    print(f"  - kfold_barchart_combined.png (4指标综合图)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
