import os
import sys
import copy
import numpy as np
import torch
import torch.optim as optim
from config import Config
from core.dataset import load_data
from core.model import MLPClassifier, CORALNet
from core.loss import get_loss
from training.train import train_model
from core.evaluate import evaluate_epoch, print_report
from training.utils import (save_results, plot_comparison, plot_confusion_matrices,
                   create_versioned_dir, save_run_metadata)
from visualization.plot_prediction_probs import plot_all_losses_probs, plot_ordinal_comparison, plot_smoothness_statistics

class Logger:
    """将 print 输出同时打印到控制台和日志文件"""
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


def save_full_config(config, loss_configs, save_dir):
    """保存完整的配置信息"""
    config_dict = {}

    # 保存所有 Config 类的属性
    for attr in dir(config):
        if not attr.startswith('_'):
            value = getattr(config, attr)
            if not callable(value):
                # 转换为可序列化的类型
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    config_dict[attr] = value
                elif hasattr(value, 'device'):  # torch.device
                    config_dict[attr] = str(value)
                else:
                    config_dict[attr] = str(value)

    # 添加 loss 配置
    config_dict['loss_configs'] = loss_configs

    import json
    with open(os.path.join(save_dir, 'full_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    print(f"完整配置已保存至: {os.path.join(save_dir, 'full_config.json')}")


def main():
    config = Config()
    version_dir = create_versioned_dir(config.OUTPUT_DIR)

    # 创建日志文件
    log_file = os.path.join(version_dir, "training.log")
    logger = Logger(log_file)
    sys.stdout = logger

    print(f"\n{'='*60}")
    print(f"Version output dir: {version_dir}")
    print(f"Log file: {log_file}")
    print(f"{'='*60}")

    # 创建权重保存目录
    weights_dir = os.path.join(version_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    print(f"权重保存目录: {weights_dir}")

    # Load data
    train_loader, val_loader, test_loader, input_dim, num_classes, class_weights, le = load_data(config)
    print(f"\n数据集信息:")
    print(f"  Input dim: {input_dim}")
    print(f"  Classes: {num_classes}")
    print(f"  Class weights: {class_weights.cpu().numpy()}")
    print(f"  训练集大小: {len(train_loader.dataset)}")
    print(f"  验证集大小: {len(val_loader.dataset)}")
    print(f"  测试集大小: {len(test_loader.dataset)}")

    # 检查标签分布
    train_y = [y for _, y in train_loader]
    train_labels = torch.cat(train_y).numpy()
    print(f"  训练集标签分布: {np.bincount(train_labels)}")

    val_y = [y for _, y in val_loader]
    val_labels = torch.cat(val_y).numpy()
    print(f"  验证集标签分布: {np.bincount(val_labels)}")

    test_y = [y for _, y in test_loader]
    test_labels = torch.cat(test_y).numpy()
    print(f"  测试集标签分布: {np.bincount(test_labels)}")
    print(f"  标签映射: {le.classes_}")

    # Loss functions to compare
    loss_configs = {
        "ce":              ("ce", None),
        "cdw_ce":          ("cdw_ce", None),
        "mse":             ("mse", None),
        "cdw_ada":         ("cdw_ada", None),  # 权重版本：下0.75，上0.5
        "cdw_exp":         ("cdw_exp", None),  # 指数版本：下1.3，上1.2
    }

    # 保存完整配置
    save_full_config(config, loss_configs, version_dir)

    results = {}
    histories = {}
    best_models = {}

    for loss_name, (loss_type, weight) in loss_configs.items():
        print(f"\n{'='*60}")
        print(f"Training with Loss: {loss_name}")
        print(f"{'='*60}")

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
            'arch': 'CORALNet' if loss_type == 'coral' else ('MLP-Ordinal' if loss_type == 'mlp_coral' else 'MLP'),
            'input_dim': input_dim,
            'num_classes': num_classes
        }
        log_model_params(config, loss_type, loss_kwargs, model_params, criterion)

        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        history = train_model(
            model, train_loader, val_loader, criterion, optimizer, config,
            loss_name=loss_name, save_dir=weights_dir  # 权重保存到 weights 目录
        )
        histories[loss_name] = history
        best_models[loss_name.upper()] = copy.deepcopy(model)

        # Evaluate on test set
        test_metrics = evaluate_epoch(model, test_loader, criterion, config.DEVICE)
        print_report(test_metrics, le, title=f"Test ({loss_name})")
        results[loss_name] = test_metrics

    # Summary comparison
    print(f"\n{'='*60}")
    print("  Final Comparison (Test Set)")
    print(f"{'='*60}")
    print(f"{'Loss':<16} {'Acc':>8} {'AdjAcc':>8} {'MacroF1':>8} {'QWK':>8} {'MAE':>8}")
    print("-" * 65)
    for name, m in results.items():
        print(f"{name:<16} {m['accuracy']:>8.4f} {m['adjacent_accuracy']:>8.4f} "
              f"{m['macro_f1']:>8.4f} {m['qwk']:>8.4f} {m['mae']:>8.4f}")
    print(f"{'='*60}")

    # 保存结果和图表
    save_results(results, version_dir)
    plot_comparison(results, histories, version_dir)
    plot_confusion_matrices(results, le, version_dir)

    # === 新增：预测概率折线图 ===
    print(f"\n{'='*60}")
    print("生成预测概率可视化...")
    print(f"{'='*60}")

    # 准备模型字典用于可视化
    models_dict = {}
    for loss_name, model in best_models.items():
        loss_type = loss_configs.get(loss_name.lower(), (loss_name.lower(), None))[0]
        criterion = get_loss(loss_type, num_classes=num_classes, device=config.DEVICE)
        models_dict[loss_name] = (model, criterion)

    label_names = [le.classes_[i] for i in range(len(le.classes_))]
    probs_plot_dir = os.path.join(version_dir, "probability_plots")

    # 1. 生成每个 loss 的预测概率折线图
    print("生成各模型的概率折线图...")
    plot_all_losses_probs(
        models_dict=models_dict,
        test_loader=test_loader,
        device=config.DEVICE,
        label_names=label_names,
        save_dir=probs_plot_dir,
        num_samples=9
    )

    # 2. 生成平滑性统计对比图
    print("生成平滑性统计对比图...")
    plot_smoothness_statistics(
        models_dict=models_dict,
        test_loader=test_loader,
        device=config.DEVICE,
        label_names=label_names,
        save_dir=probs_plot_dir
    )

    # 3. 生成有序回归 vs 标准回归的详细对比图
    if 'CORAL' in best_models and 'CE' in best_models:
        ordinal_comparison_dir = os.path.join(probs_plot_dir, "ordinal_vs_standard")
        print("生成 CORAL vs CE 对比图...")
        plot_ordinal_comparison(
            ordinal_model=best_models['CORAL'],
            ordinal_loss_name='CORAL',
            standard_model=best_models['CE'],
            standard_loss_name='CE',
            test_loader=test_loader,
            device=config.DEVICE,
            label_names=label_names,
            save_dir=ordinal_comparison_dir,
            num_samples=9
        )

    save_run_metadata(config, model, loss_configs, version_dir)

    print(f"\n{'='*60}")
    print(f"所有结果已保存至: {version_dir}")
    print(f"  - 训练日志: training.log")
    print(f"  - 模型权重: weights/")
    print(f"  - 概率可视化: probability_plots/")
    print(f"  - 完整配置: full_config.json")
    print(f"{'='*60}")

    # 关闭日志
    logger.close()


if __name__ == "__main__":
    main()
