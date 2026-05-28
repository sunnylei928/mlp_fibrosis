import torch
import torch.nn as nn
import torch.nn.functional as F


class CDWCELoss(nn.Module):
    """
    Class Distance Weighted Cross-Entropy Loss.
    Source: Polat et al., *Class Distance Weighted Cross Entropy Loss
    for Classification of Disease Severity*, 2025.

    Core idea: the farther the predicted class is from the true class,
    the heavier the penalty.

    Basic formula:
        CDW-CE = -sum_{i=0}^{C-1} log(1 - p_i) * |i - c|^alpha

    where p_i = softmax(logits)_i, c is the ground-truth class index.
    When i = c the weight becomes 0, so correct predictions are not penalised.

    Margin variant (document formula has numerical issues; we clamp
    probabilities to [0, 1-eps] for stability):
        probs' = clamp(probs + margin, max=1-eps)
    """
    def __init__(self, alpha=1.0, margin=0.0, reduction='mean', eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)           # [B, C]
        num_classes = logits.size(-1)
        device = logits.device

        # Distance weights: |i - c|^alpha  [B, C]
        indices = torch.arange(num_classes, device=device).unsqueeze(0)
        c = targets.unsqueeze(1)
        weights = torch.abs(indices - c).float() ** self.alpha

        if self.margin > 0:
            probs = torch.clamp(probs + self.margin, max=1.0 - self.eps)

        loss_matrix = -torch.log(1.0 - probs + self.eps) * weights
        loss_per_sample = loss_matrix.sum(dim=-1)   # [B]

        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        return loss_per_sample


class CDWCEMultiBoundaryLoss(nn.Module):
    """
    Class Distance Weighted Cross-Entropy Loss with Multi-Boundary Dynamic Margin.

    创新点：多边界动态Margin（Multi-Boundary Dynamic Margin）

    临床背景：对于肝纤维化分期 F0-F4
        - F0-F1: 早期/轻度，保守治疗
        - F2-F3: 中期/明显纤维化，需要积极干预
        - F4: 肝硬化，治疗策略完全不同

    关键边界：
        - F1-F2: 轻度 vs 中期（是否开始抗纤维化治疗）
        - F3-F4: 晚期 vs 肝硬化（预后差异巨大）

    核心思想：
        - 在类簇内部（F0-F1, F2-F3, F4单独）使用极小 margin
        - 跨越 F1-F2 或 F3-F4 边界时，施加强力惩罚
        - 保护两个关键临床决策边界

    公式：
        CDW-CE-MB = -sum_{i=0}^{C-1} log(1 - p_i') * |i - c|^alpha

    其中：
        p_i' = clamp(p_i + m(i, c), max=1-eps)
        m(i, c) = {
            m_tiny,     if i 和 c 在同一簇内
            m_medium,   if 跨越 F1-F2 边界
            m_large,    if 跨越 F3-F4 边界
        }

    Args:
        alpha: 距离惩罚的幂指数
        m_tiny: 簇内 margin（极小）
        m_medium: F1-F2 边界 margin（中等）
        m_large: F3-F4 边界 margin（大）
        reduction: 'mean', 'sum', 或 'none'
        eps: 数值稳定性常数
    """
    def __init__(self, alpha=1.2, m_tiny=0.005, m_medium=0.05, m_large=0.08,
                 reduction='mean', eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.m_tiny = m_tiny
        self.m_medium = m_medium
        self.m_large = m_large
        self.reduction = reduction
        self.eps = eps

        # 定义簇：[F0, F1], [F2, F3], [F4]
        self.cluster_boundaries = [1, 3]  # F1/F2 和 F3/F4 之间

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)           # [B, C]
        num_classes = logits.size(-1)
        batch_size = logits.size(0)
        device = logits.device

        # Distance weights: |i - c|^alpha  [B, C]
        indices = torch.arange(num_classes, device=device).unsqueeze(0).expand(batch_size, -1)
        c = targets.unsqueeze(1)
        weights = torch.abs(indices - c).float() ** self.alpha

        # 计算动态 margin
        # 1. 判断 i 和 c 的簇编号
        # 簇0: [F0, F1] → indices <= 1
        # 簇1: [F2, F3] → 2 <= indices <= 3
        # 簇2: [F4] → indices >= 4
        cluster_i = torch.zeros_like(indices, dtype=torch.long)
        cluster_i = torch.where(indices <= 1, torch.zeros_like(cluster_i), cluster_i)
        cluster_i = torch.where((indices >= 2) & (indices <= 3), torch.ones_like(cluster_i), cluster_i)
        cluster_i = torch.where(indices >= 4, 2 * torch.ones_like(cluster_i), cluster_i)

        cluster_c = torch.zeros_like(c, dtype=torch.long)
        cluster_c = torch.where(c <= 1, torch.zeros_like(cluster_c), cluster_c)
        cluster_c = torch.where((c >= 2) & (c <= 3), torch.ones_like(cluster_c), cluster_c)
        cluster_c = torch.where(c >= 4, 2 * torch.ones_like(cluster_c), cluster_c)

        # 2. 根据 cluster 差异确定 margin
        cluster_diff = torch.abs(cluster_i - cluster_c)  # [B, C]

        # 构造 margin 矩阵
        dynamic_margin = torch.zeros_like(probs)
        dynamic_margin = torch.where(cluster_diff == 0, torch.full_like(dynamic_margin, self.m_tiny), dynamic_margin)
        dynamic_margin = torch.where(cluster_diff == 1, torch.full_like(dynamic_margin, self.m_medium), dynamic_margin)
        dynamic_margin = torch.where(cluster_diff >= 2, torch.full_like(dynamic_margin, self.m_large), dynamic_margin)

        # 应用动态 margin
        probs_adjusted = torch.clamp(probs + dynamic_margin, max=1.0 - self.eps)

        loss_matrix = -torch.log(1.0 - probs_adjusted + self.eps) * weights
        loss_per_sample = loss_matrix.sum(dim=-1)   # [B]

        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        return loss_per_sample


class CDWCEClinicalCostLoss(nn.Module):
    """
    Class Distance Weighted Cross-Entropy Loss with Clinical Cost-Aware Distance.

    创新点：临床成本感知距离（Clinical Cost-Aware Distance）

    临床背景：
        - 漏诊（False Negative）：有病判为无病，延误治疗，风险极高
        - 过诊（False Positive）：无病判为有病，可能过度治疗，风险相对较低
        - 肝纤维化 F4（肝硬化）漏诊的后果最严重

    核心思想：
        - 用非对称的临床风险矩阵代替对称的距离权重 |i-c|
        - 漏诊方向（病情低估）的惩罚 > 过诊方向（病情高估）
        - F4漏诊（F4→F0）的惩罚 >> F0过诊（F0→F4）

    公式：
        CDW-CE-CC = -sum_{i=0}^{C-1} log(1 - p_i) * Cost[c, i]^alpha

    其中 Cost[c, i] 是临床风险矩阵（v2: 范围0-4，更平衡）：
        - 对角线：0（正确预测）
        - 上三角（病情高估）：低代价（0.3-3.0）
        - 下三角（病情低估）：高代价（0.5-4.0）
        - F4漏诊：最高代价（4.0）
        - 相邻类别间惩罚较低（0.3-1.0）

    Args:
        alpha: 距离惩罚的幂指数（默认1.0）
        cost_matrix: 临床成本矩阵 [5, 5]，如为None则使用默认矩阵
        reduction: 'mean', 'sum', 或 'none'
        eps: 数值稳定性常数
    """
    def __init__(self, alpha=1.0, cost_matrix=None, reduction='mean', eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps

        # 默认临床成本矩阵（行=真实，列=预测）
        # 原则：漏诊(下三角) > 过诊(上三角)，F4漏诊惩罚最大
        # v2: 降低惩罚范围（0-4），让模型不那么保守
        if cost_matrix is None:
            #      F0   F1   F2   F3   F4
            cost_matrix = torch.tensor([
                [0.0, 0.3, 0.8, 1.5, 3.0],  # 真实F0: 过诊F4代价3
                [0.5, 0.0, 0.3, 1.0, 2.5],  # 真实F1: 漏诊F4代价2.5
                [1.0, 0.5, 0.0, 0.3, 1.5],  # 真实F2: 漏诊F4代价1.5
                [2.0, 1.2, 0.5, 0.0, 0.8],  # 真实F3: 漏诊F4代价0.8
                [4.0, 3.0, 2.0, 1.0, 0.0]   # 真实F4: 过诊F0代价4，漏诊F3代价1.0
            ])
        self.register_buffer('cost_matrix', cost_matrix)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)           # [B, C]
        num_classes = logits.size(-1)
        batch_size = logits.size(0)
        device = logits.device

        # 使用临床成本矩阵代替距离权重
        # weights[b, i] = cost_matrix[targets[b], i] ^ alpha
        c = targets.unsqueeze(1)  # [B, 1]

        # 对每个样本，根据其真实类别获取对应的成本行
        weights = torch.zeros(batch_size, num_classes, device=device)
        for b in range(batch_size):
            true_class = targets[b].item()
            weights[b] = self.cost_matrix[true_class] ** self.alpha

        loss_matrix = -torch.log(1.0 - probs + self.eps) * weights
        loss_per_sample = loss_matrix.sum(dim=-1)   # [B]

        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        return loss_per_sample


class CORALNetLoss(nn.Module):
    """
    CORALNet ordinal-regression loss.
    Source: Saito et al., *Evaluation of ultrasonic fibrosis diagnostic
    system using convolutional network for ordinal regression*, 2021.

    Transforms a K-class ordinal problem into K-1 binary sub-problems.
    Input logits must have shape [B, K-1], produced by a network whose
    last layer uses **shared weights + K-1 independent biases**.

    At the loss level this is mathematically equivalent to CORN-style
    BCEWithLogitsLoss; the critical difference lies in the network head
    (see model.py).
    """
    def __init__(self, num_classes, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, targets):
        # logits: [B, K-1], targets: [B] in {0, ..., K-1}
        batch_size = logits.size(0)
        binary_targets = torch.zeros_like(logits)
        for i in range(batch_size):
            binary_targets[i, :targets[i]] = 1.0
        return self.bce(logits, binary_targets)

    @torch.no_grad()
    def predict(self, logits):
        """
        CORALNet inference: accumulate K-1 binary probabilities.
        pred = sum( sigmoid(logits_k) > 0.5 )
        """
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1).long()


class MSEClassificationLoss(nn.Module):
    """
    MSE adapted for a multi-class classifier with C-dim logits output.
    Converts hard labels to one-hot and computes MSE against softmax probs.
    This allows using MSE without changing the model architecture.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        num_classes = logits.size(-1)
        one_hot = F.one_hot(targets, num_classes=num_classes).float()
        return self.mse(probs, one_hot)


class CDWCEProbLoss(nn.Module):
    """
    Class Distance Weighted Cross-Entropy Loss with Prediction Confidence Weighting.

    Extends CDW-CE by multiplying the entire loss by the predicted class probability.
    This adds additional penalty to high-confidence wrong predictions.

    Formula:
        CDW-CE-Prob = p_y_hat * (-sum_{i=0}^{C-1} log(1 - p_i) * |i - c|^alpha)

    where:
        p_i = softmax(logits)_i
        y_hat = argmax_i p_i (predicted class)
        p_y_hat = max_i p_i (model confidence in prediction)
        c = ground-truth class index

    The p_y_hat factor increases the penalty when the model makes wrong
    predictions with high confidence.
    """
    def __init__(self, alpha=1.0, margin=0.0, reduction='mean', eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)           # [B, C]
        num_classes = logits.size(-1)
        device = logits.device

        # Distance weights: |i - c|^alpha  [B, C]
        indices = torch.arange(num_classes, device=device).unsqueeze(0)
        c = targets.unsqueeze(1)
        weights = torch.abs(indices - c).float() ** self.alpha

        if self.margin > 0:
            probs = torch.clamp(probs + self.margin, max=1.0 - self.eps)

        # Base CDW-CE loss
        loss_matrix = -torch.log(1.0 - probs + self.eps) * weights
        loss_per_sample = loss_matrix.sum(dim=-1)   # [B]

        # Multiply by max predicted probability (model confidence)
        pred_probs = probs.max(dim=-1)[0]  # [B]
        loss_per_sample = loss_per_sample * pred_probs

        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        return loss_per_sample


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        n_classes = inputs.size(-1)
        one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        smoothed = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        return (-smoothed * log_probs).sum(dim=-1).mean()

class OrdinalCrossEntropy(nn.Module):
    """
    CORN-style ordinal regression loss.
    Treats K-class ordinal problem as K-1 binary classification tasks.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # logits: [B, num_classes-1]
        # targets: [B] in {0, 1, ..., num_classes-1}
        batch_size = logits.size(0)
        # Create binary labels: for class k, first k tasks are 1, rest are 0
        binary_targets = torch.zeros_like(logits)
        for i in range(batch_size):
            binary_targets[i, :targets[i]] = 1.0
        return self.bce(logits, binary_targets)


def get_loss(name, class_weights=None, num_classes=5, device='cpu',
             alpha=1.0, margin=0.05, cost_matrix=None):
    """
    Factory for loss functions.

    Available losses:
        'ce'                  – Standard Cross-Entropy
        'cdw_ce'              – CDW-CE (alpha controls distance-penalty strength)
        'cdw_ce_margin'       – CDW-CE with additive margin
        'cdw_ce_prob'         – CDW-CE with prediction confidence weighting
        'cdw_ce_cc'           – CDW-CE with Clinical Cost-Aware Distance (NEW)
        'coral'               – CORALNet ordinal loss (expects logits [B, K-1])
        'mlp_coral'           – Same as 'coral', but used with MLP architecture
        'mse'                 – MSE on softmax probabilities vs one-hot targets
    """
    if name == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif name == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif name == 'focal':
        return FocalLoss(alpha=class_weights, gamma=2.0)
    elif name == 'label_smoothing':
        return LabelSmoothingCrossEntropy(smoothing=0.1)
    elif name == 'ordinal':
        return OrdinalCrossEntropy(num_classes=num_classes)
    elif name == 'cdw_ce':
        return CDWCELoss(alpha=alpha, margin=0.0)
    elif name == 'cdw_ce_margin':
        return CDWCELoss(alpha=alpha, margin=margin)
    elif name == 'cdw_ce_prob':
        return CDWCEProbLoss(alpha=alpha, margin=margin)
    elif name == 'cdw_ce_cc':
        return CDWCEClinicalCostLoss(alpha=alpha, cost_matrix=cost_matrix)
    elif name == 'coral' or name == 'mlp_coral':
        return CORALNetLoss(num_classes=num_classes)
    elif name == 'mse':
        return MSEClassificationLoss()
    else:
        raise ValueError(f"Unknown loss: {name}")
