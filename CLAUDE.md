# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLP-based framework for liver fibrosis staging (F0-F4) from clinical biomarkers. The research focus is comparing loss functions on ordinal multi-class classification with class imbalance. Key architectural distinction: **CORALNet is NOT an MLP with modified output dimensions** - it uses a fundamentally different architecture with shared weights and learnable ordinal biases.

## Common Commands

```bash
# Single training run (quick iteration)
python scripts/main.py

# 5-fold cross-validation (more reliable)
python scripts/main_kfold.py

# Repeated 5-fold CV (recommended for research)
# 3 repeats × 5 folds = 15 evaluations per loss
python scripts/main_repeated_kfold.py

# Interactive menu (all options)
python run.py

# Generate plots from saved weights
python scripts/plot_from_weights.py

# Inference with trained model
python scripts/inference.py --weight outputs/weights/best_coral_r0_f0.pt --data data/深三数据useful_填充.xlsx
```

## Architecture

### Models (`core/model.py`)

**MLPClassifier**:
- Architecture: `Linear → BatchNorm1d → ReLU → Dropout` repeated per hidden dims
- **Key parameter**: `ordinal_head` controls output dimension
  - `ordinal_head=False`: Output `[B, 5]` logits for standard classification (CE, MSE, Focal, CDW-CE)
  - `ordinal_head=True`: Output `[B, 4]` logits for ordinal losses with special bias initialization
- **Used for**: CE, MSE, Focal, CDW-CE, MLP_CORAL

**CORALNet** (original implementation):
- Shared feature backbone + single shared linear (output 1-dim) + K-1 learnable biases
- Output: `[B, K-1]` via `z + biases` (broadcast)
- **Used for**: CORAL (original method with shared weights)

### Loss Functions (`core/loss.py`)

Factory pattern via `get_loss(name, **kwargs)`:

**Currently active in training:**
- `ce`: Standard cross-entropy
- `cdw_ce`: Class Distance Weighted CE (Polat et al., 2025)
- `mse`: MSE on softmax probabilities
- `cdw_ada`: Adaptive clinical cost-aware CE with asymmetric weights (下三角0.75，上三角0.5, alpha=1.2)
- `cdw_exp`: Exponential distance CE (下三角base=1.3，上三角base=1.2)

**Other available losses (not currently in training):**
- `cdw_ce_margin`: CDW-CE with margin
- `cdw_ce_prob`: CDW-CE with prediction confidence weighting
- `cdw_ce_cc`: Manual clinical cost matrix (older version)
- `coral`: CORAL ordinal loss (Saito et al., 2021) - used with CORALNet
- `mlp_coral`: Same CORAL loss - used with MLP (for ablation study)
- `focal`: Focal loss (gamma=2.0)
- `label_smoothing`: Label smoothing CE (smoothing=0.1)

**⚠️ Failed experiments (DO NOT USE):**
- `cdw_exp_w`: Exponential + class weights (caused Accuracy collapse 51%→16%)
- `cdw_be`: Boundary enhancement (caused Accuracy collapse 51%→25%)

> See `docs/loss_functions_comparison.md` for detailed mathematical formulas, experiment results, and failure analysis.

**Loss Function Formulas**:

| Loss | Formula |
|------|---------|
| CE | $-\log(p_c)$ where $p_c$ is true class probability |
| CDW-CE | $-\sum_{i} \log(1-p_i) \cdot \|i-c\|^\alpha$ |
| CDW-CE-Margin | Same, with $p_i' = \min(p_i + m, 1-\epsilon)$ |
| CDW-CE-Prob | $p_{\hat{y}} \cdot \left(-\sum_{i} \log(1-p_i) \cdot \|i-c\|^\alpha\right)$ |
| CDW-ADA | $-\sum_{i} \log(1-p_i) \cdot Cost[c,i]$ where $Cost[c,i] = \|c-i\|^{1.2} \times w$ (下三角w=0.75，上三角w=0.5) |
| CDW-EXP | $-\sum_{i} \log(1-p_i) \cdot Cost[c,i]$ where $Cost[c,i] = b^{\|c-i\|}$ (下三角b=1.3，上三角b=1.2) |
| CORAL | BCE over $K-1$ binary tasks (cumulative probabilities) |

where $p_i = \text{softmax}(\text{logits})_i$, $c$ is true class, $\hat{y} = \arg\max_i p_i$ is predicted class.

**Cost Matrices**:

**CDW-ADA** (lower=0.75 for 病情低估, upper=0.5 for 病情高估, alpha=1.2):
```
     | F0   | F1   | F2   | F3   | F4   |
-----|------|------|------|------|------|
 F0  | 0.00 | 0.50 | 1.15 | 1.90 | 2.76 | (高估: d^1.2 × 0.5)
 F1  | 0.75 | 0.00 | 0.50 | 1.15 | 1.90 |
 F2  | 1.72 | 0.75 | 0.00 | 0.50 | 1.15 |
 F3  | 2.81 | 1.72 | 0.75 | 0.00 | 0.50 |
 F4  | 4.00 | 2.81 | 1.72 | 0.75 | 0.00 | (低估: d^1.2 × 0.75)
```

**CDW-EXP** (lower_base=1.3 for 病情低估, upper_base=1.2 for 病情高估):
```
     | F0   | F1   | F2   | F3   | F4   |
-----|------|------|------|------|------|
 F0  | 0.00 | 1.20 | 1.44 | 1.73 | 2.07 | (高估: 1.2^d)
 F1  | 1.30 | 0.00 | 1.20 | 1.44 | 1.73 |
 F2  | 1.69 | 1.30 | 0.00 | 1.20 | 1.44 |
 F3  | 2.20 | 1.69 | 1.30 | 0.00 | 1.20 |
 F4  | 2.86 | 2.20 | 1.69 | 1.30 | 0.00 | (低估: 1.3^d)
```

**Adding a new loss function**:
1. Implement class in `core/loss.py` with `forward(logits, targets)` method
2. Add entry to `get_loss()` factory
3. For ordinal losses, implement `predict(logits)` method
4. Add to `loss_configs` in all three training scripts (main.py, main_kfold.py, main_repeated_kfold.py)
5. Add parameter settings in the loss_kwargs section (if needed)
6. Training scripts automatically log model parameters via `log_model_params()`

## Data Pipeline

**Critical preprocessing order** (`core/dataset.py`):
1. **HA cleaning**: Regex fix for malformed Excel entries like `"19..56"` → proper numbers
2. Target encoding: LabelEncoder maps F0-F4 → 0-4
3. Categorical one-hot: `性别` (男/女), `Machine` (MIindary/PHILIPS) → 4 binary features (`drop_first=False`)
4. Standardization: StandardScaler fit on train only (avoid data leakage)
5. Stratified split: 70% train / 10% val / 20% test
6. **Final input shape**: `[B, 24]` where B = batch size

**Class weights**: Calculated as `1 / class_counts`, normalized to sum to num_classes

## Training

**Early stopping**: Monitors validation **MAE** (not loss!), patience=30 epochs
**LR scheduler**: ReduceLROnPlateau on validation MAE, factor=0.5, patience=5
**BatchNorm stability**: Skips batches < 2 samples (`training/train.py:46`)

## Evaluation Metrics

**Ordinal-specific metrics** (`core/evaluate.py`):
- **Adjacent accuracy**: Predictions within ±1 grade (critical for clinical tasks)
- **QWK**: Quadratic Weighted Kappa (measures ordinal agreement)
- **MAE**: Mean Absolute Error

Standard metrics: accuracy, macro_f1, weighted_f1

## Experiment Frameworks

### K-Fold CV (`scripts/main_kfold.py`)
- Stratified 5-fold cross-validation
- Each fold: 90% train / 10% val from fold train data
- Outputs mean ± std across folds

### Repeated K-Fold CV (`scripts/main_repeated_kfold.py`) - Recommended
- 3 repeats × 5 folds = 15 evaluations per loss
- Seeds: [42, 123, 456]
- Multi-level statistics: fold-to-fold variability, split sensitivity, 95% CI
- Auto-generates Markdown reports

## Configuration

**Config class** (`config/__init__.py`):
- `NUMERIC_COLS = 20` (年龄, 身高, 体重, BMI, PLT, ALT, AST, GGT, ALB, ALP, HA, PIIIP, CIV, LN, AST/PLT, AST/ALT, P-SWE, 2D-SWE, ARPI（AST/40）, FIB-4)
- `CATEGORICAL_COLS = ["性别", "Machine"]` → one-hot encoded with `drop_first=False`
- **Final input dimension: 24** (20 numeric + 4 binary from categorical)
- `HIDDEN_DIMS = [64, 32, 16]`
- `DROPOUT = 0.3`
- `LEARNING_RATE = 1e-3`
- `BATCH_SIZE = 32`
- `EPOCHS = 200`
- `PATIENCE = 30`

**Device**: Auto CUDA detection

## Key Implementation Details

- **HA data cleaning**: Uses regex to fix malformed numeric entries in Excel (e.g., `"19..56"`)
- **CORAL inference**: Uses `criterion.predict()` if available, not standard `argmax`
- **Model selection**: Automatic based on loss type
  - `coral` → CORALNet (original architecture with shared weights)
  - `mlp_coral` → MLPClassifier(ordinal_head=True) (ablation study)
  - Other losses → MLPClassifier(ordinal_head=False)
- **MLP ordinal_head**: When `ordinal_head=True`, final layer bias initialized to `torch.linspace(1.5, -1.5, K-1)`
- **Output paths**: Config uses Windows paths (Z:/), but training outputs to Linux server (/home/ubuntu/lq/MLP_results)

## Adding New Loss Function Comparison

To compare a new loss function:

1. Implement in `core/loss.py`:
```python
class NewLoss(nn.Module):
    def forward(self, logits, targets):
        # logits: [B, 5] for standard classification
        pass
```

2. Add to factory in `get_loss()`:
```python
elif name == 'new_loss':
    return NewLoss(**kwargs)
```

3. Add to all three training scripts:
   - `scripts/main.py` (line ~125): Add to `loss_configs` dict
   - `scripts/main_kfold.py` (line ~278): Add to `loss_configs` dict + `loss_kwargs`
   - `scripts/main_repeated_kfold.py` (line ~845): Add to `loss_configs` dict + `loss_kwargs`

4. Use MLPClassifier for all new losses:
```python
model = MLPClassifier(input_dim, config.HIDDEN_DIMS, num_classes,
                      config.DROPOUT, ordinal_head=False)
```

5. Parameter logging is automatic via `log_model_params()` in all scripts

## Data Structure

**Source**: `data/深三数据useful_填充.xlsx`
- 391 samples, 25 columns
- 20 numeric features (年龄, 身高, 体重, BMI, PLT, ALT, AST, GGT, ALB, ALP, HA, PIIIP, CIV, LN, AST/PLT, AST/ALT, P-SWE, 2D-SWE, ARPI（AST/40）, FIB-4)
- 2 categorical (性别, Machine) → one-hot encoded to 4 binary features
- **Final input dimension: 24 features** (20 numeric + 4 binary)
- Target: LABLE_F (F0-F4)
- Class distribution: F0:57, F1:89, F2:59, F3:33, F4:32 (imbalanced)

---

## Current Training Configuration (2026-05-29)

**Active loss functions** (5 total):
1. `ce` - Standard cross-entropy (baseline)
2. `cdw_ce` - Distance weighted CE (α=1.0)
3. `mse` - Mean squared error
4. `cdw_ada` - Adaptive clinical cost-aware CE (下三角0.75，上三角0.5, alpha=1.2)
5. `cdw_exp` - **Exponential distance CE** (下三角base=1.3，上三角base=1.2) - **当前最佳**

**Training parameters**:
- Architecture: MLPClassifier (ordinal_head=False)
- Hidden dims: [64, 32, 16]
- Dropout: 0.3
- Learning rate: 1e-3
- Batch size: 32
- Epochs: 200 (with early stopping on val MAE, patience=30)
- Optimizer: AdamW (weight_decay default)
- Class weights: **sqrt-smoothed** (`1.0 / sqrt(class_counts)`, normalized to [1, max])

**Validation**:
- Single run: `python scripts/main.py`
- 5-fold CV: `python scripts/main_kfold.py`
- Repeated 5-fold CV: `python scripts/main_repeated_kfold.py` (recommended)

**Performance summary** (from repeated 5-fold CV):
- `cdw_exp`: Accuracy 55.19%, QWK 0.715, MAE 0.596 (最佳整体性能)
- `cdw_ada`: Accuracy ~53%, QWK ~0.68, MAE ~0.65 (稳定性能)
- `ce`: Accuracy ~51%, QWK ~0.67, MAE ~0.65 (基线)

**Key design choices**:

1. **cdw_ada** (alpha-非线性版本):
   - Cost[c,i] = |c-i|^1.2 × w (下三角w=0.75，上三角w=0.5)
   - Alpha参数控制距离敏感度（>1 加剧远距离惩罚）

2. **cdw_exp** (指数版本，当前最优):
   - Cost[c,i] = b^|c-i| (下三角b=1.3，上三角b=1.2)
   - 指数增长代价，对远距离误判惩罚更强

3. **Class weights** (sqrt-smoothed):
   - 避免原始权重（1/n_c）过于激进
   - 使用 sqrt 平滑：1/sqrt(n_c)，归一化到 [1, max]

---

## ⚠️ Warnings and Failed Experiments

### DO NOT Combine Class Weights with CDW-CE Losses

**Failed experiment**: `cdw_exp_w` (exponential + class weights)
- **Result**: Accuracy collapsed from 51% → 16%, QWK from 0.67 → 0.04
- **Root cause**: Dual weighting (distance weights × category weights) created severe imbalance
- **Lesson**: CDW-CE type losses already encode class priorities through distance cost matrices. Adding class weights on top creates over-weighting that destroys performance.

### DO NOT Apply Local Boundary Enhancement

**Failed experiment**: `cdw_be` (boundary enhancement on F2→F1)
- **Result**: Accuracy collapsed from 51% → 25%, F2 recall = 0%
- **Root cause**: 2.0× local enhancement on F2→F1 boundary destroyed global cost matrix balance
- **Lesson**: CDW-CE cost matrices are globally designed. Local enhancement on specific edges creates imbalance that causes the model to overcompensate elsewhere.

### Parameter Sensitivity

CDW-CE type losses are highly sensitive to parameter changes:
- Small adjustments (0.75→0.80, 0.5→0.48) may have minimal impact
- Original parameters (cdw_ada: lower=0.75, upper=0.5, alpha=1.2; cdw_exp: lower=1.3, upper=1.2) are well-tuned
- If adjusting parameters, use very small increments and validate thoroughly

### F2→F1 Misclassification Problem

Current models still show significant F2→F1 misclassification (~50% of F2 samples).
**Better approaches** (not yet tried):
- Data level: Augment F2 samples, collect more F2 data
- Feature engineering: Extract features that better distinguish F1/F2
- Model architecture: Attention mechanisms, ensemble methods
- Post-processing: Rule-based prediction calibration

---
