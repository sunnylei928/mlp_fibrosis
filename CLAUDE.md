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
- `ce`: Standard cross-entropy
- `cdw_ce`: Class Distance Weighted CE (Polat et al., 2025)
- `cdw_ce_margin`: CDW-CE with margin
- `cdw_ce_prob`: CDW-CE with prediction confidence weighting - multiplies entire loss by max predicted probability $p_{\hat{y}}$
- `coral`: CORAL ordinal loss (Saito et al., 2021) - used with CORALNet
- `mlp_coral`: Same CORAL loss - used with MLP (for ablation study)
- `mse`: MSE on softmax probabilities
- `focal`: Focal loss (gamma=2.0)
- `label_smoothing`: Label smoothing CE (smoothing=0.1)

**Loss Function Formulas**:

| Loss | Formula |
|------|---------|
| CE | $-\log(p_c)$ where $p_c$ is true class probability |
| CDW-CE | $-\sum_{i} \log(1-p_i) \cdot \|i-c\|^\alpha$ |
| CDW-CE-Margin | Same, with $p_i' = \min(p_i + m, 1-\epsilon)$ |
| CDW-CE-Prob | $p_{\hat{y}} \cdot \left(-\sum_{i} \log(1-p_i) \cdot \|i-c\|^\alpha\right)$ |
| CORAL | BCE over $K-1$ binary tasks (cumulative probabilities) |

where $p_i = \text{softmax}(\text{logits})_i$, $c$ is true class, $\hat{y} = \arg\max_i p_i$ is predicted class.

**Adding a new loss function**:
1. Implement class in `core/loss.py` with `forward(logits, targets)` method
2. Add entry to `get_loss()` factory
3. For ordinal losses, implement `predict(logits)` method

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
        # logits shape depends on architecture
        pass
```

2. Add to factory in `get_loss()`:
```python
elif name == 'new_loss':
    return NewLoss(**kwargs)
```

3. Add to experiment lists in `scripts/main.py`, `scripts/main_kfold.py`, `scripts/main_repeated_kfold.py`

4. **Choose architecture in training scripts**:
```python
if loss_type == 'new_loss':
    # Option A: Use MLP
    model = MLPClassifier(..., ordinal_head=False/True)
    # Option B: Create custom architecture (like CORALNet)
```

## Data Structure

**Source**: `data/深三数据useful_填充.xlsx`
- 391 samples, 25 columns
- 20 numeric features (年龄, 身高, 体重, BMI, PLT, ALT, AST, GGT, ALB, ALP, HA, PIIIP, CIV, LN, AST/PLT, AST/ALT, P-SWE, 2D-SWE, ARPI（AST/40）, FIB-4)
- 2 categorical (性别, Machine) → one-hot encoded to 4 binary features
- **Final input dimension: 24 features** (20 numeric + 4 binary)
- Target: LABLE_F (F0-F4)
- Class distribution: F0:57, F1:89, F2:59, F3:33, F4:32 (imbalanced)
