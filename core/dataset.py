import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from config import Config

class FibrosisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(config):
    df = pd.read_excel(config.DATA_PATH)

    # Convert HA to numeric if needed (handle malformed entries like "19..56")
    df["HA"] = df["HA"].astype(str).str.replace(r"\.\.", ".", regex=True)
    df["HA"] = pd.to_numeric(df["HA"], errors="coerce")
    if df["HA"].isnull().sum() > 0:
        df["HA"] = df["HA"].fillna(df["HA"].median())

    # Encode target
    le = LabelEncoder()
    df[config.TARGET_COL] = le.fit_transform(df[config.TARGET_COL])
    label_map = {i: cls for i, cls in enumerate(le.classes_)}
    print(f"Label mapping: {label_map}")

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=config.CATEGORICAL_COLS, drop_first=False)

    # Determine feature columns
    feature_cols = [c for c in df.columns
                    if c not in config.DROP_COLS + [config.TARGET_COL]]

    X = df[feature_cols].values.astype(np.float32)
    y = df[config.TARGET_COL].values.astype(np.int64)

    # Train / val / test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config.TEST_SIZE + config.VAL_SIZE,
        random_state=config.RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=config.TEST_SIZE / (config.TEST_SIZE + config.VAL_SIZE),
        random_state=config.RANDOM_SEED, stratify=y_temp
    )

    # Standardize using training set statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Compute class weights for imbalanced data (sqrt 平滑版本)
    class_counts = np.bincount(y_train)
    # 使用 sqrt 平滑：weight = 1/sqrt(count)，避免权重过大
    class_weights = torch.FloatTensor(1.0 / np.sqrt(class_counts + 1e-6))
    # 归一化到 [1, max]，保证最小权重为1
    class_weights = class_weights / class_weights.min()
    class_weights = class_weights.to(config.DEVICE)

    train_loader = DataLoader(FibrosisDataset(X_train, y_train),
                              batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FibrosisDataset(X_val, y_val),
                            batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(FibrosisDataset(X_test, y_test),
                             batch_size=config.BATCH_SIZE, shuffle=False)

    return (train_loader, val_loader, test_loader,
            X_train.shape[1], len(le.classes_), class_weights, le)
