#!/usr/bin/env python3
"""
NGAFID 训练脚本
- 加载 ngafid_X.npy / ngafid_y.npy（或调用 ngafid_loader 生成）
- MiniRocket 提取特征
- RidgeClassifierCV 分类器
- 5 折 StratifiedKFold 交叉验证
- 输出指标：准确率、F1、AUC（每折 + 均值 ± 标准差）
- 保存结果到 results_<timestamp>.txt
"""
from datetime import datetime
import time
import sys
import os
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---------- 1. 加载数据 ----------

def load_data():
    x_path = "ngafid_X.npy"
    y_path = "ngafid_y.npy"

    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"Loading pre-saved data from {x_path} and {y_path} ...")
        X = np.load(x_path)
        y = np.load(y_path)
    else:
        print("No .npy files found, running ngafid_loader ...")
        from ngafid_loader import NGAFIDDataLoader
        loader = NGAFIDDataLoader()
        X, y = loader.load_all_flights()
        if len(X) == 0:
            raise RuntimeError("No data loaded. Check data directory.")
        loader.save_data(X, y)

    print(f"X shape: {X.shape}  (n_samples, timesteps, channels)")
    print(f"y shape: {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    label_dist = dict(zip(unique.astype(int), counts))
    print(f"Label distribution: {label_dist}")
    return X, y, label_dist

# ---------- 2. 主流程 ----------

def main():
    start = time.time()

    # 2.1 加载数据
    X, y, label_dist = load_data()
    n_samples, n_timesteps, n_channels = X.shape

    # 2.2 MiniRocket 特征提取
    # MiniRocketMultivariate 要求输入 (n_instances, n_channels, n_timepoints)
    # 当前 X 是 (n_samples, n_timesteps, n_channels)，需要 transpose
    print(f"\nTransposing data for MiniRocket: {X.shape} -> ", end="")
    X_t = X.transpose(0, 2, 1).astype(np.float64)
    print(X_t.shape)

    print("Extracting features with MiniRocketMultivariate (num_kernels=10000) ...")
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    t_rocket = time.time()
    minirocket = MiniRocketMultivariate(num_kernels=10000, n_jobs=-1)
    X_feat = minirocket.fit_transform(X_t)
    X_feat = X_feat.to_numpy()
    print(f"MiniRocket feature extraction done in {time.time() - t_rocket:.1f}s")
    print(f"Feature matrix shape: {X_feat.shape}")

    # 2.3 5 折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)

    acc_list, f1_list, auc_list = [], [], []
    fold_results = []

    print("\n" + "=" * 55)
    print("  5-Fold Stratified Cross-Validation")
    print("=" * 55)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_feat, y), start=1):
        X_train, X_test = X_feat[train_idx], X_feat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 训练
        clf.fit(X_train, y_train)

        # 预测
        y_pred = clf.predict(X_test)
        # RidgeClassifierCV 用 decision_function 得到连续值（用于 AUC）
        y_score = clf.decision_function(X_test)

        # 指标
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_val = roc_auc_score(y_test, y_score)

        acc_list.append(acc)
        f1_list.append(f1)
        auc_list.append(auc_val)

        line = (f"Fold {fold_idx}  |  Acc={acc:.4f}  "
                f"F1={f1:.4f}  AUC={auc_val:.4f}  "
                f"(alpha={clf.alpha_:.4f})")
        print(line)
        fold_results.append(line)

    # 2.4 汇总输出
    elapsed = time.time() - start
    acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
    f1_mean,  f1_std  = np.mean(f1_list),  np.std(f1_list)
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list)

    summary = f"""
{'=' * 55}
Summary (mean ± std over 5 folds)
{'=' * 55}
Accuracy : {acc_mean:.4f} ± {acc_std:.4f}
F1 Score : {f1_mean:.4f} ± {f1_std:.4f}
AUC      : {auc_mean:.4f} ± {auc_std:.4f}
{'=' * 55}
Training time: {elapsed:.1f}s  ({elapsed/60:.1f}min)
"""

    print(summary)

    # 2.5 保存到 results_<timestamp>.txt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.txt"

    header = f"""Training Results
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data: ngafid_X.npy / ngafid_y.npy
Samples: {n_samples}
Timesteps: {n_timesteps}
Channels: {n_channels}
Label distribution: {label_dist}
Features (MiniRocket num_kernels=10000): {X_feat.shape[1]}
Classifier: RidgeClassifierCV (alphas=logspace(-3,3,10), inner_cv=5)
Cross-validation: StratifiedKFold (n_splits=5, shuffle=True, random_state=42)
{"=" * 55}
"""
    output = header + "\n".join(fold_results) + summary

    with open(filename, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()
