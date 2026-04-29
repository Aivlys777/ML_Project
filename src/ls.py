import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(path):
    df = pd.read_csv(path)
    X = df[["x1", "x2", "x3", "x4"]].values
    y = df["y"].values
    return X, y

def add_bias(X):
    """增加偏置项"""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])

def normalize_zscore(X):
    """Z-score归一化：均值为0，标准差为1"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # 防止除以0
    std[std == 0] = 1
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

def least_squares(X, y):
    """
    最小二乘法求解
    beta = (X^T X)^{-1} X^T y
    """
    # 使用伪逆矩阵求解，更稳定
    # beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    # 或者直接使用 np.linalg.lstsq
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def plot_prediction(y_true, y_pred):
    """绘制真值房价和预测房价的可视化图像"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # 绘制y=x对角线作为参考
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    
    plt.xlabel("True Price", fontsize=12)
    plt.ylabel("Predicted Price", fontsize=12)
    plt.title("Least Squares: True vs Predicted House Prices", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    X, y = load_data("./house_data.csv")
    
    """
    对X的特征进行归一化 - 使用Z-score方法
    """
    X_normalized, mean, std = normalize_zscore(X)
    
    X_bias = add_bias(X_normalized)
    
    beta = least_squares(X_bias, y)
    
    print("=" * 50)
    print("最小二乘法结果")
    print("=" * 50)
    print(f"参数 (beta): {beta}")
    print(f"偏置项 (bias): {beta[0]}")
    print(f"特征权重: {beta[1:]}")
    
    # 可视化
    y_pred = X_bias @ beta
    plot_prediction(y, y_pred)
    
    # 计算评估指标
    mse = np.mean((y_pred - y) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y))
    
    print("\n评估指标:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

if __name__ == "__main__":
    main()