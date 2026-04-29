import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(loss_dict):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    for name, losses in loss_dict.items():
        plt.plot(losses, label=name, linewidth=2)
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("Loss Curve Comparison", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_prediction(y_true, y_pred, title):
    """绘制预测结果散点图"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    
    plt.xlabel("True Price", fontsize=12)
    plt.ylabel("Predicted Price", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def load_data(path):
    df = pd.read_csv(path)
    X = df[["x1", "x2", "x3", "x4"]].values
    y = df["y"].values
    return X, y

def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])

def normalize_zscore(X):
    """Z-score归一化"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std, mean, std

def compute_loss(X, y, beta):
    """均方误差"""
    pred = X @ beta
    return np.mean((pred - y) ** 2)

def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    return mse, rmse, mae

# =========================
# 1️⃣ 批量梯度下降
# =========================
def batch_gradient_descent(X, y, lr=1e-2, epochs=500):
    """批量梯度下降"""
    beta = np.zeros(X.shape[1])
    loss_history = []
    n = X.shape[0]
    
    for epoch in range(epochs):
        # 计算所有样本的梯度
        pred = X @ beta
        grad = (2 / n) * X.T @ (pred - y)
        
        # 更新参数
        beta = beta - lr * grad
        
        loss = compute_loss(X, y, beta)
        loss_history.append(loss)
        
        # 打印进度
        if (epoch + 1) % 100 == 0:
            print(f"BGD - Epoch {epoch+1}/{epochs}, Loss: {loss:.2f}")
    
    return beta, loss_history

# =========================
# 2️⃣ 随机梯度下降
# =========================
def stochastic_gradient_descent(X, y, lr=1e-2, epochs=100):
    """随机梯度下降"""
    beta = np.zeros(X.shape[1])
    n = X.shape[0]
    loss_history = []
    
    for epoch in range(epochs):
        # 随机打乱数据顺序
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss_sum = 0
        for i in range(n):
            xi = X_shuffled[i:i+1]  # 保持二维形状
            yi = y_shuffled[i]
            
            # 计算梯度（单样本）
            pred = xi @ beta
            grad = 2 * xi.T @ (pred - yi)
            
            # 更新参数
            beta = beta - lr * grad.flatten()
        
        # 每轮记录一次loss
        loss = compute_loss(X, y, beta)
        loss_history.append(loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"SGD - Epoch {epoch+1}/{epochs}, Loss: {loss:.2f}")
    
    return beta, loss_history

# =========================
# 3️⃣ 小批量梯度下降
# =========================
def mini_batch_gradient_descent(X, y, lr=1e-2, epochs=200, batch_size=16):
    """小批量梯度下降"""
    beta = np.zeros(X.shape[1])
    n = X.shape[0]
    loss_history = []
    
    for epoch in range(epochs):
        # 随机打乱数据
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        # 分批次处理
        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_size_actual = len(X_batch)
            
            # 计算梯度（批次）
            pred = X_batch @ beta
            grad = (2 / batch_size_actual) * X_batch.T @ (pred - y_batch)
            
            # 更新参数
            beta = beta - lr * grad
        
        loss = compute_loss(X, y, beta)
        loss_history.append(loss)
        
        if (epoch + 1) % 40 == 0:
            print(f"Mini-batch - Epoch {epoch+1}/{epochs}, Loss: {loss:.2f}")
    
    return beta, loss_history

def main():
    X, y = load_data("./house_data.csv")
    
    """
    对X的特征进行归一化
    """
    X_normalized, mean, std = normalize_zscore(X)
    X = add_bias(X_normalized)
    
    loss_dict = {}
    models_results = {}
    
    print("=" * 60)
    print("开始训练梯度下降模型")
    print("=" * 60)
    
    print("\n=== 批量梯度下降 ===")
    beta_bgd, loss_bgd = batch_gradient_descent(X, y)
    loss_dict["BGD"] = loss_bgd
    models_results["BGD"] = beta_bgd
    
    print("\n=== 随机梯度下降 ===")
    beta_sgd, loss_sgd = stochastic_gradient_descent(X, y)
    loss_dict["SGD"] = loss_sgd
    models_results["SGD"] = beta_sgd
    
    print("\n=== 小批量梯度下降 ===")
    beta_mgd, loss_mgd = mini_batch_gradient_descent(X, y)
    loss_dict["Mini-batch"] = loss_mgd
    models_results["Mini-batch"] = beta_mgd
    
    # 绘制损失曲线（所有方法一起）
    plot_loss(loss_dict)
    
    # 绘制预测结果可视化
    print("\n" + "=" * 60)
    print("预测结果可视化")
    print("=" * 60)
    
    plot_prediction(y, X @ beta_bgd, "BGD Prediction")
    plot_prediction(y, X @ beta_sgd, "SGD Prediction")
    plot_prediction(y, X @ beta_mgd, "Mini-batch SGD Prediction")
    
    # 计算并打印评估指标
    print("\n" + "=" * 60)
    print("评估指标对比")
    print("=" * 60)
    print(f"{'Method':<15} {'MSE':<15} {'RMSE':<15} {'MAE':<15}")
    print("-" * 60)
    
    for name, beta in models_results.items():
        y_pred = X @ beta
        mse, rmse, mae = compute_metrics(y, y_pred)
        
        if name == "BGD":
            mse_bgd, rmse_bgd, mae_bgd = mse, rmse, mae
        elif name == "SGD":
            mse_sgd, rmse_sgd, mae_sgd = mse, rmse, mae
        elif name == "Mini-batch":
            mse_mgd, rmse_mgd, mae_mgd = mse, rmse, mae
        
        print(f"{name:<15} {mse:<15.2f} {rmse:<15.2f} {mae:<15.2f}")
    
    # 误差柱状图对比
    plot_error_comparison(models_results, X, y)

def plot_error_comparison(models_results, X, y):
    """绘制误差对比柱状图"""
    metrics = {'MSE': [], 'RMSE': [], 'MAE': []}
    names = list(models_results.keys())
    
    for name in names:
        y_pred = X @ models_results[name]
        mse, rmse, mae = compute_metrics(y, y_pred)
        metrics['MSE'].append(mse)
        metrics['RMSE'].append(rmse)
        metrics['MAE'].append(mae)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        axes[idx].bar(names, values, color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[idx].set_title(f'{metric_name} Comparison', fontsize=12)
        axes[idx].set_ylabel(metric_name)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值标签
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.01 * max(values), f'{v:.2f}', 
                          ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()