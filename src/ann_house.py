"""
ANN实现房价预测（回归任务）
使用PyTorch构建神经网络
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
from datetime import datetime

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class HousePriceANN(nn.Module):
    """房价预测神经网络"""
    def __init__(self, input_dim):
        super(HousePriceANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 回归任务输出1个值
        )
    
    def forward(self, x):
        return self.network(x)


def load_house_data():
    """加载房价数据"""
    data_path = 'data/house/house_data.csv'
    df = pd.read_csv(data_path)
    
    # 假设最后一列是房价
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    
    # 标准化特征
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    
    # 标准化目标值
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


def train_ann_house(epochs=200, batch_size=32, lr=0.001):
    """训练房价预测ANN"""
    print("\n" + "="*50)
    print("开始训练ANN房价预测模型")
    print("="*50)
    
    # 加载数据
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_house_data()
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    input_dim = X_train.shape[1]
    model = HousePriceANN(input_dim)
    criterion = nn.MSELoss()  # 回归用均方误差
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录损失
    train_losses = []
    test_losses = []
    
    # 训练
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_tensor)
            test_loss = criterion(test_pred, y_test_tensor)
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss.item():.4f}")
    
    # 评估
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor).numpy()
        test_pred = model(X_test_tensor).numpy()
    
    # 反标准化
    train_pred_original = scaler_y.inverse_transform(train_pred)
    y_train_original = scaler_y.inverse_transform(y_train_tensor.numpy())
    test_pred_original = scaler_y.inverse_transform(test_pred)
    y_test_original = scaler_y.inverse_transform(y_test_tensor.numpy())
    
    # 计算指标
    train_mse = mean_squared_error(y_train_original, train_pred_original)
    test_mse = mean_squared_error(y_test_original, test_pred_original)
    train_r2 = r2_score(y_train_original, train_pred_original)
    test_r2 = r2_score(y_test_original, test_pred_original)
    
    print("\n" + "="*50)
    print("模型评估结果:")
    print(f"训练集 MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    print(f"测试集 MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
    
    # 绘制损失曲线
    plot_loss_curve(train_losses, test_losses, "house")
    
    # 绘制真实值vs预测值
    plot_predictions(y_test_original.flatten(), test_pred_original.flatten(), "house")
    
    # 保存模型
    save_model(model, scaler_X, scaler_y, "house")
    
    return model


def plot_loss_curve(train_losses, test_losses, dataset_name):
    """绘制损失曲线"""
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'ANN Training Loss Curve - {dataset_name}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    date = datetime.now().strftime('%Y%m%d')
    filename = f'results/ann_{dataset_name}_loss_{date}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"损失曲线已保存: {filename}")


def plot_predictions(y_true, y_pred, dataset_name):
    """绘制真实值vs预测值散点图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predictions', fontsize=12)
    plt.title(f'ANN Predictions vs True Values - {dataset_name}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    date = datetime.now().strftime('%Y%m%d')
    filename = f'results/ann_{dataset_name}_scatter_{date}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"散点图已保存: {filename}")


def save_model(model, scaler_X, scaler_y, dataset_name):
    """保存模型和预处理器"""
    os.makedirs('models', exist_ok=True)
    date = datetime.now().strftime('%Y%m%d')
    
    # 保存模型权重
    model_path = f'models/ann_{dataset_name}_{date}.pth'
    torch.save(model.state_dict(), model_path)
    
    # 保存预处理器
    scaler_path = f'models/ann_{dataset_name}_scaler_{date}.pkl'
    joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, scaler_path)
    
    print(f"模型已保存: {model_path}")
    print(f"预处理器已保存: {scaler_path}")


if __name__ == '__main__':
    train_ann_house()
