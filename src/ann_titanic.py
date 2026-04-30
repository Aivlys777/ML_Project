"""
ANN实现泰坦尼克生存预测（二分类任务）
训练和测试分离 - 修复版
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import joblib
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)


class TitanicANN(nn.Module):
    """泰坦尼克生存预测神经网络（二分类）"""
    def __init__(self, input_dim):
        super(TitanicANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),  # 添加批归一化，稳定训练
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 确保输出在[0,1]之间
        )
    
    def forward(self, x):
        return self.network(x)


def load_titanic_data():
    """加载泰坦尼克数据"""
    train_path = 'data/titanic/titanic_train_knn.csv'
    test_path = 'data/titanic/titanic_test_knn.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"原始训练集大小: {train_df.shape}")
    print(f"原始测试集大小: {test_df.shape}")
    
    # 删除zero列
    zero_cols = [col for col in train_df.columns if 'zero' in col.lower() or col == '0']
    train_df = train_df.drop(columns=zero_cols)
    test_df = test_df.drop(columns=zero_cols)
    
    print(f"删除zero列后保留: {train_df.columns.tolist()}")
    
    # 处理缺失值
    for df in [train_df, test_df]:
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # 特征工程
    for df in [train_df, test_df]:
        df['FamilySize'] = df['sibsp'] + df['Parch']
        df['IsAlone'] = (df['FamilySize'] == 0).astype(int)
    
    # 特征选择
    features = ['Pclass', 'Age', 'sibsp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
    
    # 性别编码
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
    test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
    features.append('Sex')
    
    # 港口独热编码
    train_df = pd.get_dummies(train_df, columns=['Embarked'], prefix='Emb')
    test_df = pd.get_dummies(test_df, columns=['Embarked'], prefix='Emb')
    emb_cols = [col for col in train_df.columns if col.startswith('Emb_')]
    features.extend(emb_cols)
    
    # 确保测试集也有相同的列
    for col in features:
        if col not in test_df.columns:
            test_df[col] = 0
    
    X_train = train_df[features].values.astype(np.float32)
    y_train = train_df['2urvived'].values.astype(np.float32).reshape(-1, 1)
    X_test = test_df[features].values.astype(np.float32)
    
    print(f"特征数量: {len(features)}")
    print(f"训练集X形状: {X_train.shape}")
    print(f"训练集y形状: {y_train.shape}")
    print(f"训练集y值范围: [{y_train.min()}, {y_train.max()}]")
    
    # 检查y_train中是否有异常值
    if y_train.min() < 0 or y_train.max() > 1:
        print(f"警告：目标值超出[0,1]范围！")
        y_train = np.clip(y_train, 0, 1)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 检查标准化后是否有NaN
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("警告：处理后存在NaN或Inf值")
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
    
    return X_train, y_train, X_test, scaler, features


def train_titanic_ann(epochs=150, batch_size=32, lr=0.001):
    """训练泰坦尼克ANN"""
    print("\n" + "="*50)
    print("开始训练ANN泰坦尼克生存预测模型")
    print("="*50)
    
    # 加载数据
    X_train, y_train, X_test, scaler, features = load_titanic_data()
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    input_dim = X_train.shape[1]
    model = TitanicANN(input_dim)
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # 记录
    train_losses = []
    train_accs = []
    best_loss = float('inf')
    
    # 训练
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # 确保输出在[0,1]范围内
            outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        train_accs.append(accuracy)
        
        # 调整学习率
        scheduler.step(avg_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 30 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 评估
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor)
        train_pred = torch.clamp(train_pred, 0, 1)
        train_pred_class = (train_pred > 0.5).float().numpy()
        train_accuracy = accuracy_score(y_train, train_pred_class)
        
        test_pred = model(X_test_tensor)
        test_pred = torch.clamp(test_pred, 0, 1)
        test_pred_class = (test_pred > 0.5).float().numpy()
    
    print(f"\n最终训练集准确率: {train_accuracy:.4f}")
    print(f"测试集预测完成，共 {len(test_pred_class)} 个样本")
    
    # 绘制曲线
    plot_training_curves(train_losses, train_accs, "titanic")
    
    # 保存模型
    save_titanic_model(model, scaler, features, "titanic")
    
    return model


def plot_training_curves(losses, accuracies, dataset_name):
    """绘制训练曲线"""
    os.makedirs('results', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(losses, linewidth=2, color='blue')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training Loss - {dataset_name}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(accuracies, linewidth=2, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Training Accuracy - {dataset_name}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    date = datetime.now().strftime('%Y%m%d')
    filename = f'results/ann_{dataset_name}_curves_{date}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"训练曲线已保存: {filename}")


def save_titanic_model(model, scaler, features, dataset_name):
    """保存模型"""
    os.makedirs('models', exist_ok=True)
    date = datetime.now().strftime('%Y%m%d')
    
    model_path = f'models/ann_{dataset_name}_{date}.pth'
    torch.save(model.state_dict(), model_path)
    
    scaler_path = f'models/ann_{dataset_name}_scaler_{date}.pkl'
    joblib.dump({'scaler': scaler, 'features': features}, scaler_path)
    
    print(f"模型已保存: {model_path}")
    print(f"预处理器已保存: {scaler_path}")


def load_titanic_model():
    """加载模型（供test.py使用）"""
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.startswith('ann_titanic_') and f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError("未找到泰坦尼克模型，请先运行训练")
    
    latest_model = sorted(model_files)[-1]
    scaler_files = [f for f in os.listdir(models_dir) if f.startswith('ann_titanic_scaler_')]
    latest_scaler = sorted(scaler_files)[-1] if scaler_files else None
    
    return latest_model, latest_scaler


if __name__ == '__main__':
    train_titanic_ann()
