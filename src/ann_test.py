"""
统一测试入口
用法: python test.py --model house/titanic/cifar10
"""

import argparse
import sys
import os
import torch
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(__file__))

def test_house():
    """测试房价预测模型"""
    print("\n测试房价预测模型...")
    from ann_house import HousePriceANN, load_house_data
    
    # 加载最新的模型
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.startswith('ann_house_') and f.endswith('.pth')]
    scaler_files = [f for f in os.listdir(models_dir) if f.startswith('ann_house_scaler_')]
    
    if not model_files:
        print("未找到模型，请先运行: python train.py --model house")
        return
    
    latest_model = sorted(model_files)[-1]
    latest_scaler = sorted(scaler_files)[-1]
    
    # 加载数据
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_house_data()
    
    # 加载模型
    input_dim = X_train.shape[1]
    model = HousePriceANN(input_dim)
    model.load_state_dict(torch.load(os.path.join(models_dir, latest_model)))
    model.eval()
    
    # 预测
    X_test_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()
        predictions_original = scaler_y.inverse_transform(predictions)
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)
    
    print(f"测试集 MSE: {mse:.4f}")
    print(f"测试集 R²: {r2:.4f}")


def test_titanic():
    """测试泰坦尼克模型"""
    print("\n测试泰坦尼克模型...")
    from ann_titanic import TitanicANN, load_titanic_data
    
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.startswith('ann_titanic_') and f.endswith('.pth')]
    scaler_files = [f for f in os.listdir(models_dir) if f.startswith('ann_titanic_scaler_')]
    
    if not model_files:
        print("未找到模型，请先运行: python train.py --model titanic")
        return
    
    latest_model = sorted(model_files)[-1]
    
    # 加载数据
    X_train, y_train, X_test, scaler, features = load_titanic_data()
    
    # 加载模型
    input_dim = X_train.shape[1]
    model = TitanicANN(input_dim)
    model.load_state_dict(torch.load(os.path.join(models_dir, latest_model)))
    model.eval()
    
    # 预测
    X_test_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions_class = (predictions > 0.5).float().numpy()
    
    # 训练集评估
    X_train_tensor = torch.FloatTensor(X_train)
    with torch.no_grad():
        train_pred = model(X_train_tensor)
        train_pred_class = (train_pred > 0.5).float().numpy()
        train_acc = accuracy_score(y_train, train_pred_class)
    
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集预测完成，共 {len(predictions_class)} 个样本")


def test_cifar10():
    """测试CIFAR-10模型"""
    print("\n测试CIFAR-10模型...")
    from ann_cifar10 import CIFAR10ANN, load_cifar10_data
    
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.startswith('ann_cifar10_') and f.endswith('.pth')]
    
    if not model_files:
        print("未找到模型，请先运行: python train.py --model cifar10")
        return
    
    latest_model = sorted(model_files)[-1]
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_cifar10_data()
    
    # 加载模型
    model = CIFAR10ANN()
    model.load_state_dict(torch.load(os.path.join(models_dir, latest_model)))
    model.eval()
    
    # 预测
    X_test_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predictions = torch.max(outputs, 1)
        test_acc = accuracy_score(y_test, predictions.numpy())
    
    print(f"测试集准确率: {test_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description='测试ANN模型')
    parser.add_argument('--model', type=str, required=True,
                        choices=['house', 'titanic', 'cifar10'],
                        help='选择要测试的模型')
    
    args = parser.parse_args()
    
    if args.model == 'house':
        test_house()
    elif args.model == 'titanic':
        test_titanic()
    elif args.model == 'cifar10':
        test_cifar10()

if __name__ == '__main__':
    main()
