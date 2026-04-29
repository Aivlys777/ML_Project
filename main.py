"""
机器学习项目统一入口
用法示例:
    python main.py --algo logistic --data titanic --process train
    python main.py --algo knn --data titanic --process test
    python main.py --algo linear --data house --method gd
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
from datetime import datetime

# 设置中文显示
import warnings
warnings.filterwarnings('ignore')

# ============ 逻辑回归实现 ============
class LogisticRegression:
    """逻辑回归二分类"""
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

# ============ KNN实现 ============
class KNN:
    """K近邻算法"""
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return max(set(k_nearest_labels), key=k_nearest_labels.count)

# ============ 线性回归实现 ============
class LinearRegression:
    """线性回归 - 支持最小二乘法和梯度下降"""
    def __init__(self, method='gd', learning_rate=0.01, n_iterations=1000, batch_size=32):
        self.method = method
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        if self.method == 'ls':
            # 最小二乘法
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
            self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            self.bias = self.weights[0]
            self.weights = self.weights[1:]
        else:
            # 梯度下降
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0
            
            for i in range(self.n_iterations):
                if self.method == 'sgd':
                    # 随机梯度下降
                    idx = np.random.choice(n_samples, self.batch_size)
                    X_batch = X[idx]
                    y_batch = y[idx]
                else:
                    # 批量梯度下降
                    X_batch = X
                    y_batch = y
                
                y_predicted = np.dot(X_batch, self.weights) + self.bias
                dw = (1/len(X_batch)) * np.dot(X_batch.T, (y_predicted - y_batch))
                db = (1/len(X_batch)) * np.sum(y_predicted - y_batch)
                
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# ============ 数据加载函数 ============
def load_titanic_data():
    """加载泰坦尼克数据集"""
    data_dir = 'data/titanic'
    train_path = os.path.join(data_dir, 'titanic_train_knn.csv')
    test_path = os.path.join(data_dir, 'titanic_test_knn.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print("原始列名:", train_df.columns.tolist())
    
    # 删除所有'zero'列
    zero_cols = [col for col in train_df.columns if 'zero' in col.lower() or col == '0']
    train_df = train_df.drop(columns=zero_cols)
    test_df = test_df.drop(columns=zero_cols)
    
    print(f"删除zero列后，保留的列: {train_df.columns.tolist()}")
    
    # 处理缺失值
    for df in [train_df, test_df]:
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # 特征工程 - 使用正确的列名（注意：sibsp是小写）
    for df in [train_df, test_df]:
        df['FamilySize'] = df['sibsp'] + df['Parch']
        df['IsAlone'] = (df['FamilySize'] == 0).astype(int)
    
    # 选择特征
    features = ['Pclass', 'Age', 'sibsp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
    
    # 处理性别
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
    test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
    features.append('Sex')
    
    # 处理港口
    train_df = pd.get_dummies(train_df, columns=['Embarked'], prefix='Emb')
    test_df = pd.get_dummies(test_df, columns=['Embarked'], prefix='Emb')
    
    # 添加独热编码后的列
    emb_cols = [col for col in train_df.columns if col.startswith('Emb_')]
    features.extend(emb_cols)
    
    # 目标变量列名是 '2urvived'（注意：不是Survived）
    target_col = '2urvived' if '2urvived' in train_df.columns else 'Survived'
    
    X_train = train_df[features].values
    y_train = train_df[target_col].values
    X_test = test_df[features].values
    
    print(f"特征数量: {len(features)}")
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, scaler

def load_house_data():
    """加载房价数据集"""
    data_path = 'data/house/house_data.csv'
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 找不到文件 {data_path}")
        print("请确保 house_data.csv 文件在 data/house/ 目录下")
        return None, None, None, None, None
    
    df = pd.read_csv(data_path)
    
    # 选择特征和目标变量
    target_col = 'price' if 'price' in df.columns else df.columns[-1]
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler

# ============ 训练函数 ============
def train_logistic(data_name):
    """训练逻辑回归模型"""
    print(f"\n开始训练逻辑回归模型 - 数据集: {data_name}")
    
    if data_name == 'titanic':
        result = load_titanic_data()
        X_train, y_train, X_test, scaler = result
        
        model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
        model.fit(X_train, y_train)
        
        # 评估
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        print(f"训练集准确率: {train_acc:.4f}")
        
        # 保存模型
        save_model(model, f'logistic_{data_name}', scaler=scaler)
        return model
    else:
        print(f"逻辑回归暂不支持 {data_name} 数据集")
        return None

def train_knn(data_name):
    """训练KNN模型"""
    print(f"\n开始训练KNN模型 - 数据集: {data_name}")
    
    if data_name == 'titanic':
        result = load_titanic_data()
        X_train, y_train, X_test, scaler = result
        
        model = KNN(k=5)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        print(f"训练集准确率: {train_acc:.4f}")
        
        # 保存模型
        save_model(model, f'knn_{data_name}', scaler=scaler)
        return model
    else:
        print(f"KNN暂不支持 {data_name} 数据集")
        return None

def train_linear(data_name, method='gd'):
    """训练线性回归模型"""
    print(f"\n开始训练线性回归模型 - 数据集: {data_name}, 方法: {method}")
    
    if data_name == 'house':
        result = load_house_data()
        if result[0] is None:
            return None
        X_train, X_test, y_train, y_test, scaler = result
        
        model = LinearRegression(method=method, learning_rate=0.01, n_iterations=500)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        print(f"训练集MSE: {train_mse:.4f}")
        print(f"测试集MSE: {test_mse:.4f}")
        print(f"R²分数: {1 - test_mse/np.var(y_test):.4f}")
        
        # 保存模型
        save_model(model, f'linear_{data_name}', scaler=scaler)
        return model
    else:
        print(f"线性回归暂不支持 {data_name} 数据集")
        return None

# ============ 保存和加载模型 ============
def save_model(model, name, scaler=None):
    """保存模型"""
    os.makedirs('models', exist_ok=True)
    date = datetime.now().strftime('%Y%m%d')
    filename = f'models/{name}_{date}.pkl'
    joblib.dump({'model': model, 'scaler': scaler}, filename)
    print(f"模型已保存: {filename}")
    return filename

def load_model(name):
    """加载模型"""
    models_dir = 'models'
    # 查找最新的模型文件
    if not os.path.exists(models_dir):
        print(f"models目录不存在")
        return None, None
    model_files = [f for f in os.listdir(models_dir) if f.startswith(name)]
    if not model_files:
        print(f"未找到模型: {name}")
        return None, None
    latest_model = sorted(model_files)[-1]
    data = joblib.load(os.path.join(models_dir, latest_model))
    print(f"加载模型: {latest_model}")
    return data['model'], data.get('scaler')

# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser(description='机器学习项目统一入口')
    parser.add_argument('--algo', type=str, required=True,
                        choices=['logistic', 'knn', 'linear'],
                        help='选择算法: logistic, knn, linear')
    parser.add_argument('--data', type=str, required=True,
                        choices=['titanic', 'house'],
                        help='选择数据集: titanic, house')
    parser.add_argument('--process', type=str, default='train',
                        choices=['train', 'test'],
                        help='执行过程: train(训练), test(测试)')
    parser.add_argument('--method', type=str, default='gd',
                        choices=['ls', 'gd', 'sgd'],
                        help='线性回归求解方法: ls(最小二乘), gd(梯度下降), sgd(随机梯度下降)')
    
    args = parser.parse_args()
    
    print("="*50)
    print("机器学习项目启动")
    print(f"算法: {args.algo}")
    print(f"数据集: {args.data}")
    print(f"过程: {args.process}")
    print("="*50)
    
    if args.process == 'train':
        # 训练模式
        if args.algo == 'logistic':
            train_logistic(args.data)
        elif args.algo == 'knn':
            train_knn(args.data)
        elif args.algo == 'linear':
            train_linear(args.data, args.method)
    else:
        # 测试模式
        print("测试模式 - 需要先训练保存模型")
        model_name = f"{args.algo}_{args.data}"
        model, scaler = load_model(model_name)
        if model:
            print("模型加载成功，可以开始预测")
        else:
            print("请先运行训练: python main.py --algo logistic --data titanic --process train")

if __name__ == '__main__':
    main()