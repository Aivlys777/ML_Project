import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =========== 获取脚本所在目录 ============
script_dir = os.path.dirname(os.path.abspath(__file__))

# =========== 1. 数据读取 ============
train_df = pd.read_csv(os.path.join(script_dir, 'titanic_train_knn.csv'))
test_df = pd.read_csv(os.path.join(script_dir, 'titanic_test_knn.csv'))

print("="*50)
print("数据加载完成")
print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# ==================== 2. 数据清洗 ====================
# 2.1 删除所有zero列
zero_cols = [col for col in train_df.columns if 'zero' in col.lower() or col == '0']
train_df = train_df.drop(columns=zero_cols)
test_df = test_df.drop(columns=zero_cols)

print(f"\n删除zero列后，训练集列名: {train_df.columns.tolist()}")

# 2.2 处理缺失值
print("\n缺失值处理前:")
print(f"训练集Age缺失: {train_df['Age'].isnull().sum()}")
print(f"测试集Age缺失: {test_df['Age'].isnull().sum()}")
print(f"训练集Fare缺失: {train_df['Fare'].isnull().sum()}")
print(f"测试集Fare缺失: {test_df['Fare'].isnull().sum()}")
print(f"训练集Embarked缺失: {train_df['Embarked'].isnull().sum()}")
print(f"测试集Embarked缺失: {test_df['Embarked'].isnull().sum()}")

# 用中位数填充Age
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

# 用中位数填充Fare
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# 用众数填充Embarked
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

# 2.3 处理异常值（Fare中的异常大值）
fare_median_train = train_df[train_df['Fare'] <= 1000000]['Fare'].median()
fare_median_test = test_df[test_df['Fare'] <= 1000000]['Fare'].median()
train_df.loc[train_df['Fare'] > 1000000, 'Fare'] = fare_median_train
test_df.loc[test_df['Fare'] > 1000000, 'Fare'] = fare_median_test

print(f"\n异常值处理后，Fare最大值: {train_df['Fare'].max()}")

# ==================== 3. 特征工程 ====================
# 3.1 选择有效特征列
feature_cols = ['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked']

X_train_raw = train_df[feature_cols].copy()
y_train = train_df['2urvived'].copy()

X_test_raw = test_df[feature_cols].copy()
y_test = test_df['2urvived'].copy()

print(f"\n特征列: {feature_cols}")

# 3.2 One-hot编码类别型特征（Sex和Embarked）
X_train = pd.get_dummies(X_train_raw, columns=['Sex', 'Embarked'], drop_first=True)
X_test = pd.get_dummies(X_test_raw, columns=['Sex', 'Embarked'], drop_first=True)

# 确保训练集和测试集列一致
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0
for col in X_test.columns:
    if col not in X_train.columns:
        X_test = X_test.drop(columns=[col])

print(f"\nOne-hot编码后特征数: {len(X_train.columns)}")
print(f"特征列: {X_train.columns.tolist()}")

# 3.3 数值特征归一化
numeric_cols = ['Age', 'Fare', 'sibsp', 'Parch', 'Pclass']
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print(f"\n归一化完成")
print(f"X_train形状: {X_train.shape}")
print(f"X_test形状: {X_test.shape}")

# ==================== 4. sklearn逻辑回归 ====================
print("\n" + "="*50)
print("4.1 sklearn逻辑回归")
print("="*50)

sklearn_lr = LogisticRegression(random_state=42, max_iter=1000)
sklearn_lr.fit(X_train, y_train)

y_pred_sklearn = sklearn_lr.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"sklearn逻辑回归测试准确率: {accuracy_sklearn:.4f} ({accuracy_sklearn*100:.2f}%)")

# ==================== 5. 手写mini-batch梯度下降（带L2正则化） ====================
print("\n" + "="*50)
print("5.1 手写mini-batch梯度下降（L2正则化）")
print("="*50)

class LogisticRegressionMiniBatch:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32, 
                 lambda_reg=0.01, random_state=42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg  # L2正则化系数
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, X, y):
        """计算带L2正则化的损失"""
        n_samples = X.shape[0]
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        
        # 二分类交叉熵损失
        epsilon = 1e-9  # 防止log(0)
        loss = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        
        # L2正则化项
        l2_loss = (self.lambda_reg / (2 * n_samples)) * np.sum(self.weights ** 2)
        
        return loss + l2_loss
    
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Mini-batch梯度下降
        for iteration in range(self.n_iterations):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch迭代
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                batch_size_actual = X_batch.shape[0]
                
                # 前向传播
                linear_pred = np.dot(X_batch, self.weights) + self.bias
                y_pred = self.sigmoid(linear_pred)
                
                # 计算梯度
                dw = (1/batch_size_actual) * np.dot(X_batch.T, (y_pred - y_batch))
                db = (1/batch_size_actual) * np.sum(y_pred - y_batch)
                
                # 添加L2正则化梯度
                dw += (self.lambda_reg / batch_size_actual) * self.weights
                
                # 更新参数
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # 每10个epoch记录一次损失
            if iteration % 10 == 0:
                loss = self.compute_loss(X, y)
                self.loss_history.append(loss)
                if iteration % 100 == 0:
                    print(f"Iteration {iteration}, Loss: {loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

# 转换为numpy数组
X_train_np = X_train.values.astype(np.float64)
y_train_np = y_train.values.astype(np.float64)
X_test_np = X_test.values.astype(np.float64)
y_test_np = y_test.values.astype(np.float64)

# 训练手写模型
print("\n开始训练mini-batch梯度下降模型...")
manual_lr = LogisticRegressionMiniBatch(
    learning_rate=0.1,
    n_iterations=500,
    batch_size=32,
    lambda_reg=0.01,
    random_state=42
)
manual_lr.fit(X_train_np, y_train_np)

# 预测
y_pred_manual = manual_lr.predict(X_test_np)
accuracy_manual = accuracy_score(y_test_np, y_pred_manual)
print(f"\n手写mini-batch梯度下降测试准确率: {accuracy_manual:.4f} ({accuracy_manual*100:.2f}%)")

# ==================== 6. 绘制loss曲线 ====================
plt.figure(figsize=(10, 6))
plt.plot(range(0, manual_lr.n_iterations, 10), manual_lr.loss_history, 'b-', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss (with L2 Regularization)', fontsize=12)
plt.title('Mini-batch Gradient Descent Loss Curve', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('loss_curve.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"\n最终损失值: {manual_lr.loss_history[-1]:.4f}")

# ==================== 7. 结果对比 ====================
print("\n" + "="*50)
print("结果对比")
print("="*50)
print(f"sklearn逻辑回归准确率: {accuracy_sklearn:.4f}")
print(f"手写mini-batch梯度下降准确率: {accuracy_manual:.4f}")
print(f"准确率差异: {abs(accuracy_sklearn - accuracy_manual):.4f}")

# 显示特征重要性（sklearn模型）
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': sklearn_lr.coef_[0]
})
feature_importance = feature_importance.sort_values('coefficient', key=abs, ascending=False)
print("\n特征重要性（按系数绝对值排序）:")
print(feature_importance.to_string(index=False))

# 额外分析：查看模型预测效果
print(f"\n手写模型预测结果统计:")
print(f"预测为0的数量: {(y_pred_manual == 0).sum()}")
print(f"预测为1的数量: {(y_pred_manual == 1).sum()}")
print(f"真实为0的数量: {(y_test_np == 0).sum()}")
print(f"真实为1的数量: {(y_test_np == 1).sum()}")