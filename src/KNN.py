import numpy as np
import pandas as pd
import time
from collections import Counter

# ==================== 1. 数据加载 ====================
def load_data(train_path, test_path):
    """加载训练集和测试集"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

# ==================== 2. 数据清洗 ====================
def clean_data(train_df, test_df):
    """数据清洗：处理缺失值、选择特征、归一化"""
    
    # 打印列名以便调试
    print("训练集列名:", train_df.columns.tolist())
    print("测试集列名:", test_df.columns.tolist())
    
    # 选择特征列（注意列名是小写）
    feature_cols = ['Age', 'Fare', 'Sex', 'Pclass', 'sibsp', 'Parch']
    target_col = '2urvived'
    
    # 处理训练集
    train_data = train_df[feature_cols + [target_col]].copy()
    
    # 处理缺失值：Age用中位数填充
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
    
    # 处理测试集
    test_data = test_df[feature_cols].copy()
    test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
    
    # 获取对应的标签
    test_labels = test_df.loc[test_data.index, target_col].values
    
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 分离特征和标签
    X_train = train_data[feature_cols].values
    y_train = train_data[target_col].values
    X_test = test_data.values
    y_test = test_labels
    
    # 特征归一化（Min-Max标准化）
    X_train, X_test = normalize_features(X_train, X_test)
    
    return X_train, y_train, X_test, y_test

def normalize_features(X_train, X_test):
    """Min-Max归一化，将特征缩放到[0,1]区间"""
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    # 避免除零
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    
    X_train_norm = (X_train - min_vals) / range_vals
    X_test_norm = (X_test - min_vals) / range_vals
    
    print(f"归一化后特征范围: [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]")
    
    return X_train_norm, X_test_norm

# ==================== 3. KNN实现（向量化） ====================
class KNNVectorized:
    """向量化实现的KNN分类器"""
    
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        """训练：存储训练数据"""
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        """预测：使用向量化距离计算"""
        # 使用公式 ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab^T
        test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)  # (n_test, 1)
        train_sq = np.sum(self.X_train ** 2, axis=1)  # (n_train,)
        
        # 计算距离平方矩阵
        dist_sq = test_sq + train_sq - 2 * np.dot(X_test, self.X_train.T)
        # 处理数值误差导致的负数
        dist_sq = np.maximum(dist_sq, 0)
        distances = np.sqrt(dist_sq)
        
        # 找到最近的k个邻居
        k_indices = np.argsort(distances, axis=1)[:, :self.k]
        
        # 投票决定类别
        predictions = []
        for indices in k_indices:
            k_labels = self.y_train[indices]
            # 取众数
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)

class KNNLoop:
    """循环实现的KNN分类器（用于对比速度）"""
    
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            # 计算到所有训练点的距离
            distances = np.sqrt(np.sum((self.X_train - test_point) ** 2, axis=1))
            
            # 找到k个最近邻
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)

# ==================== 4. KD-Tree实现 ====================
class KDNode:
    """KD-Tree节点"""
    def __init__(self, point, label, left=None, right=None, axis=0):
        self.point = point      # 数据点坐标
        self.label = label      # 类别标签
        self.left = left        # 左子树
        self.right = right      # 右子树
        self.axis = axis        # 分割轴

class KDTree:
    """手写KD-Tree构建和搜索"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.root = self.build_kdtree(np.arange(len(X)), depth=0)
    
    def build_kdtree(self, indices, depth=0):
        """递归构建KD-Tree"""
        if len(indices) == 0:
            return None
        
        # 选择分割轴
        axis = depth % self.X.shape[1]
        
        # 按当前轴排序
        sorted_indices = indices[np.argsort(self.X[indices, axis])]
        median_idx = len(sorted_indices) // 2
        
        # 创建节点
        node = KDNode(
            point=self.X[sorted_indices[median_idx]],
            label=self.y[sorted_indices[median_idx]],
            axis=axis
        )
        
        # 递归构建左右子树
        node.left = self.build_kdtree(sorted_indices[:median_idx], depth + 1)
        node.right = self.build_kdtree(sorted_indices[median_idx + 1:], depth + 1)
        
        return node
    
    def find_k_nearest(self, target, k=5):
        """寻找k个最近邻"""
        self.best_points = []  # 存储(-dist, point, label)用于最大堆
        self.k = k
        
        def distance_sq(p1, p2):
            return np.sum((p1 - p2) ** 2)
        
        def search(node, depth=0):
            if node is None:
                return 
            
            axis = node.axis
            target_val = target[axis]
            node_val = node.point[axis]
            
            # 决定搜索顺序：先搜索更可能包含最近邻的分支
            if target_val < node_val:
                next_branch = node.left
                opposite_branch = node.right
            else:
                next_branch = node.right
                opposite_branch = node.left
            
            # 递归搜索优先分支
            search(next_branch, depth + 1)
            
            # 检查当前节点
            dist = distance_sq(target, node.point)
            if len(self.best_points) < self.k:
                # 堆中存储(-dist, node, label)实现最大堆效果
                self.best_points.append((-dist, node.label))
                self.best_points.sort()  # 按-dist排序，最小的在最后
            else:
                # 如果当前距离小于堆中最远的距离
                if dist < -self.best_points[-1][0]:
                    self.best_points.pop()
                    self.best_points.append((-dist, node.label))
                    self.best_points.sort()
            
            # 检查是否需要搜索另一分支
            if len(self.best_points) < self.k or (target_val - node_val) ** 2 < -self.best_points[-1][0]:
                search(opposite_branch, depth + 1)
        
        search(self.root)
        # 返回最近k个点的标签
        return [label for _, label in self.best_points]

# ==================== 5. 主程序 ====================
def main():
    # 加载数据
    print("=" * 60)
    print("1. 加载数据...")
    train_df, test_df = load_data('titanic_train_knn.csv', 'titanic_test_knn.csv')
    
    # 数据清洗和归一化
    print("\n2. 数据清洗和归一化...")
    X_train, y_train, X_test, y_test = clean_data(train_df, test_df)
    
    # ========== 3. KNN不同k值测试 ==========
    print("\n" + "=" * 60)
    print("3. 测试不同k值的准确率（向量化实现）")
    print("-" * 60)
    
    k_values = range(3, 11)
    accuracies = {}
    
    for k in k_values:
        knn = KNNVectorized(k=k)
        knn.fit(X_train, y_train)
        
        # 预测
        y_pred = knn.predict(X_test)
        
        # 计算准确率
        accuracy = np.mean(y_pred == y_test)
        accuracies[k] = accuracy
        print(f"k = {k}: 准确率 = {accuracy:.4f}")
    
    # 找出最佳k值
    best_k = max(accuracies, key=accuracies.get)
    print(f"\n最佳k值: {best_k}, 准确率: {accuracies[best_k]:.4f}")
    
    # ========== 速度对比 ==========
    print("\n" + "=" * 60)
    print("4. 向量化 vs 循环实现 速度对比")
    print("-" * 60)
    
    # 向量化实现
    knn_vec = KNNVectorized(k=best_k)
    knn_vec.fit(X_train, y_train)
    start = time.time()
    y_pred_vec = knn_vec.predict(X_test)
    vec_time = time.time() - start
    print(f"向量化实现耗时: {vec_time:.4f} 秒")
    
    # 循环实现
    knn_loop = KNNLoop(k=best_k)
    knn_loop.fit(X_train, y_train)
    start = time.time()
    y_pred_loop = knn_loop.predict(X_test)
    loop_time = time.time() - start
    print(f"循环实现耗时: {loop_time:.4f} 秒")
    print(f"向量化加速比: {loop_time/vec_time:.2f}x")
    
    # 验证两种实现结果一致
    if np.array_equal(y_pred_vec, y_pred_loop):
        print("✓ 向量化和循环实现结果一致")
    else:
        print("✗ 两种实现结果不一致")
    
    # ========== KD-Tree实现 ==========
    print("\n" + "=" * 60)
    print("5. KD-Tree实现（仅使用Age和Fare特征）")
    print("-" * 60)
    
    # 只保留Age和Fare两个特征（在归一化后的特征中，Age和Fare是前两列）
    # 确认特征顺序: ['Age', 'Fare', 'Sex', 'Pclass', 'sibsp', 'Parch']
    X_train_af = X_train[:, :2]  # Age和Fare
    X_test_af = X_test[:, :2]
    
    print(f"训练集(Age+Fare): {X_train_af.shape}")
    print(f"测试集(Age+Fare): {X_test_af.shape}")
    
    # 构建KD-Tree
    print("\n构建KD-Tree...")
    start = time.time()
    kdtree = KDTree(X_train_af, y_train)
    build_time = time.time() - start
    print(f"KD-Tree构建耗时: {build_time:.4f} 秒")
    
    # 使用KD-Tree进行预测（k=5）
    k_neighbors = 5
    predictions = []
    
    print(f"\n使用KD-Tree进行预测（k={k_neighbors}）...")
    start = time.time()
    
    for i, test_point in enumerate(X_test_af):
        neighbors = kdtree.find_k_nearest(test_point, k=k_neighbors)
        # 投票
        most_common = Counter(neighbors).most_common(1)[0][0]
        predictions.append(most_common)
        
        # 打印前5个测试点的邻居信息
        if i < 5:
            print(f"\n测试点{i+1}: Age={test_point[0]:.3f}, Fare={test_point[1]:.3f}")
            print(f"  邻居标签: {neighbors}")
            print(f"  投票结果: {most_common}")
            print(f"  真实标签: {y_test[i]}")
    
    search_time = time.time() - start
    print(f"\nKD-Tree搜索总耗时: {search_time:.4f} 秒")
    print(f"平均每点搜索耗时: {search_time/len(X_test_af)*1000:.2f} 毫秒")
    
    # 计算KD-Tree准确率
    kdtree_accuracy = np.mean(np.array(predictions) == y_test)
    print(f"\nKD-Tree预测准确率 (k={k_neighbors}): {kdtree_accuracy:.4f}")
    
    # 对比
    print("\n" + "=" * 60)
    print("6. 总结")
    print("-" * 60)
    print(f"完整特征KNN最佳准确率: {accuracies[best_k]:.4f} (k={best_k})")
    print(f"Age+Fare特征KD-Tree准确率: {kdtree_accuracy:.4f} (k={k_neighbors})")
    print(f"KD-Tree准确率下降: {accuracies[best_k] - kdtree_accuracy:.4f}")
    
    return accuracies, best_k, kdtree_accuracy

if __name__ == "__main__":
    main()