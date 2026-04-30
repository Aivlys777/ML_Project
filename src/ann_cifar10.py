"""
ANN实现CIFAR-10图像分类（10分类任务）
使用PyTorch构建多层感知机
"""
import os
# 使用华为云镜像源
os.environ['TORCH_HOME'] = 'https://mirrors.huaweicloud.com/pytorch/'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from datetime import datetime
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)


class CIFAR10ANN(nn.Module):
    """CIFAR-10分类神经网络（输入是展平的图像）"""
    def __init__(self):
        super(CIFAR10ANN, self).__init__()
        # CIFAR-10图像是32x32x3 = 3072个像素
        self.network = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10个类别
        )
    
    def forward(self, x):
        return self.network(x)


def load_cifar10_data(batch_size=64):
    """加载CIFAR-10数据集"""
    print("正在下载/加载CIFAR-10数据集...")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 下载训练集和测试集
    trainset = torchvision.datasets.CIFAR10(
        root='data/cifar10', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='data/cifar10', train=False, download=True, transform=transform
    )
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # 获取整个数据集用于评估
    X_train = []
    y_train = []
    for images, labels in trainloader:
        X_train.append(images.view(images.size(0), -1))  # 展平
        y_train.append(labels)
    X_train = torch.cat(X_train).numpy()
    y_train = torch.cat(y_train).numpy()
    
    X_test = []
    y_test = []
    for images, labels in testloader:
        X_test.append(images.view(images.size(0), -1))
        y_test.append(labels)
    X_test = torch.cat(X_test).numpy()
    y_test = torch.cat(y_test).numpy()
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


def train_cifar10_ann(epochs=100, batch_size=64, lr=0.001):
    """训练CIFAR-10 ANN分类器"""
    print("\n" + "="*50)
    print("开始训练ANN CIFAR-10图像分类模型")
    print("="*50)
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_cifar10_data()
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = CIFAR10ANN()
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录
    train_losses = []
    train_accs = []
    test_accs = []
    
    # 训练
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = epoch_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(avg_loss)
        train_accs.append(train_accuracy)
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, test_predicted = torch.max(test_outputs, 1)
            test_accuracy = accuracy_score(y_test, test_predicted.numpy())
            test_accs.append(test_accuracy)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        _, train_pred = torch.max(train_outputs, 1)
        test_outputs = model(X_test_tensor)
        _, test_pred = torch.max(test_outputs, 1)
        
        final_train_acc = accuracy_score(y_train, train_pred.numpy())
        final_test_acc = accuracy_score(y_test, test_pred.numpy())
    
    print("\n" + "="*50)
    print(f"最终训练集准确率: {final_train_acc:.4f}")
    print(f"最终测试集准确率: {final_test_acc:.4f}")
    
    # 绘制曲线
    plot_cifar10_curves(train_losses, train_accs, test_accs)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, test_pred.numpy(), "cifar10")
    
    # 保存模型
    save_cifar10_model(model)
    
    return model


def plot_cifar10_curves(train_losses, train_accs, test_accs):
    """绘制训练曲线"""
    os.makedirs('results', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(train_losses, linewidth=2, color='blue')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss - CIFAR-10', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(test_accs, label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Curves - CIFAR-10', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    date = datetime.now().strftime('%Y%m%d')
    filename = f'results/ann_cifar10_curves_{date}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"训练曲线已保存: {filename}")


def plot_confusion_matrix(y_true, y_pred, dataset_name):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix - {dataset_name}', fontsize=14)
    
    date = datetime.now().strftime('%Y%m%d')
    filename = f'results/ann_{dataset_name}_confusion_{date}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"混淆矩阵已保存: {filename}")


def save_cifar10_model(model):
    """保存CIFAR-10模型"""
    os.makedirs('models', exist_ok=True)
    date = datetime.now().strftime('%Y%m%d')
    model_path = f'models/ann_cifar10_{date}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存: {model_path}")


if __name__ == '__main__':
    train_cifar10_ann()
