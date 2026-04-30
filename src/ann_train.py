"""
统一训练入口
用法: python train.py --model house/titanic/cifar10
"""

import argparse
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser(description='训练ANN模型')
    parser.add_argument('--model', type=str, required=True,
                        choices=['house', 'titanic', 'cifar10', 'all'],
                        help='选择要训练的模型')
    
    args = parser.parse_args()
    
    if args.model == 'house':
        from ann_house import train_ann_house
        train_ann_house(epochs=200)
    elif args.model == 'titanic':
        from ann_titanic import train_titanic_ann
        train_titanic_ann(epochs=150)
    elif args.model == 'cifar10':
        from ann_cifar10 import train_cifar10_ann
        train_cifar10_ann(epochs=100)
    elif args.model == 'all':
        from ann_house import train_ann_house
        from ann_titanic import train_titanic_ann
        from ann_cifar10 import train_cifar10_ann
        print("\n开始训练所有模型...")
        train_ann_house(epochs=200)
        train_titanic_ann(epochs=150)
        train_cifar10_ann(epochs=100)
        print("\n所有模型训练完成！")

if __name__ == '__main__':
    main()
