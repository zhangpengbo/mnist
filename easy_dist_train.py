#!/usr/bin/env python3
"""
最小化分布式训练示例 - 适用于 K8s PyTorchJob
使用 mock 数据验证多机多卡训练流程 - 仅训练，无测试
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import argparse
import time
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_rank_from_env():
    """从环境变量获取分布式训练信息"""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    global_rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"Local rank: {local_rank}, Global rank: {global_rank}, World size: {world_size}")
    return local_rank, global_rank, world_size

def setup_distributed():
    """初始化分布式训练环境"""
    local_rank, global_rank, world_size = get_rank_from_env()
    
    # 设置 CUDA 设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    # 初始化分布式进程组
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=global_rank,
            world_size=world_size
        )
    
    logging.info(f"Distributed training initialized - Rank: {global_rank}/{world_size}, Device: {device}")
    return local_rank, global_rank, world_size, device

class MockDataset(Dataset):
    """Mock 数据集 - 生成随机数据用于训练验证"""
    
    def __init__(self, size=1000, input_dim=784, num_classes=10):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 生成随机数据
        torch.manual_seed(42)  # 确保数据一致性
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, num_classes, (size,))
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SimpleModel(nn.Module):
    """简单的多层感知机模型"""
    
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class DistributedTrainer:
    """分布式训练器"""
    
    def __init__(self, args):
        self.args = args
        
        # 初始化分布式环境
        self.local_rank, self.global_rank, self.world_size, self.device = setup_distributed()
        
        # 设置随机种子
        torch.manual_seed(args.seed + self.global_rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + self.global_rank)
        
        # 创建模型
        self.model = SimpleModel(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes
        ).to(self.device)
        
        # 包装为 DDP 模型
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        # 创建训练数据集和数据加载器
        self.train_dataset = MockDataset(
            size=args.dataset_size,
            input_dim=args.input_dim,
            num_classes=args.num_classes
        )
        
        # 分布式采样器
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=True
        ) if self.world_size > 1 else None
        
        # 数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            num_workers=2,
            pin_memory=True
        )
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
        
        if self.global_rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logging.info(f"Model created - Total params: {total_params}, Trainable: {trainable_params}")
            logging.info(f"Training data: {len(self.train_dataset)} samples")
    
    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        
        if self.train_sampler:
            self.train_sampler.set_epoch(epoch)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 定期输出进度（仅 rank 0）
            if self.global_rank == 0 and batch_idx % self.args.log_interval == 0:
                logging.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {loss.item():.6f}, '
                    f'Acc: {100. * correct / total:.2f}%'
                )
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        # 同步所有进程的统计信息
        if self.world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            acc_tensor = torch.tensor(accuracy, device=self.device)
            
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
            
            avg_loss = loss_tensor.item() / self.world_size
            accuracy = acc_tensor.item() / self.world_size
        
        return avg_loss, accuracy
    
    def train(self):
        """主训练循环"""
        if self.global_rank == 0:
            logging.info("Starting distributed training...")
            logging.info(f"Training for {self.args.epochs} epochs")
        
        best_accuracy = 0.0
        
        for epoch in range(1, self.args.epochs + 1):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # 输出结果（仅 rank 0）
            if self.global_rank == 0:
                logging.info(
                    f'Epoch {epoch}/{self.args.epochs} completed in {epoch_time:.2f}s'
                )
                logging.info(
                    f'Train - Loss: {train_loss:.6f}, Acc: {train_acc:.2f}%'
                )
                
                if train_acc > best_accuracy:
                    best_accuracy = train_acc
                    logging.info(f'New best accuracy: {best_accuracy:.2f}%')
                
                logging.info('-' * 60)
            
            # 同步所有进程
            if self.world_size > 1:
                dist.barrier()
        
        if self.global_rank == 0:
            logging.info(f"Training completed! Best accuracy: {best_accuracy:.2f}%")
    
    def cleanup(self):
        """清理分布式环境"""
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='Minimal Distributed Training Example')
    
    # 模型参数
    parser.add_argument('--input-dim', type=int, default=784, help='Input dimension')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=5, help='LR scheduler step size')
    parser.add_argument('--lr-gamma', type=float, default=0.5, help='LR scheduler gamma')
    
    # 数据参数
    parser.add_argument('--dataset-size', type=int, default=10000, help='Dataset size')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=10, help='Logging interval')
    
    args = parser.parse_args()
    
    try:
        # 创建训练器并开始训练
        trainer = DistributedTrainer(args)
        trainer.train()
        trainer.cleanup()
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)

if __name__ == "__main__":
    main() 
