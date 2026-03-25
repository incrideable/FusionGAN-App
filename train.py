#!/usr/bin/env python3
"""
FusionGAN 训练脚本
基于生成对抗网络的红外与可见光图像融合
"""

import os
import time
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 导入自定义模块
from models.generator import FusionGenerator
from models.discriminator import PatchDiscriminator
from datasets.data_loader import FusionDataset, get_data_loaders
from utils.losses import FusionLoss
from utils.metrics import ImageFusionMetrics
from utils.visualization import visualize_fusion_results, visualize_training_progress
from utils.logger import setup_logger

class FusionGANTrainer:
    def __init__(self, config_path):
        """初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self.load_config(config_path)
        self.device = torch.device(f'cuda:{self.config["device"]["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 初始化日志
        self.logger = setup_logger('FusionGAN', self.config['logging']['log_dir'])
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self.setup_models()
        
        # 初始化数据加载器
        self.setup_data_loaders()
        
        # 初始化损失函数
        self.setup_losses()
        
        # 初始化优化器
        self.setup_optimizers()
        
        # 初始化评估指标
        self.metrics = ImageFusionMetrics()
        
        # 初始化TensorBoard
        if self.config['logging']['tensorboard']:
            self.writer = SummaryWriter(log_dir=self.config['logging']['log_dir'])
        
        # 训练历史记录
        self.train_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'total_loss': [],
            'ssim': [],
            'psnr': [],
            'mi': [],
            'std': [],
            'entropy': []
        }
        
        # 最佳模型指标
        self.best_metric = 0.0
        self.best_epoch = 0
        
    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_models(self):
        """初始化模型"""
        # 生成器
        self.generator = FusionGenerator(
            in_channels=self.config['model']['generator']['in_channels'],
            out_channels=self.config['model']['generator']['out_channels'],
            features=self.config['model']['generator']['features']
        ).to(self.device)
        
        # 判别器
        self.discriminator = PatchDiscriminator(
            in_channels=self.config['model']['discriminator']['in_channels'],
            features=self.config['model']['discriminator']['features']
        ).to(self.device)
        
        self.logger.info("模型初始化完成")
        self.logger.info(f"生成器参数量: {sum(p.numel() for p in self.generator.parameters()):,}")
        self.logger.info(f"判别器参数量: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def setup_data_loaders(self):
        """初始化数据加载器"""
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            data_dir=self.config['dataset']['data_dir'],
            dataset_name=self.config['dataset']['name'],
            batch_size=self.config['training']['batch_size'],
            image_size=tuple(self.config['dataset']['image_size']),
            train_split=self.config['dataset']['train_split'],
            val_split=self.config['dataset']['val_split'],
            augmentation=self.config['augmentation'],
            num_workers=self.config['device']['num_workers'],
            pin_memory=self.config['device']['pin_memory']
        )
        
        self.logger.info(f"数据加载器初始化完成")
        self.logger.info(f"训练集: {len(self.train_loader.dataset)} 样本")
        self.logger.info(f"验证集: {len(self.val_loader.dataset)} 样本")
        self.logger.info(f"测试集: {len(self.test_loader.dataset)} 样本")
    
    def setup_losses(self):
        """初始化损失函数"""
        self.criterion = FusionLoss(
            lambda_adv=self.config['training']['loss_weights']['adversarial'],
            lambda_fm=self.config['training']['loss_weights']['feature_matching'],
            lambda_l1=self.config['training']['loss_weights']['l1'],
            lambda_perceptual=self.config['training']['loss_weights']['perceptual']
        ).to(self.device)
        
        self.logger.info("损失函数初始化完成")
    
    def setup_optimizers(self):
        """初始化优化器"""
        # 生成器优化器
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config['training']['learning_rate']['generator'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2'])
        )
        
        # 判别器优化器
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['training']['learning_rate']['discriminator'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2'])
        )
        
        # 学习率调度器
        if self.config['training']['lr_scheduler']['enabled']:
            self.g_scheduler = torch.optim.lr_scheduler.StepLR(
                self.g_optimizer,
                step_size=self.config['training']['lr_scheduler']['step_size'],
                gamma=self.config['training']['lr_scheduler']['gamma']
            )
            self.d_scheduler = torch.optim.lr_scheduler.StepLR(
                self.d_optimizer,
                step_size=self.config['training']['lr_scheduler']['step_size'],
                gamma=self.config['training']['lr_scheduler']['gamma']
            )
        
        self.logger.info("优化器初始化完成")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_total_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["training"]["epochs"]}')
        
        for batch_idx, (ir_imgs, vis_imgs, target_imgs) in enumerate(progress_bar):
            ir_imgs = ir_imgs.to(self.device)
            vis_imgs = vis_imgs.to(self.device)
            target_imgs = target_imgs.to(self.device)
            
            batch_size = ir_imgs.size(0)
            
            # 训练判别器
            d_loss = self.train_discriminator(ir_imgs, vis_imgs, target_imgs)
            
            # 训练生成器
            g_loss, total_loss = self.train_generator(ir_imgs, vis_imgs, target_imgs, epoch)
            
            # 记录损失
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_total_loss += total_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}',
                'Total': f'{total_loss.item():.4f}'
            })
            
            # TensorBoard记录
            if batch_idx % self.config['logging']['print_interval'] == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                if self.config['logging']['tensorboard']:
                    self.writer.add_scalar('Train/Discriminator_Loss', d_loss.item(), global_step)
                    self.writer.add_scalar('Train/Generator_Loss', g_loss.item(), global_step)
                    self.writer.add_scalar('Train/Total_Loss', total_loss.item(), global_step)
        
        # 计算平均损失
        avg_d_loss = epoch_d_loss / len(self.train_loader)
        avg_g_loss = epoch_g_loss / len(self.train_loader)
        avg_total_loss = epoch_total_loss / len(self.train_loader)
        
        return avg_g_loss, avg_d_loss, avg_total_loss
    
    def train_discriminator(self, ir_imgs, vis_imgs, target_imgs):
        """训练判别器"""
        self.d_optimizer.zero_grad()
        
        # 生成融合图像
        with torch.no_grad():
            fake_imgs = self.generator(ir_imgs, vis_imgs)
        
        # 判别器损失
        d_real = self.discriminator(target_imgs)
        d_fake = self.discriminator(fake_imgs.detach())
        
        d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
        d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss
    
    def train_generator(self, ir_imgs, vis_imgs, target_imgs, epoch):
        """训练生成器"""
        self.g_optimizer.zero_grad()
        
        # 生成融合图像
        fake_imgs = self.generator(ir_imgs, vis_imgs)
        
        # 计算各种损失
        d_fake = self.discriminator(fake_imgs)
        
        # 渐进式训练策略
        if self.config['training']['progressive']['enabled'] and epoch <= self.config['training']['progressive']['l1_warmup_epochs']:
            # 初期主要使用L1损失
            lambda_adv = 0.1
            lambda_fm = 0.1
            lambda_l1 = 1.0
        else:
            lambda_adv = self.config['training']['loss_weights']['adversarial']
            lambda_fm = self.config['training']['loss_weights']['feature_matching']
            lambda_l1 = self.config['training']['loss_weights']['l1']
        
        # 计算损失
        adversarial_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        l1_loss = F.l1_loss(fake_imgs, target_imgs)
        
        # 特征匹配损失
        if lambda_fm > 0:
            feature_matching_loss = self.criterion.feature_matching_loss(
                self.discriminator, target_imgs, fake_imgs
            )
        else:
            feature_matching_loss = torch.tensor(0.0, device=self.device)
        
        # 总生成器损失
        g_loss = (lambda_adv * adversarial_loss + 
                 lambda_fm * feature_matching_loss + 
                 lambda_l1 * l1_loss)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss, g_loss
    
    def validate(self, epoch):
        """验证模型"""
        self.generator.eval()
        self.discriminator.eval()
        
        val_metrics = {
            'ssim': [],
            'psnr': [],
            'mi': [],
            'std': [],
            'entropy': []
        }
        
        with torch.no_grad():
            for ir_imgs, vis_imgs, target_imgs in tqdm(self.val_loader, desc='Validation'):
                ir_imgs = ir_imgs.to(self.device)
                vis_imgs = vis_imgs.to(self.device)
                target_imgs = target_imgs.to(self.device)
                
                # 生成融合图像
                fake_imgs = self.generator(ir_imgs, vis_imgs)
                
                # 计算评估指标
                for i in range(ir_imgs.size(0)):
                    ir_img = ir_imgs[i].cpu()
                    vis_img = vis_imgs[i].cpu()
                    fused_img = fake_imgs[i].cpu()
                    
                    metrics = self.metrics.calculate_all_metrics(ir_img, vis_img, fused_img)
                    
                    for key in val_metrics:
                        val_metrics[key].append(metrics[key])
        
        # 计算平均指标
        avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
        
        # TensorBoard记录
        if self.config['logging']['tensorboard']:
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'Val/{key.upper()}', value, epoch)
        
        return avg_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'best_metric': self.best_metric,
            'train_history': self.train_history,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config['checkpoint']['save_dir'], 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_checkpoint_path = os.path.join(self.config['checkpoint']['save_dir'], 'best_checkpoint.pth')
            torch.save(checkpoint, best_checkpoint_path)
            self.logger.info(f"保存最佳模型 (epoch {epoch})")
        
        # 定期保存检查点
        if epoch % self.config['logging']['save_interval'] == 0:
            periodic_checkpoint_path = os.path.join(self.config['checkpoint']['save_dir'], f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, periodic_checkpoint_path)
    
    def train(self):
        """主训练函数"""
        self.logger.info("开始训练...")
        start_time = time.time()
        
        # 创建保存目录
        os.makedirs(self.config['checkpoint']['save_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['log_dir'], exist_ok=True)
        
        # 早停机制
        patience_counter = 0
        best_val_metric = 0.0
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            self.logger.info(f"\nEpoch {epoch}/{self.config['training']['epochs']}")
            
            # 训练
            train_g_loss, train_d_loss, train_total_loss = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch)
            
            # 记录训练历史
            self.train_history['generator_loss'].append(train_g_loss)
            self.train_history['discriminator_loss'].append(train_d_loss)
            self.train_history['total_loss'].append(train_total_loss)
            for key in ['ssim', 'psnr', 'mi', 'std', 'entropy']:
                self.train_history[key].append(val_metrics[key])
            
            # 打印训练结果
            self.logger.info(f"训练损失 - G: {train_g_loss:.4f}, D: {train_d_loss:.4f}, Total: {train_total_loss:.4f}")
            self.logger.info(f"验证指标 - SSIM: {val_metrics['ssim']:.4f}, PSNR: {val_metrics['psnr']:.2f}, MI: {val_metrics['mi']:.4f}")
            
            # 学习率调度
            if self.config['training']['lr_scheduler']['enabled']:
                self.g_scheduler.step()
                self.d_scheduler.step()
            
            # 检查最佳模型
            monitor_metric = val_metrics[self.config['evaluation']['monitor_metric']]
            if self.config['evaluation']['mode'] == 'max':
                is_best = monitor_metric > best_val_metric
            else:
                is_best = monitor_metric < best_val_metric
            
            if is_best:
                best_val_metric = monitor_metric
                self.best_metric = monitor_metric
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 早停检查
            if self.config['training']['early_stopping']['enabled'] and patience_counter >= self.config['training']['early_stopping']['patience']:
                self.logger.info(f"早停触发，在第 {epoch} 个epoch停止训练")
                break
            
            # 可视化训练进度
            if epoch % 10 == 0 and self.config['visualization']['enabled']:
                viz_path = os.path.join(self.config['logging']['log_dir'], f'training_progress_epoch_{epoch}.png')
                visualize_training_progress(self.train_history, save_path=viz_path)
        
        # 训练结束
        training_time = time.time() - start_time
        self.logger.info(f"训练完成！总耗时: {training_time:.2f}秒")
        self.logger.info(f"最佳模型在第 {self.best_epoch} 个epoch，{self.config['evaluation']['monitor_metric']}: {self.best_metric:.4f}")
        
        # 保存最终训练历史
        history_path = os.path.join(self.config['logging']['log_dir'], 'train_history.npy')
        np.save(history_path, self.train_history)
        self.logger.info(f"训练历史已保存到: {history_path}")
        
        # 关闭TensorBoard
        if self.config['logging']['tensorboard']:
            self.writer.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FusionGAN Training')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = FusionGANTrainer(args.config)
    
    # 恢复训练（如果指定）
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()