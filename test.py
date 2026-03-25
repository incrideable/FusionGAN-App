#!/usr/bin/env python3
"""
FusionGAN 测试脚本
用于模型推理和结果评估
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import yaml
from tqdm import tqdm

# 导入自定义模块
from models.generator import FusionGenerator
from datasets.data_loader import FusionDataset
from utils.metrics import ImageFusionMetrics
from utils.visualization import (
    visualize_fusion_results, 
    create_comparison_grid,
    create_metrics_comparison,
    create_heatmap_comparison
)
from utils.logger import setup_logger

class FusionGANTester:
    def __init__(self, config_path, checkpoint_path, output_dir="results"):
        """初始化测试器
        
        Args:
            config_path: 配置文件路径
            checkpoint_path: 模型检查点路径
            output_dir: 输出目录
        """
        self.config = self.load_config(config_path)
        self.device = torch.device(f'cuda:{self.config["device"]["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
        
        # 设置输出目录
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "comparisons"), exist_ok=True)
        
        # 初始化日志
        self.logger = setup_logger('FusionGAN_Test', self.output_dir)
        self.logger.info(f"使用设备: {self.device}")
        
        # 加载模型
        self.load_model(checkpoint_path)
        
        # 初始化评估指标
        self.metrics = ImageFusionMetrics()
        
        # 测试结果存储
        self.test_results = {
            'ssim': [],
            'psnr': [],
            'mi': [],
            'std': [],
            'entropy': [],
            'sf': [],
            'avg_grad': [],
            'qabf': []
        }
    
    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_model(self, checkpoint_path):
        """加载训练好的模型"""
        self.logger.info(f"加载模型检查点: {checkpoint_path}")
        
        # 初始化生成器
        self.generator = FusionGenerator(
            in_channels=self.config['model']['generator']['in_channels'],
            out_channels=self.config['model']['generator']['out_channels'],
            features=self.config['model']['generator']['features']
        ).to(self.device)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        
        self.logger.info("模型加载完成")
    
    def test_single_image(self, ir_path, vis_path, save_name=None):
        """测试单对图像
        
        Args:
            ir_path: 红外图像路径
            vis_path: 可见光图像路径
            save_name: 保存文件名
            
        Returns:
            fused_img: 融合后的图像
            metrics: 评估指标
        """
        # 加载图像
        ir_img = self.load_image(ir_path)
        vis_img = self.load_image(vis_path)
        
        # 预处理
        ir_tensor = self.preprocess_image(ir_img)
        vis_tensor = self.preprocess_image(vis_img)
        
        # 模型推理
        with torch.no_grad():
            fused_tensor = self.generator(ir_tensor, vis_tensor)
        
        # 后处理
        fused_img = self.postprocess_image(fused_tensor)
        
        # 计算评估指标
        metrics = self.metrics.calculate_all_metrics(
            ir_tensor.squeeze().cpu(), 
            vis_tensor.squeeze().cpu(), 
            fused_tensor.squeeze().cpu()
        )
        
        # 保存结果
        if save_name:
            self.save_single_result(ir_img, vis_img, fused_img, metrics, save_name)
        
        return fused_img, metrics
    
    def load_image(self, image_path):
        """加载图像"""
        img = Image.open(image_path).convert('L')  # 转为灰度图
        return np.array(img)
    
    def preprocess_image(self, img_array):
        """预处理图像"""
        # 调整大小
        target_size = tuple(self.config['dataset']['image_size'])
        img_resized = cv2.resize(img_array, target_size)
        
        # 归一化到[-1, 1]
        img_normalized = (img_resized.astype(np.float32) / 255.0) * 2.0 - 1.0
        
        # 转为tensor并添加batch维度
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).float()
        
        return img_tensor.to(self.device)
    
    def postprocess_image(self, tensor):
        """后处理tensor为图像"""
        # 移除batch维度并转为numpy
        img_array = tensor.squeeze().cpu().numpy()
        
        # 反归一化到[0, 255]
        img_array = ((img_array + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        
        return img_array
    
    def save_single_result(self, ir_img, vis_img, fused_img, metrics, save_name):
        """保存单个测试结果"""
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(ir_img, cmap='gray')
        axes[0].set_title('红外图像')
        axes[0].axis('off')
        
        axes[1].imshow(vis_img, cmap='gray')
        axes[1].set_title('可见光图像')
        axes[1].axis('off')
        
        axes[2].imshow(fused_img, cmap='gray')
        axes[2].set_title(f'融合图像\nSSIM: {metrics["ssim"]:.3f}, PSNR: {metrics["psnr"]:.2f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.output_dir, "comparisons", f"{save_name}_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存融合图像
        fused_path = os.path.join(self.output_dir, "images", f"{save_name}_fused.png")
        Image.fromarray(fused_img).save(fused_path)
        
        self.logger.info(f"保存结果: {save_name}")
        self.logger.info(f"指标 - SSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.2f}, MI: {metrics['mi']:.4f}")
    
    def test_dataset(self, data_dir=None, num_samples=None):
        """测试整个数据集
        
        Args:
            data_dir: 数据目录，如果为None则使用配置文件中的路径
            num_samples: 测试样本数量，如果为None则测试所有样本
        """
        if data_dir is None:
            data_dir = self.config['dataset']['data_dir']
        
        self.logger.info(f"开始测试数据集: {data_dir}")
        
        # 创建测试数据集
        test_dataset = FusionDataset(
            data_dir=data_dir,
            split='test',
            image_size=tuple(self.config['dataset']['image_size']),
            augmentation={'enabled': False}
        )
        
        if num_samples is None:
            num_samples = len(test_dataset)
        else:
            num_samples = min(num_samples, len(test_dataset))
        
        self.logger.info(f"测试样本数量: {num_samples}")
        
        # 随机选择样本进行测试
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        for i, idx in enumerate(tqdm(indices, desc="Testing")):
            ir_img, vis_img, target_img = test_dataset[idx]
            
            # 添加batch维度
            ir_tensor = ir_img.unsqueeze(0).to(self.device)
            vis_tensor = vis_img.unsqueeze(0).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                fused_tensor = self.generator(ir_tensor, vis_tensor)
            
            # 计算指标
            metrics = self.metrics.calculate_all_metrics(
                ir_img, vis_img, fused_tensor.squeeze().cpu()
            )
            
            # 保存结果
            for key in self.test_results:
                self.test_results[key].append(metrics[key])
            
            # 可视化部分结果
            if i < self.config['visualization']['num_samples']:
                ir_array = ((ir_img.squeeze().cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
                vis_array = ((vis_img.squeeze().cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
                fused_array = ((fused_tensor.squeeze().cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
                
                self.save_single_result(
                    ir_array, vis_array, fused_array, metrics, 
                    f"sample_{i:03d}"
                )
        
        # 计算平均指标
        avg_metrics = {key: np.mean(values) for key, values in self.test_results.items()}
        
        self.logger.info("测试完成！")
        self.logger.info("平均指标:")
        for key, value in avg_metrics.items():
            self.logger.info(f"  {key.upper()}: {value:.4f}")
        
        return avg_metrics
    
    def compare_methods(self, test_data_dir, methods=['avg', 'max', 'pcnn']):
        """比较不同融合方法
        
        Args:
            test_data_dir: 测试数据目录
            methods: 要比较的方法列表
        """
        self.logger.info("开始比较不同融合方法")
        
        # 创建简单的测试数据集
        test_dataset = FusionDataset(
            data_dir=test_data_dir,
            split='test',
            image_size=tuple(self.config['dataset']['image_size']),
            augmentation={'enabled': False}
        )
        
        # 选择几个样本进行测试
        num_samples = min(5, len(test_dataset))
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        results = {}
        
        for method in methods:
            results[method] = {
                'ssim': [],
                'psnr': [],
                'mi': [],
                'std': []
            }
        
        for i, idx in enumerate(tqdm(indices, desc="Comparing methods")):
            ir_img, vis_img, _ = test_dataset[idx]
            
            # 传统方法融合
            ir_array = ((ir_img.squeeze().cpu().numpy() + 1.0) / 2.0)
            vis_array = ((vis_img.squeeze().cpu().numpy() + 1.0) / 2.0)
            
            for method in methods:
                if method == 'avg':
                    fused = (ir_array + vis_array) / 2.0
                elif method == 'max':
                    fused = np.maximum(ir_array, vis_array)
                elif method == 'pcnn':
                    # 简化的PCNN方法
                    fused = 0.6 * vis_array + 0.4 * ir_array
                else:
                    continue
                
                # 计算指标
                fused_tensor = torch.from_numpy(fused).float()
                ir_tensor = torch.from_numpy(ir_array).float()
                vis_tensor = torch.from_numpy(vis_array).float()
                
                metrics = self.metrics.calculate_all_metrics(ir_tensor, vis_tensor, fused_tensor)
                
                for key in results[method]:
                    results[method][key].append(metrics[key])
        
        # 计算平均值
        avg_results = {}
        for method in methods:
            avg_results[method] = {
                key: np.mean(values) for key, values in results[method].items()
            }
        
        # 创建对比图表
        self.create_comparison_chart(avg_results)
        
        return avg_results
    
    def create_comparison_chart(self, results):
        """创建方法对比图表"""
        methods = list(results.keys())
        metrics = ['ssim', 'psnr', 'mi', 'std']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [results[method][metric] for method in methods]
            
            axes[i].bar(methods, values, alpha=0.7)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].grid(True, alpha=0.3)
            
            # 添加数值标签
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01 * max(values), f'{v:.3f}', 
                           ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(self.output_dir, "method_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"方法对比图表已保存: {save_path}")
    
    def create_test_report(self):
        """创建测试报告"""
        report_path = os.path.join(self.output_dir, "test_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("FusionGAN 测试报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 平均指标
            f.write("平均评估指标:\n")
            f.write("-" * 30 + "\n")
            for key, values in self.test_results.items():
                if values:
                    avg_value = np.mean(values)
                    std_value = np.std(values)
                    f.write(f"{key.upper()}: {avg_value:.4f} ± {std_value:.4f}\n")
            
            # 模型信息
            f.write(f"\n模型配置:\n")
            f.write("-" * 30 + "\n")
            f.write(f"生成器参数量: {sum(p.numel() for p in self.generator.parameters()):,}\n")
            f.write(f"输入图像大小: {self.config['dataset']['image_size']}\n")
            f.write(f"设备: {self.device}\n")
            
            # 测试配置
            f.write(f"\n测试配置:\n")
            f.write("-" * 30 + "\n")
            f.write(f"测试样本数: {len(list(self.test_results.values())[0])}\n")
            f.write(f"输出目录: {self.output_dir}\n")
        
        self.logger.info(f"测试报告已保存: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FusionGAN Testing')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='测试数据目录')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='测试样本数量')
    parser.add_argument('--compare_methods', action='store_true',
                        help='比较不同融合方法')
    parser.add_argument('--ir_image', type=str, default=None,
                        help='单张红外图像路径')
    parser.add_argument('--vis_image', type=str, default=None,
                        help='单张可见光图像路径')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = FusionGANTester(args.config, args.checkpoint, args.output_dir)
    
    # 单张图像测试
    if args.ir_image and args.vis_image:
        fused_img, metrics = tester.test_single_image(
            args.ir_image, args.vis_image, "single_test"
        )
        print(f"单张图像测试结果:")
        for key, value in metrics.items():
            print(f"  {key.upper()}: {value:.4f}")
    
    # 数据集测试
    elif args.test_dir:
        avg_metrics = tester.test_dataset(args.test_dir, args.num_samples)
        
        # 方法比较
        if args.compare_methods:
            comparison_results = tester.compare_methods(args.test_dir)
            print("\n方法对比结果:")
            for method, metrics in comparison_results.items():
                print(f"\n{method.upper()}:")
                for key, value in metrics.items():
                    print(f"  {key.upper()}: {value:.4f}")
    
    # 创建测试报告
    tester.create_test_report()
    
    print(f"\n测试完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()