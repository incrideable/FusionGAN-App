import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from matplotlib.gridspec import GridSpec
import seaborn as sns
from utils.metrics import ImageFusionMetrics

def visualize_fusion_results(ir_img, vis_img, fused_img, save_path=None, figsize=(15, 10)):
    """可视化图像融合结果
    
    Args:
        ir_img: 红外图像
        vis_img: 可见光图像
        fused_img: 融合图像
        save_path: 保存路径（可选）
        figsize: 图像大小
    """
    # 转换张量为numpy数组
    if torch.is_tensor(ir_img):
        ir_img = ir_img.detach().cpu().numpy()
    if torch.is_tensor(vis_img):
        vis_img = vis_img.detach().cpu().numpy()
    if torch.is_tensor(fused_img):
        fused_img = fused_img.detach().cpu().numpy()
    
    # 确保图像是2D的
    if ir_img.ndim == 3 and ir_img.shape[0] == 1:
        ir_img = ir_img.squeeze(0)
    if vis_img.ndim == 3 and vis_img.shape[0] == 1:
        vis_img = vis_img.squeeze(0)
    if fused_img.ndim == 3 and fused_img.shape[0] == 1:
        fused_img = fused_img.squeeze(0)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 红外图像
    im1 = axes[0, 0].imshow(ir_img, cmap='hot')
    axes[0, 0].set_title('红外图像 (Infrared)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # 可见光图像
    im2 = axes[0, 1].imshow(vis_img, cmap='gray')
    axes[0, 1].set_title('可见光图像 (Visible)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 融合图像
    im3 = axes[1, 0].imshow(fused_img, cmap='gray')
    axes[1, 0].set_title('融合图像 (FusionGAN)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 差异图
    diff_img = np.abs(fused_img - vis_img)
    im4 = axes[1, 1].imshow(diff_img, cmap='jet')
    axes[1, 1].set_title('差异图 (Difference)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"融合结果已保存到: {save_path}")
    
    return fig

def visualize_training_progress(loss_history, metric_history, save_path=None, figsize=(15, 8)):
    """可视化训练进度
    
    Args:
        loss_history: 损失历史记录字典
        metric_history: 指标历史记录字典
        save_path: 保存路径（可选）
        figsize: 图像大小
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 绘制损失曲线
    if 'generator_loss' in loss_history:
        epochs = range(1, len(loss_history['generator_loss']) + 1)
        axes[0, 0].plot(epochs, loss_history['generator_loss'], 'b-', label='Generator Loss', linewidth=2)
        axes[0, 0].plot(epochs, loss_history['discriminator_loss'], 'r-', label='Discriminator Loss', linewidth=2)
        axes[0, 0].set_title('训练损失', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 绘制SSIM曲线
    if 'ssim' in metric_history and metric_history['ssim']:
        epochs = range(1, len(metric_history['ssim']) + 1)
        axes[0, 1].plot(epochs, metric_history['ssim'], 'g-', linewidth=2)
        axes[0, 1].set_title('SSIM指标', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
    
    # 绘制PSNR曲线
    if 'psnr' in metric_history and metric_history['psnr']:
        epochs = range(1, len(metric_history['psnr']) + 1)
        axes[0, 2].plot(epochs, metric_history['psnr'], 'm-', linewidth=2)
        axes[0, 2].set_title('PSNR指标', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('PSNR (dB)')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 绘制互信息曲线
    if 'mi' in metric_history and metric_history['mi']:
        epochs = range(1, len(metric_history['mi']) + 1)
        axes[1, 0].plot(epochs, metric_history['mi'], 'c-', linewidth=2)
        axes[1, 0].set_title('互信息 (MI)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MI')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 绘制标准差曲线
    if 'std' in metric_history and metric_history['std']:
        epochs = range(1, len(metric_history['std']) + 1)
        axes[1, 1].plot(epochs, metric_history['std'], 'y-', linewidth=2)
        axes[1, 1].set_title('标准差 (Standard Deviation)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Std')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 绘制熵曲线
    if 'entropy' in metric_history and metric_history['entropy']:
        epochs = range(1, len(metric_history['entropy']) + 1)
        axes[1, 2].plot(epochs, metric_history['entropy'], 'k-', linewidth=2)
        axes[1, 2].set_title('图像熵 (Entropy)', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Entropy')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练进度图已保存到: {save_path}")
    
    return fig

def create_comparison_table(results_dict, save_path=None):
    """创建方法对比表格
    
    Args:
        results_dict: 不同方法的结果字典
        save_path: 保存路径（可选）
    """
    import pandas as pd
    
    # 创建DataFrame
    df = pd.DataFrame(results_dict).T
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(cellText=df.round(4).values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # 设置样式
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('图像融合方法性能对比', fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比表格已保存到: {save_path}")
    
    return fig

def visualize_feature_maps(feature_maps, save_path=None, max_channels=16):
    """可视化特征图
    
    Args:
        feature_maps: 特征图张量 [C, H, W] 或 [B, C, H, W]
        save_path: 保存路径（可选）
        max_channels: 最大通道数
    """
    if torch.is_tensor(feature_maps):
        feature_maps = feature_maps.detach().cpu().numpy()
    
    # 处理批次维度
    if feature_maps.ndim == 4:
        feature_maps = feature_maps[0]  # 取第一个样本
    
    channels = min(feature_maps.shape[0], max_channels)
    rows = int(np.sqrt(channels))
    cols = int(np.ceil(channels / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(channels):
        row = i // cols
        col = i % cols
        if rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        
        feature_map = feature_maps[i]
        im = ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f'Channel {i+1}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for i in range(channels, rows * cols):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].axis('off')
        else:
            axes[col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征图已保存到: {save_path}")
    
    return fig

def create_heatmap_comparison(ir_img, vis_img, fused_img, save_path=None):
    """创建热力图对比
    
    Args:
        ir_img: 红外图像
        vis_img: 可见光图像
        fused_img: 融合图像
        save_path: 保存路径（可选）
    """
    # 转换张量为numpy数组
    if torch.is_tensor(ir_img):
        ir_img = ir_img.detach().cpu().numpy()
    if torch.is_tensor(vis_img):
        vis_img = vis_img.detach().cpu().numpy()
    if torch.is_tensor(fused_img):
        fused_img = fused_img.detach().cpu().numpy()
    
    # 确保图像是2D的
    if ir_img.ndim == 3 and ir_img.shape[0] == 1:
        ir_img = ir_img.squeeze(0)
    if vis_img.ndim == 3 and vis_img.shape[0] == 1:
        vis_img = vis_img.squeeze(0)
    if fused_img.ndim == 3 and fused_img.shape[0] == 1:
        fused_img = fused_img.squeeze(0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 红外热力图
    im1 = axes[0, 0].imshow(ir_img, cmap='hot')
    axes[0, 0].set_title('红外热力图', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 可见光热力图
    im2 = axes[0, 1].imshow(vis_img, cmap='gray')
    axes[0, 1].set_title('可见光图像', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 融合热力图
    im3 = axes[1, 0].imshow(fused_img, cmap='plasma')
    axes[1, 0].set_title('融合热力图', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 温度分布对比
    axes[1, 1].hist(ir_img.flatten(), bins=50, alpha=0.5, label='红外', color='red')
    axes[1, 1].hist(fused_img.flatten(), bins=50, alpha=0.5, label='融合', color='blue')
    axes[1, 1].set_title('温度分布对比', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('像素值')
    axes[1, 1].set_ylabel('频率')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图对比已保存到: {save_path}")
    
    return fig

def test_visualization():
    """测试可视化功能"""
    print("测试可视化功能...")
    
    # 创建测试图像
    np.random.seed(42)
    h, w = 256, 256
    
    # 模拟红外图像（高对比度）
    ir_img = np.random.rand(h, w)
    ir_img[100:150, 100:150] = 1.0  # 热点
    
    # 模拟可见光图像（丰富纹理）
    vis_img = np.random.rand(h, w)
    vis_img = cv2.GaussianBlur(vis_img, (5, 5), 1.0)
    
    # 模拟融合图像
    fused_img = 0.6 * vis_img + 0.4 * ir_img
    
    # 测试基本可视化
    print("测试基本融合结果可视化...")
    fig1 = visualize_fusion_results(ir_img, vis_img, fused_img)
    plt.close(fig1)
    
    # 测试训练进度可视化
    print("测试训练进度可视化...")
    loss_history = {
        'generator_loss': np.random.rand(50) * 0.1,
        'discriminator_loss': np.random.rand(50) * 0.1
    }
    metric_history = {
        'ssim': np.linspace(0.6, 0.8, 50),
        'psnr': np.linspace(25, 30, 50),
        'mi': np.linspace(5, 7, 50),
        'std': np.linspace(10, 15, 50),
        'entropy': np.linspace(6, 7, 50)
    }
    fig2 = visualize_training_progress(loss_history, metric_history)
    plt.close(fig2)
    
    # 测试对比表格
    print("测试对比表格...")
    results_dict = {
        'FusionGAN': {'SSIM': 0.81, 'PSNR': 31.4, 'MI': 7.03, 'STD': 14.6},
        'DeepFuse': {'SSIM': 0.75, 'PSNR': 29.1, 'MI': 6.67, 'STD': 13.2},
        'GTF': {'SSIM': 0.72, 'PSNR': 28.5, 'MI': 6.31, 'STD': 12.4}
    }
    fig3 = create_comparison_table(results_dict)
    plt.close(fig3)
    
    # 测试热力图对比
    print("测试热力图对比...")
    fig4 = create_heatmap_comparison(ir_img, vis_img, fused_img)
    plt.close(fig4)
    
    print("可视化功能测试完成！")

if __name__ == "__main__":
    test_visualization()