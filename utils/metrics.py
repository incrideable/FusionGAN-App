import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import entropy
import cv2

def calculate_ssim(img1, img2, data_range=None):
    """计算结构相似性指数 (SSIM)
    
    Args:
        img1: 图像1 (numpy数组或torch张量)
        img2: 图像2 (numpy数组或torch张量)
        data_range: 数据范围，如果为None则自动计算
        
    Returns:
        SSIM值
    """
    # 转换为numpy数组
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # 确保图像是2D的
    if img1.ndim == 3:
        if img1.shape[0] == 1:  # [1, H, W]
            img1 = img1.squeeze(0)
        else:  # [C, H, W]
            img1 = np.transpose(img1, (1, 2, 0))
    
    if img2.ndim == 3:
        if img2.shape[0] == 1:
            img2 = img2.squeeze(0)
        else:
            img2 = np.transpose(img2, (1, 2, 0))
    
    # 计算数据范围
    if data_range is None:
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
    
    # 计算SSIM
    if img1.ndim == 2 and img2.ndim == 2:
        # 灰度图像
        return ssim(img1, img2, data_range=data_range)
    else:
        # 彩色图像，计算每个通道的SSIM然后平均
        ssim_values = []
        for i in range(img1.shape[2]):
            ssim_values.append(ssim(img1[:,:,i], img2[:,:,i], data_range=data_range))
        return np.mean(ssim_values)

def calculate_psnr(img1, img2, data_range=None):
    """计算峰值信噪比 (PSNR)
    
    Args:
        img1: 图像1 (numpy数组或torch张量)
        img2: 图像2 (numpy数组或torch张量)
        data_range: 数据范围，如果为None则自动计算
        
    Returns:
        PSNR值 (dB)
    """
    # 转换为numpy数组
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # 确保图像是2D的
    if img1.ndim == 3:
        if img1.shape[0] == 1:  # [1, H, W]
            img1 = img1.squeeze(0)
        else:  # [C, H, W]
            img1 = np.transpose(img1, (1, 2, 0))
    
    if img2.ndim == 3:
        if img2.shape[0] == 1:
            img2 = img2.squeeze(0)
        else:
            img2 = np.transpose(img2, (1, 2, 0))
    
    # 计算数据范围
    if data_range is None:
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
    
    # 计算PSNR
    return psnr(img1, img2, data_range=data_range)

def calculate_mutual_information(img1, img2, bins=256):
    """计算互信息 (Mutual Information)
    
    Args:
        img1: 图像1 (numpy数组或torch张量)
        img2: 图像2 (numpy数组或torch张量)
        bins: 直方图bin数量
        
    Returns:
        互信息值
    """
    # 转换为numpy数组
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # 确保图像是2D的
    if img1.ndim == 3:
        if img1.shape[0] == 1:  # [1, H, W]
            img1 = img1.squeeze(0)
        else:  # [C, H, W]
            img1 = np.mean(np.transpose(img1, (1, 2, 0)), axis=2)  # 转为灰度
    
    if img2.ndim == 3:
        if img2.shape[0] == 1:
            img2 = img2.squeeze(0)
        else:
            img2 = np.mean(np.transpose(img2, (1, 2, 0)), axis=2)
    
    # 确保图像在同一范围
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    
    # 计算联合直方图
    hist_2d, x_edges, y_edges = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    
    # 计算边缘直方图
    hist_1 = np.histogram(img1.ravel(), bins=bins)[0]
    hist_2 = np.histogram(img2.ravel(), bins=bins)[0]
    
    # 转换为概率分布
    p_12 = hist_2d / np.sum(hist_2d)
    p_1 = hist_1 / np.sum(hist_1)
    p_2 = hist_2 / np.sum(hist_2)
    
    # 避免log(0)
    p_12 = np.maximum(p_12, 1e-10)
    p_1 = np.maximum(p_1, 1e-10)
    p_2 = np.maximum(p_2, 1e-10)
    
    # 计算互信息
    mi = 0
    for i in range(bins):
        for j in range(bins):
            if p_12[i, j] > 0:
                mi += p_12[i, j] * np.log(p_12[i, j] / (p_1[i] * p_2[j]))
    
    return mi

def calculate_standard_deviation(img):
    """计算标准差 (Standard Deviation)
    
    Args:
        img: 图像 (numpy数组或torch张量)
        
    Returns:
        标准差值
    """
    # 转换为numpy数组
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    
    # 确保图像是2D的
    if img.ndim == 3:
        if img.shape[0] == 1:  # [1, H, W]
            img = img.squeeze(0)
        else:  # [C, H, W]
            img = np.mean(np.transpose(img, (1, 2, 0)), axis=2)  # 转为灰度
    
    return np.std(img)

def calculate_entropy(img):
    """计算图像熵
    
    Args:
        img: 图像 (numpy数组或torch张量)
        
    Returns:
        熵值
    """
    # 转换为numpy数组
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    
    # 确保图像是2D的
    if img.ndim == 3:
        if img.shape[0] == 1:  # [1, H, W]
            img = img.squeeze(0)
        else:  # [C, H, W]
            img = np.mean(np.transpose(img, (1, 2, 0)), axis=2)  # 转为灰度
    
    # 计算直方图
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    
    # 转换为概率分布
    prob = hist / np.sum(hist)
    
    # 避免log(0)
    prob = prob[prob > 0]
    
    # 计算熵
    return entropy(prob, base=2)

def calculate_spatial_frequency(img):
    """计算空间频率 (Spatial Frequency)
    
    Args:
        img: 图像 (numpy数组或torch张量)
        
    Returns:
        空间频率值
    """
    # 转换为numpy数组
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    
    # 确保图像是2D的
    if img.ndim == 3:
        if img.shape[0] == 1:  # [1, H, W]
            img = img.squeeze(0)
        else:  # [C, H, W]
            img = np.mean(np.transpose(img, (1, 2, 0)), axis=2)  # 转为灰度
    
    # 计算行频率
    rf = np.sqrt(np.mean((img[:, 1:] - img[:, :-1])**2))
    
    # 计算列频率
    cf = np.sqrt(np.mean((img[1:, :] - img[:-1, :])**2))
    
    # 空间频率
    return np.sqrt(rf**2 + cf**2)

class ImageFusionMetrics:
    """图像融合评估指标计算器"""
    
    def __init__(self):
        self.metrics_history = {
            'ssim': [],
            'psnr': [],
            'mi': [],
            'std': [],
            'entropy': [],
            'sf': []
        }
    
    def calculate_all_metrics(self, fused_img, ir_img, vis_img, reference_img=None):
        """计算所有评估指标
        
        Args:
            fused_img: 融合图像
            ir_img: 红外图像
            vis_img: 可见光图像
            reference_img: 参考图像（可选）
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 计算SSIM（如果有参考图像）
        if reference_img is not None:
            metrics['ssim'] = calculate_ssim(fused_img, reference_img)
        
        # 计算PSNR（如果有参考图像）
        if reference_img is not None:
            metrics['psnr'] = calculate_psnr(fused_img, reference_img)
        
        # 计算互信息（与红外和可见光图像）
        metrics['mi_ir'] = calculate_mutual_information(fused_img, ir_img)
        metrics['mi_vis'] = calculate_mutual_information(fused_img, vis_img)
        metrics['mi_avg'] = (metrics['mi_ir'] + metrics['mi_vis']) / 2
        
        # 计算标准差
        metrics['std'] = calculate_standard_deviation(fused_img)
        
        # 计算熵
        metrics['entropy'] = calculate_entropy(fused_img)
        
        # 计算空间频率
        metrics['sf'] = calculate_spatial_frequency(fused_img)
        
        return metrics
    
    def update_history(self, metrics):
        """更新历史记录"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def get_average_metrics(self):
        """获取平均指标"""
        avg_metrics = {}
        for key, values in self.metrics_history.items():
            if values:
                avg_metrics[f'{key}_avg'] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
        return avg_metrics
    
    def reset_history(self):
        """重置历史记录"""
        for key in self.metrics_history:
            self.metrics_history[key] = []

def compare_fusion_methods(results_dict):
    """比较不同融合方法的性能
    
    Args:
        results_dict: 不同方法的结果字典，格式为 {method_name: metrics_dict}
        
    Returns:
        比较结果表格
    """
    import pandas as pd
    
    # 创建DataFrame
    df = pd.DataFrame(results_dict).T
    
    # 格式化输出
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    return df

def test_metrics():
    """测试评估指标"""
    print("测试图像融合评估指标...")
    
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
    
    # 计算指标
    print("计算SSIM...")
    ssim_value = calculate_ssim(fused_img, vis_img)
    print(f"SSIM: {ssim_value:.4f}")
    
    print("计算PSNR...")
    psnr_value = calculate_psnr(fused_img, vis_img)
    print(f"PSNR: {psnr_value:.2f} dB")
    
    print("计算互信息...")
    mi_value = calculate_mutual_information(fused_img, ir_img)
    print(f"MI (IR): {mi_value:.4f}")
    
    print("计算标准差...")
    std_value = calculate_standard_deviation(fused_img)
    print(f"Standard Deviation: {std_value:.4f}")
    
    print("计算熵...")
    entropy_value = calculate_entropy(fused_img)
    print(f"Entropy: {entropy_value:.4f}")
    
    print("计算空间频率...")
    sf_value = calculate_spatial_frequency(fused_img)
    print(f"Spatial Frequency: {sf_value:.4f}")
    
    # 使用指标计算器
    print("\n使用ImageFusionMetrics计算器...")
    calculator = ImageFusionMetrics()
    metrics = calculator.calculate_all_metrics(fused_img, ir_img, vis_img)
    
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\n评估指标测试完成！")

if __name__ == "__main__":
    test_metrics()