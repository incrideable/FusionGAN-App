import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import random

class ImageFusionDataset(Dataset):
    """红外与可见光图像融合数据集"""
    
    def __init__(self, ir_dir, vis_dir, transform=None, target_size=(256, 256), mode='train'):
        """
        Args:
            ir_dir: 红外图像目录
            vis_dir: 可见光图像目录
            transform: 图像变换
            target_size: 目标图像尺寸
            mode: 训练模式 ('train', 'val', 'test')
        """
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.transform = transform
        self.target_size = target_size
        self.mode = mode
        
        # 获取图像文件名列表
        self.ir_images = sorted([f for f in os.listdir(ir_dir) if f.endswith(('.jpg', '.png', '.bmp', '.tiff'))])
        self.vis_images = sorted([f for f in os.listdir(vis_dir) if f.endswith(('.jpg', '.png', '.bmp', '.tiff'))])
        
        # 确保两个目录中的图像数量一致
        assert len(self.ir_images) == len(self.vis_images), "红外和可见光图像数量不匹配"
        
        print(f"数据集初始化完成: {len(self.ir_images)} 对图像")
    
    def __len__(self):
        return len(self.ir_images)
    
    def __getitem__(self, idx):
        # 加载红外图像
        ir_path = os.path.join(self.ir_dir, self.ir_images[idx])
        ir_image = Image.open(ir_path).convert('L')  # 转为灰度图
        
        # 加载可见光图像
        vis_path = os.path.join(self.vis_dir, self.vis_images[idx])
        vis_image = Image.open(vis_path).convert('L')  # 转为灰度图
        
        # 确保图像尺寸一致
        if ir_image.size != vis_image.size:
            vis_image = vis_image.resize(ir_image.size, Image.LANCZOS)
        
        # 图像预处理
        ir_image = self.preprocess_image(ir_image)
        vis_image = self.preprocess_image(vis_image)
        
        # 数据增强（仅在训练模式下）
        if self.mode == 'train':
            ir_image, vis_image = self.apply_augmentation(ir_image, vis_image)
        
        # 调整图像尺寸
        ir_image = cv2.resize(ir_image, self.target_size)
        vis_image = cv2.resize(vis_image, self.target_size)
        
        # 转换为张量
        ir_tensor = torch.from_numpy(ir_image).float().unsqueeze(0)  # [1, H, W]
        vis_tensor = torch.from_numpy(vis_image).float().unsqueeze(0)  # [1, H, W]
        
        # 归一化到[-1, 1]
        ir_tensor = (ir_tensor / 127.5) - 1.0
        vis_tensor = (vis_tensor / 127.5) - 1.0
        
        # 拼接为双通道输入
        input_tensor = torch.cat([ir_tensor, vis_tensor], dim=0)  # [2, H, W]
        
        return {
            'input': input_tensor,
            'ir': ir_tensor,
            'vis': vis_tensor,
            'filename': self.ir_images[idx]
        }
    
    def preprocess_image(self, image):
        """图像预处理"""
        # PIL图像转numpy数组
        img_array = np.array(image, dtype=np.float32)
        
        # 直方图均衡化
        if len(img_array.shape) == 2:  # 灰度图像
            img_array = cv2.equalizeHist(img_array.astype(np.uint8)).astype(np.float32)
        
        return img_array
    
    def apply_augmentation(self, ir_image, vis_image):
        """应用数据增强"""
        # 随机水平翻转
        if random.random() > 0.5:
            ir_image = cv2.flip(ir_image, 1)
            vis_image = cv2.flip(vis_image, 1)
        
        # 随机垂直翻转
        if random.random() > 0.5:
            ir_image = cv2.flip(ir_image, 0)
            vis_image = cv2.flip(vis_image, 0)
        
        # 随机旋转
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = ir_image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            ir_image = cv2.warpAffine(ir_image, matrix, (w, h))
            vis_image