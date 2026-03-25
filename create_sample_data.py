#!/usr/bin/env python3
"""
创建示例数据脚本
用于生成红外和可见光图像对的示例数据
"""

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_synthetic_ir_image(size=(256, 256), num_heat_sources=3):
    """创建合成的红外图像
    
    Args:
        size: 图像尺寸
        num_heat_sources: 热源数量
        
    Returns:
        合成的红外图像
    """
    # 创建基础背景（低温）
    background = np.random.normal(15, 5, size)  # 15±5°C
    
    # 添加热源
    ir_image = background.copy()
    
    for _ in range(num_heat_sources):
        # 随机热源位置和大小
        center_x = np.random.randint(size[1]//4, 3*size[1]//4)
        center_y = np.random.randint(size[0]//4, 3*size[0]//4)
        radius = np.random.randint(20, 60)
        temperature = np.random.randint(30, 80)  # 30-80°C
        
        # 创建圆形热源
        y, x = np.ogrid[:size[0], :size[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # 添加高斯分布的热源
        heat_source = temperature * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * radius**2))
        ir_image = np.maximum(ir_image, heat_source)
    
    # 添加噪声
    noise = np.random.normal(0, 2, size)
    ir_image += noise
    
    # 归一化到0-255
    ir_image = np.clip(ir_image, 0, 100)
    ir_image = (ir_image / 100.0 * 255).astype(np.uint8)
    
    return ir_image

def create_synthetic_vis_image(size=(256, 256), complexity='medium'):
    """创建合成的可见光图像
    
    Args:
        size: 图像尺寸
        complexity: 复杂度 ('simple', 'medium', 'complex')
        
    Returns:
        合成的可见光图像
    """
    if complexity == 'simple':
        # 简单场景：渐变背景 + 几何形状
        vis_image = np.zeros((*size, 3), dtype=np.uint8)
        
        # 渐变背景
        for i in range(size[0]):
            vis_image[i, :] = int(50 + 100 * i / size[0])
        
        # 添加几何形状
        # 矩形
        cv2.rectangle(vis_image, (50, 50), (150, 150), (200, 200, 200), -1)
        # 圆形
        cv2.circle(vis_image, (180, 80), 30, (100, 150, 200), -1)
        
    elif complexity == 'medium':
        # 中等复杂度：自然场景模拟
        vis_image = np.ones((*size, 3), dtype=np.uint8) * 100
        
        # 添加纹理
        texture = np.random.randint(-20, 20, size)
        for c in range(3):
            vis_image[:, :, c] = np.clip(vis_image[:, :, c] + texture, 0, 255)
        
        # 添加建筑物轮廓
        cv2.rectangle(vis_image, (30, 30), (100, 200), (150, 150, 150), 3)
        cv2.rectangle(vis_image, (120, 50), (200, 180), (120, 120, 120), 3)
        
        # 添加树木（圆形表示）
        cv2.circle(vis_image, (80, 50), 20, (50, 150, 50), -1)
        cv2.circle(vis_image, (160, 70), 25, (40, 140, 40), -1)
        
    else:  # complex
        # 复杂场景：城市街景模拟
        vis_image = np.ones((*size, 3), dtype=np.uint8) * 80
        
        # 天空
        vis_image[:size[0]//3, :] = [135, 206, 235]  # 天蓝色
        
        # 道路
        road_start = size[0]//3
        vis_image[road_start:, :] = [100, 100, 100]  # 灰色道路
        
        # 建筑物
        building_colors = [
            [200, 200, 200], [180, 180, 180], [160, 160, 160],
            [140, 140, 140], [120, 120, 120]
        ]
        
        for i, (x, w, h) in enumerate([
            (20, 40, 120), (80, 50, 140), (150, 45, 120),
            (210, 35, 100), (260, 40, 130)
        ]):
            color = building_colors[i % len(building_colors)]
            cv2.rectangle(vis_image, (x, size[0]//3), (x+w, size[0]//3 + h), color, -1)
        
        # 添加窗户
        for building in [(20, 40, 120), (80, 50, 140)]:
            x, w, h = building
            for wx in range(x+5, x+w-5, 10):
                for wy in range(size[0]//3+5, size[0]//3+h-5, 15):
                    cv2.rectangle(vis_image, (wx, wy), (wx+5, wy+5), (50, 50, 100), -1)
    
    # 添加噪声
    noise = np.random.randint(-10, 10, (*size, 3))
    vis_image = np.clip(vis_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return vis_image

def create_matching_image_pair(size=(256, 256), pair_type='person'):
    """创建匹配的图像对（红外和可见光）
    
    Args:
        size: 图像尺寸
        pair_type: 图像对类型 ('person', 'vehicle', 'building')
        
    Returns:
        ir_image, vis_image: 红外和可见光图像对
    """
    if pair_type == 'person':
        # 行人场景
        vis_image = np.ones((*size, 3), dtype=np.uint8) * 120
        
        # 背景
        cv2.rectangle(vis_image, (0, size[0]//2), (size[1], size[0]), (80, 80, 80), -1)  # 地面
        cv2.rectangle(vis_image, (0, 0), (size[1], size[0]//3), (200, 200, 200), -1)  # 天空
        
        # 行人（可见光）
        person_x = size[1]//2
        person_y = size[0]//2
        cv2.rectangle(vis_image, (person_x-10, person_y-30), (person_x+10, person_y), (100, 50, 50), -1)  # 身体
        cv2.circle(vis_image, (person_x, person_y-40), 8, (200, 150, 150), -1)  # 头部
        
        # 红外图像 - 行人温度较高
        ir_image = np.ones(size, dtype=np.uint8) * 50  # 背景低温
        
        # 地面
        cv2.rectangle(ir_image, (0, size[0]//2), (size[1], size[0]), 40, -1)
        
        # 行人（高温）
        cv2.rectangle(ir_image, (person_x-10, person_y-30), (person_x+10, person_y), 200, -1)  # 身体高温
        cv2.circle(ir_image, (person_x, person_y-40), 8, 180, -1)  # 头部
        
        # 添加环境热源（如路灯）
        cv2.circle(ir_image, (person_x+50, person_y-60), 5, 150, -1)
        
    elif pair_type == 'vehicle':
        # 车辆场景
        vis_image = np.ones((*size, 3), dtype=np.uint8) * 100
        
        # 道路
        cv2.rectangle(vis_image, (0, size[0]//3), (size[1], 2*size[0]//3), (60, 60, 60), -1)
        
        # 车辆（可见光）
        car_x, car_y = size[1]//3, size[0]//2
        cv2.rectangle(vis_image, (car_x-30, car_y-15), (car_x+30, car_y+15), (150, 150, 200), -1)
        cv2.rectangle(vis_image, (car_x-25, car_y-20), (car_x+25, car_y-10), (100, 100, 150), -1)
        
        # 红外图像
        ir_image = np.ones(size, dtype=np.uint8) * 40
        
        # 道路
        cv2.rectangle(ir_image, (0, size[0]//3), (size[1], 2*size[0]//3), 45, -1)
        
        # 车辆（发动机和轮胎温度较高）
        cv2.rectangle(ir_image, (car_x-30, car_y-15), (car_x+30, car_y+15), 180, -1)  # 车身
        cv2.rectangle(ir_image, (car_x-20, car_y-5), (car_x+20, car_y+5), 220, -1)  # 发动机区域
        
        # 轮胎（摩擦生热）
        cv2.circle(ir_image, (car_x-25, car_y+15), 8, 160, -1)
        cv2.circle(ir_image, (car_x+25, car_y+15), 8, 160, -1)
        
    else:  # building
        # 建筑物场景
        vis_image = np.ones((*size, 3), dtype=np.uint8) * 90
        
        # 建筑物（可见光）
        building_x, building_y = size[1]//2, size[0]//2
        cv2.rectangle(vis_image, (building_x-50, building_y-60), (building_x+50, building_y+20), (180, 180, 180), -1)
        
        # 窗户
        for wx in range(building_x-40, building_x+40, 20):
            for wy in range(building_y-50, building_y+10, 25):
                cv2.rectangle(vis_image, (wx, wy), (wx+8, wy+10), (100, 100, 150), -1)
        
        # 红外图像
        ir_image = np.ones(size, dtype=np.uint8) * 35
        
        # 建筑物（墙体温度）
        cv2.rectangle(ir_image, (building_x-50, building_y-60), (building_x+50, building_y+20), 80, -1)
        
        # 窗户（可能有不同的热特性）
        for wx in range(building_x-40, building_x+40, 20):
            for wy in range(building_y-50, building_y+10, 25):
                # 部分窗户温度较高（可能室内有暖气）
                temp = 90 if np.random.random() > 0.5 else 70
                cv2.rectangle(ir_image, (wx, wy), (wx+8, wy+10), temp, -1)
        
        # 空调外机（高温）
        cv2.rectangle(ir_image, (building_x+55, building_y-20), (building_x+65, building_y-10), 150, -1)
    
    return ir_image, vis_image

def create_sample_dataset(output_dir="data/samples", num_pairs=20):
    """创建完整的示例数据集
    
    Args:
        output_dir: 输出目录
        num_pairs: 图像对数量
    """
    print(f"创建示例数据集到: {output_dir}")
    
    # 创建目录
    ir_dir = os.path.join(output_dir, "ir")
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(ir_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # 场景类型
    scene_types = ['person', 'vehicle', 'building', 'simple', 'medium', 'complex']
    
    for i in tqdm(range(num_pairs), desc="创建图像对"):
        # 随机选择场景类型
        scene_type = np.random.choice(scene_types)
        
        if scene_type in ['person', 'vehicle', 'building']:
            ir_img, vis_img = create_matching_image_pair(pair_type=scene_type)
        else:
            # 创建独立的红外和可见光图像
            ir_img = create_synthetic_ir_image(num_heat_sources=np.random.randint(2, 6))
            vis_img = create_synthetic_vis_image(complexity=scene_type)
        
        # 保存图像
        ir_path = os.path.join(ir_dir, f"ir_{i:03d}.png")
        vis_path = os.path.join(vis_dir, f"vis_{i:03d}.png")
        
        Image.fromarray(ir_img).save(ir_path)
        Image.fromarray(vis_img).save(vis_path)
    
    print(f"数据集创建完成！共 {num_pairs} 对图像")
    print(f"红外图像保存在: {ir_dir}")
    print(f"可见光图像保存在: {vis_dir}")
    
    # 创建数据集信息文件
    info_path = os.path.join(output_dir, "dataset_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"FusionGAN 示例数据集\n")
        f.write(f"创建时间: {np.datetime64('now')}\n")
        f.write(f"图像对数量: {num_pairs}\n")
        f.write(f"图像尺寸: 256x256\n")
        f.write(f"场景类型: {', '.join(scene_types)}\n")
    
    print(f"数据集信息已保存: {info_path}")

def visualize_sample_data(data_dir="data/samples", num_samples=5):
    """可视化示例数据
    
    Args:
        data_dir: 数据目录
        num_samples: 要显示的样本数量
    """
    ir_dir = os.path.join(data_dir, "ir")
    vis_dir = os.path.join(data_dir, "vis")
    
    # 获取图像列表
    ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith('.png')])
    vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.png')])
    
    if len(ir_files) != len(vis_files):
        print("警告: 红外和可见光图像数量不匹配")
        return
    
    # 随机选择样本
    indices = np.random.choice(len(ir_files), min(num_samples, len(ir_files)), replace=False)
    
    # 创建可视化
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # 加载图像
        ir_img = np.array(Image.open(os.path.join(ir_dir, ir_files[idx])))
        vis_img = np.array(Image.open(os.path.join(vis_dir, vis_files[idx])))
        
        # 显示红外图像
        axes[i, 0].imshow(ir_img, cmap='hot')
        axes[i, 0].set_title(f'红外图像 - {ir_files[idx]}')
        axes[i, 0].axis('off')
        
        # 显示可见光图像
        if len(vis_img.shape) == 3:
            axes[i, 1].imshow(vis_img)
        else:
            axes[i, 1].imshow(vis_img, cmap='gray')
        axes[i, 1].set_title(f'可见光图像 - {vis_files[idx]}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "sample_visualization.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"样本可视化已保存: {os.path.join(data_dir, 'sample_visualization.png')}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='创建FusionGAN示例数据')
    parser.add_argument('--output_dir', type=str, default='data/samples',
                        help='输出目录')
    parser.add_argument('--num_pairs', type=int, default=20,
                        help='图像对数量')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化样本')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='可视化样本数量')
    
    args = parser.parse_args()
    
    # 创建示例数据
    create_sample_dataset(args.output_dir, args.num_pairs)
    
    # 可视化
    if args.visualize:
        visualize_sample_data(args.output_dir, args.num_samples)