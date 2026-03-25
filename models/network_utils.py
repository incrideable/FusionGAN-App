import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianNoise(nn.Module):
    """高斯噪声层，用于数据增强"""
    def __init__(self, std=0.1):
        super(GaussianNoise, self).__init__()
        self.std = std
    
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class SpectralNormalization(nn.Module):
    """谱归一化包装器"""
    def __init__(self, module):
        super(SpectralNormalization, self).__init__()
        self.module = module
    
    def forward(self, x):
        # 简单的权重归一化
        weight = self.module.weight
        weight_norm = weight / (weight.norm() + 1e-8)
        self.module.weight.data = weight_norm
        return self.module(x)

def initialize_weights(module):
    """权重初始化函数"""
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)

def gradient_penalty(discriminator, real_imgs, fake_imgs, device):
    """计算梯度惩罚（用于WGAN-GP）"""
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    # 在真实图像和假图像之间插值
    interpolates = alpha * real_imgs + (1 - alpha) * fake_imgs
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    
    # 判别器对插值图像的判断
    d_interpolates, _ = discriminator(interpolates)
    
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 计算梯度惩罚
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

class FeatureExtractor(nn.Module):
    """特征提取器，用于感知损失计算"""
    def __init__(self, layers=['conv1_1', 'conv2_1', 'conv3_1']):
        super(FeatureExtractor, self).__init__()
        # 使用预训练的VGG19模型
        vgg19 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        features = vgg19.features
        
        self.layers = layers
        self.features = nn.ModuleDict()
        
        # 提取指定层的特征
        layer_names = []
        for i, layer in enumerate(features):
            if isinstance(layer, nn.Conv2d):
                layer_name = f'conv{len(layer_names)//2 + 1}_{len(layer_names)%2 + 1}'
                layer_names.append(layer_name)
                if layer_name in layers:
                    self.features[layer_name] = nn.Sequential(*features[:i+1])
        
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        results = {}
        for layer_name, layer in self.features.items():
            x = layer(x)
            results[layer_name] = x
        return results

def perceptual_loss(feature_extractor, real_imgs, fake_imgs):
    """感知损失计算"""
    # 确保图像是3通道的（VGG需要RGB输入）
    if real_imgs.size(1) == 1:
        real_imgs = real_imgs.repeat(1, 3, 1, 1)
    if fake_imgs.size(1) == 1:
        fake_imgs = fake_imgs.repeat(1, 3, 1, 1)
    
    # 提取特征
    real_features = feature_extractor(real_imgs)
    fake_features = feature_extractor(fake_imgs)
    
    # 计算特征损失
    loss = 0
    for layer in real_features:
        loss += F.l1_loss(fake_features[layer], real_features[layer])
    
    return loss / len(real_features)

class ImagePool:
    """图像池，用于存储生成的图像，提高判别器训练稳定性"""
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []
    
    def query(self, images):
        """从池中查询图像"""
        if self.pool_size == 0:
            return images
        
        return_images = []
        for image in images:
            image = image.unsqueeze(0)
            
            if len(self.images) < self.pool_size:
                # 池未满，直接添加
                self.images.append(image)
                return_images.append(image)
            else:
                # 池已满，随机替换
                if np.random.rand() > 0.5:
                    # 使用池中的图像
                    random_id = np.random.randint(0, self.pool_size)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    # 使用新图像
                    return_images.append(image)
        
        return torch.cat(return_images, dim=0)

def tensor_to_image(tensor):
    """将张量转换为图像numpy数组"""
    # 将张量从GPU移到CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 去掉batch维度
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
    # 将张量转换为numpy数组
    image = tensor.detach().numpy()
    
    # 如果是单通道图像，去掉通道维度
    if image.shape[0] == 1:
        image = image.squeeze(0)
    else:
        # 将通道维度移到最后
        image = image.transpose(1, 2, 0)
    
    # 将像素值从[-1, 1]转换到[0, 1]
    image = (image + 1) / 2
    image = np.clip(image, 0, 1)
    
    return image

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"检查点已保存: {filepath}")

def load_checkpoint(model, optimizer, filepath, device):
    """加载模型检查点"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"检查点已加载: {filepath}, Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss

def adjust_learning_rate(optimizer, epoch, initial_lr, decay_epoch=50):
    """调整学习率"""
    lr = initial_lr * (0.1 ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == "__main__":
    # 测试网络工具
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试特征提取器
    print("测试特征提取器...")
    feature_extractor = FeatureExtractor().to(device)
    test_img = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        features = feature_extractor(test_img)
    
    for layer_name, feature in features.items():
        print(f"{layer_name}: {feature.shape}")
    
    print("\n网络工具测试完成！")