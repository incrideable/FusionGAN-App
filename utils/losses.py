import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from models.network_utils import FeatureExtractor

class AdversarialLoss(nn.Module):
    """对抗损失"""
    def __init__(self, loss_type='vanilla'):
        super(AdversarialLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, fake_output, real_output=None, is_generator=True):
        if self.loss_type == 'vanilla':
            if is_generator:
                # 生成器损失：让判别器认为假图像是真实的
                return F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))
            else:
                # 判别器损失：正确区分真实和虚假图像
                real_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_loss))
                fake_loss = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))
                return real_loss + fake_loss
        
        elif self.loss_type == 'lsgan':
            if is_generator:
                # LSGAN生成器损失
                return torch.mean((fake_output - 1) ** 2)
            else:
                # LSGAN判别器损失
                real_loss = torch.mean((real_output - 1) ** 2)
                fake_loss = torch.mean(fake_output ** 2)
                return (real_loss + fake_loss) * 0.5
        
        elif self.loss_type == 'wgan':
            if is_generator:
                # WGAN生成器损失
                return -torch.mean(fake_output)
            else:
                # WGAN判别器损失
                return torch.mean(fake_output) - torch.mean(real_output)
        
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")

class FeatureMatchingLoss(nn.Module):
    """特征匹配损失"""
    def __init__(self, layers=['conv1_1', 'conv2_1', 'conv3_1']):
        super(FeatureMatchingLoss, self).__init__()
        self.layers = layers
    
    def forward(self, real_features, fake_features):
        """计算特征匹配损失
        
        Args:
            real_features: 真实图像的特征字典
            fake_features: 生成图像的特征字典
            
        Returns:
            特征匹配损失
        """
        loss = 0
        for layer in self.layers:
            if layer in real_features and layer in fake_features:
                real_feat = real_features[layer]
                fake_feat = fake_features[layer]
                loss += F.l1_loss(fake_feat, real_feat.detach())
        
        return loss / len(self.layers) if self.layers else loss

class PerceptualLoss(nn.Module):
    """感知损失（VGG特征损失）"""
    def __init__(self, layers=['conv1_1', 'conv2_1', 'conv3_1']):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = FeatureExtractor(layers)
        self.criterion = nn.L1Loss()
    
    def forward(self, real_imgs, fake_imgs):
        """计算感知损失
        
        Args:
            real_imgs: 真实图像 [B, C, H, W]
            fake_imgs: 生成图像 [B, C, H, W]
            
        Returns:
            感知损失
        """
        # 确保图像是3通道的
        if real_imgs.size(1) == 1:
            real_imgs = real_imgs.repeat(1, 3, 1, 1)
        if fake_imgs.size(1) == 1:
            fake_imgs = fake_imgs.repeat(1, 3, 1, 1)
        
        # 提取特征
        real_features = self.feature_extractor(real_imgs)
        fake_features = self.feature_extractor(fake_imgs)
        
        # 计算特征损失
        loss = 0
        for layer in real_features:
            loss += self.criterion(fake_features[layer], real_features[layer].detach())
        
        return loss / len(real_features) if real_features else loss

class L1Loss(nn.Module):
    """L1损失"""
    def __init__(self):
        super(L1Loss, self).__init__()
        self.criterion = nn.L1Loss()
    
    def forward(self, pred, target):
        return self.criterion(pred, target)

class SSIMLoss(nn.Module):
    """SSIM损失"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return 1 - self.ssim(img1, img2, window, self.window_size, channel, self.size_average)

class FusionGANLoss(nn.Module):
    """FusionGAN复合损失函数"""
    def __init__(self, lambda_adv=1.0, lambda_fm=10.0, lambda_l1=100.0, lambda_perceptual=0.0):
        super(FusionGANLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        
        self.adversarial_loss = AdversarialLoss('vanilla')
        self.feature_matching_loss = FeatureMatchingLoss()
        self.l1_loss = L1Loss()
        self.perceptual_loss = PerceptualLoss() if lambda_perceptual > 0 else None
    
    def forward(self, fake_imgs, real_imgs, fake_output, real_output, discriminator):
        """计算复合损失
        
        Args:
            fake_imgs: 生成的融合图像
            real_imgs: 真实图像（可见光图像）
            fake_output: 判别器对假图像的输出
            real_output: 判别器对真实图像的输出
            discriminator: 判别器模型
            
        Returns:
            总损失，以及各分量损失的字典
        """
        # 对抗损失
        adv_loss = self.adversarial_loss(fake_output, real_output, is_generator=True)
        
        # 特征匹配损失
        fm_loss = 0
        if self.lambda_fm > 0:
            # 获取判别器的中间层特征
            _, real_features = discriminator(real_imgs, return_features=True)
            _, fake_features = discriminator(fake_imgs, return_features=True)
            fm_loss = self.feature_matching_loss(real_features, fake_features)
        
        # L1损失
        l1_loss = self.l1_loss(fake_imgs, real_imgs)
        
        # 感知损失
        perceptual_loss = 0
        if self.perceptual_loss and self.lambda_perceptual > 0:
            perceptual_loss = self.perceptual_loss(real_imgs, fake_imgs)
        
        # 总损失
        total_loss = (
            self.lambda_adv * adv_loss +
            self.lambda_fm * fm_loss +
            self.lambda_l1 * l1_loss +
            self.lambda_perceptual * perceptual_loss
        )
        
        loss_dict = {
            'total_loss': total_loss,
            'adv_loss': adv_loss,
            'fm_loss': fm_loss,
            'l1_loss': l1_loss,
            'perceptual_loss': perceptual_loss
        }
        
        return total_loss, loss_dict

class DiscriminatorLoss(nn.Module):
    """判别器损失"""
    def __init__(self, loss_type='vanilla'):
        super(DiscriminatorLoss, self).__init__()
        self.adversarial_loss = AdversarialLoss(loss_type)
    
    def forward(self, fake_output, real_output):
        """计算判别器损失"""
        return self.adversarial_loss(fake_output, real_output, is_generator=False)

def test_losses():
    """测试损失函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 2
    channels = 1
    height, width = 256, 256
    
    fake_imgs = torch.randn(batch_size, channels, height, width).to(device)
    real_imgs = torch.randn(batch_size, channels, height, width).to(device)
    fake_output = torch.randn(batch_size, 1, 16, 16).to(device)
    real_output = torch.randn(batch_size, 1, 16, 16).to(device)
    
    # 测试对抗损失
    print("测试对抗损失...")
    adv_loss = AdversarialLoss('vanilla')
    g_loss = adv_loss(fake_output, real_output, is_generator=True)
    d_loss = adv_loss(fake_output, real_output, is_generator=False)
    print(f"生成器对抗损失: {g_loss.item():.4f}")
    print(f"判别器对抗损失: {d_loss.item():.4f}")
    
    # 测试特征匹配损失
    print("\n测试特征匹配损失...")
    # 模拟特征字典
    real_features = {
        'conv1_1': torch.randn(batch_size, 64, 128, 128).to(device),
        'conv2_1': torch.randn(batch_size, 128, 64, 64).to(device),
        'conv3_1': torch.randn(batch_size, 256, 32, 32).to(device)
    }
    fake_features = {
        'conv1_1': torch.randn(batch_size, 64, 128, 128).to(device),
        'conv2_1': torch.randn(batch_size, 128, 64, 64).to(device),
        'conv3_1': torch.randn(batch_size, 256, 32, 32).to(device)
    }
    
    fm_loss = FeatureMatchingLoss()
    fm_loss_value = fm_loss(real_features, fake_features)
    print(f"特征匹配损失: {fm_loss_value.item():.4f}")
    
    # 测试L1损失
    print("\n测试L1损失...")
    l1_loss = L1Loss()
    l1_loss_value = l1_loss(fake_imgs, real_imgs)
    print(f"L1损失: {l1_loss_value.item():.4f}")
    
    # 测试复合损失
    print("\n测试复合损失...")
    fusion_loss = FusionGANLoss(lambda_adv=1.0, lambda_fm=10.0, lambda_l1=100.0)
    
    # 模拟判别器
    class MockDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 64, 3, 1, 1)
        
        def forward(self, x, return_features=False):
            features = {}
            x = self.conv(x)
            if return_features:
                features['conv1'] = x
            return x, features
    
    discriminator = MockDiscriminator().to(device)
    total_loss, loss_dict = fusion_loss(fake_imgs, real_imgs, fake_output, real_output, discriminator)
    
    print(f"总损失: {total_loss.item():.4f}")
    for loss_name, loss_value in loss_dict.items():
        if loss_name != 'total_loss':
            print(f"{loss_name}: {loss_value.item():.4f}")
    
    print("\n损失函数测试完成！")

if __name__ == "__main__":
    test_losses()