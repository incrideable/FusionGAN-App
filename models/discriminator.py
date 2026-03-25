import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorBlock(nn.Module):
    """判别器基础块：Conv -> BatchNorm -> LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True):
        super(DiscriminatorBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)
        return x

class PatchDiscriminator(nn.Module):
    """PatchGAN判别器：判断图像局部块的真实性"""
    def __init__(self, in_channels=1, features=64):
        super(PatchDiscriminator, self).__init__()
        
        # 判别器网络结构
        self.layer1 = DiscriminatorBlock(in_channels, features, use_bn=False)                    # 64
        self.layer2 = DiscriminatorBlock(features, features * 2)                                # 128
        self.layer3 = DiscriminatorBlock(features * 2, features * 4)                            # 256
        self.layer4 = DiscriminatorBlock(features * 4, features * 8)                            # 512
        self.layer5 = nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)          # 1
        
        # 用于特征匹配的损失计算
        self.feature_layers = [self.layer2, self.layer3, self.layer4]
    
    def forward(self, x):
        """前向传播
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            output: 判别结果 [B, 1, H/16, W/16]
            features: 中间层特征列表（用于特征匹配损失）
        """
        features = []
        
        # 逐层前向传播
        x = self.layer1(x)
        
        x = self.layer2(x)
        features.append(x)  # 保存中间特征
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        
        output = self.layer5(x)  # 最终判别结果
        
        return output, features
    
    def get_features(self, x):
        """获取中间层特征（用于特征匹配损失计算）"""
        features = []
        
        x = self.layer1(x)
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        
        return features

class MultiScaleDiscriminator(nn.Module):
    """多尺度判别器：在不同尺度上判断图像真实性"""
    def __init__(self, in_channels=1, features=64, num_scales=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_scales = num_scales
        
        # 创建多个判别器，每个对应不同尺度
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels, features) for _ in range(num_scales)
        ])
        
        # 下采样层用于生成不同尺度
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)
    
    def forward(self, x):
        """多尺度判别
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            outputs: 每个尺度的判别结果列表
            features: 每个尺度的特征列表
        """
        outputs = []
        features = []
        
        for i in range(self.num_scales):
            if i > 0:
                # 对输入进行下采样
                x = self.downsample(x)
            
            # 当前尺度的判别
            output, feat = self.discriminators[i](x)
            outputs.append(output)
            features.append(feat)
        
        return outputs, features

def test_discriminator():
    """测试判别器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建判别器
    discriminator = PatchDiscriminator(in_channels=1, features=64).to(device)
    
    # 创建测试输入
    batch_size = 1
    height, width = 256, 256
    input_img = torch.randn(batch_size, 1, height, width).to(device)  # 灰度图像
    
    # 前向传播
    with torch.no_grad():
        output, features = discriminator(input_img)
    
    print(f"输入图像形状: {input_img.shape}")
    print(f"输出判别结果形状: {output.shape}")
    print(f"特征层数量: {len(features)}")
    for i, feat in enumerate(features):
        print(f"特征层 {i+1} 形状: {feat.shape}")
    
    # 测试多尺度判别器
    print("\n--- 测试多尺度判别器 ---")
    multi_scale_d = MultiScaleDiscriminator(in_channels=1, features=64, num_scales=3).to(device)
    
    with torch.no_grad():
        outputs, multi_features = multi_scale_d(input_img)
    
    print(f"多尺度输出数量: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"尺度 {i+1} 输出形状: {out.shape}")
    
    return discriminator

def calculate_patchgan_loss(discriminator, real_imgs, fake_imgs):
    """计算PatchGAN损失
    Args:
        discriminator: 判别器模型
        real_imgs: 真实图像
        fake_imgs: 生成图像
    Returns:
        d_loss: 判别器损失
        g_loss: 生成器损失
    """
    # 判别器对真实图像的判断
    real_output, _ = discriminator(real_imgs)
    d_loss_real = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output))
    
    # 判别器对假图像的判断
    fake_output, _ = discriminator(fake_imgs.detach())
    d_loss_fake = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))
    
    # 判别器总损失
    d_loss = d_loss_real + d_loss_fake
    
    # 生成器损失（希望判别器认为假图像是真实的）
    g_loss = F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))
    
    return d_loss, g_loss

if __name__ == "__main__":
    test_discriminator()