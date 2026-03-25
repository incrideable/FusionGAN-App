import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """卷积块：Conv2d -> BatchNorm -> LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)
        return x

class DeconvBlock(nn.Module):
    """反卷积块：ConvTranspose2d -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True, use_dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.dropout = nn.Dropout(0.5) if use_dropout else None
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.deconv(x)
        if self.bn:
            x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class FusionGenerator(nn.Module):
    """FusionGAN生成器：U-Net架构，双流输入"""
    def __init__(self, in_channels=2, out_channels=1, features=64):
        super(FusionGenerator, self).__init__()
        
        # 编码器
        self.enc1 = ConvBlock(in_channels, features, use_bn=False)           # 64
        self.enc2 = ConvBlock(features, features * 2)                        # 128
        self.enc3 = ConvBlock(features * 2, features * 4)                    # 256
        self.enc4 = ConvBlock(features * 4, features * 8)                    # 512
        self.enc5 = ConvBlock(features * 8, features * 8)                    # 512
        self.enc6 = ConvBlock(features * 8, features * 8)                    # 512
        self.enc7 = ConvBlock(features * 8, features * 8)                    # 512
        self.enc8 = ConvBlock(features * 8, features * 8, use_bn=False)      # 512
        
        # 解码器
        self.dec7 = DeconvBlock(features * 8, features * 8, use_dropout=True)
        self.dec6 = DeconvBlock(features * 16, features * 8, use_dropout=True)
        self.dec5 = DeconvBlock(features * 16, features * 8, use_dropout=True)
        self.dec4 = DeconvBlock(features * 16, features * 4)
        self.dec3 = DeconvBlock(features * 8, features * 2)
        self.dec2 = DeconvBlock(features * 4, features)
        self.dec1 = DeconvBlock(features * 2, out_channels, use_bn=False)
        
        # 最终激活函数
        self.tanh = nn.Tanh()
    
    def forward(self, ir_img, vis_img):
        # 拼接红外和可见光图像
        x = torch.cat([ir_img, vis_img], dim=1)  # [B, 2, H, W]
        
        # 编码过程
        enc1 = self.enc1(x)      # 64
        enc2 = self.enc2(enc1)   # 128
        enc3 = self.enc3(enc2)   # 256
        enc4 = self.enc4(enc3)   # 512
        enc5 = self.enc5(enc4)   # 512
        enc6 = self.enc6(enc5)   # 512
        enc7 = self.enc7(enc6)   # 512
        enc8 = self.enc8(enc7)   # 512
        
        # 解码过程（带跳跃连接）
        dec7 = self.dec7(enc8)
        dec7 = torch.cat([dec7, enc7], dim=1)  # 跳跃连接
        
        dec6 = self.dec6(dec7)
        dec6 = torch.cat([dec6, enc6], dim=1)
        
        dec5 = self.dec5(dec6)
        dec5 = torch.cat([dec5, enc5], dim=1)
        
        dec4 = self.dec4(dec5)
        dec4 = torch.cat([dec4, enc4], dim=1)
        
        dec3 = self.dec3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        
        dec2 = self.dec2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        
        dec1 = self.dec1(dec2)
        
        # 使用tanh激活函数将输出归一化到[-1, 1]
        output = self.tanh(dec1)
        
        return output

class MSFModule(nn.Module):
    """多尺度特征融合模块（Multi-Scale Fusion Module）"""
    def __init__(self, channels):
        super(MSFModule, self).__init__()
        self.channels = channels
        
        # 自适应平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 1x1卷积用于特征变换
        self.conv_query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv_key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv_value = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Softmax用于注意力权重计算
        self.softmax = nn.Softmax(dim=-1)
        
        # 最后的1x1卷积
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 生成查询、键、值
        proj_query = self.conv_query(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, H*W, C//8]
        proj_key = self.conv_key(x).view(B, -1, H * W)  # [B, C//8, H*W]
        proj_value = self.conv_value(x).view(B, -1, H * W)  # [B, C, H*W]
        
        # 计算注意力权重
        energy = torch.bmm(proj_query, proj_key)  # [B, H*W, H*W]
        attention = self.softmax(energy)
        
        # 应用注意力
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, H*W]
        out = out.view(B, C, H, W)
        
        # 残差连接
        out = out + x
        
        # 最后的卷积
        out = self.conv_out(out)
        
        return out

def test_generator():
    """测试生成器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建生成器
    generator = FusionGenerator(in_channels=2, out_channels=1, features=64).to(device)
    
    # 创建测试输入
    batch_size = 1
    height, width = 256, 256
    ir_img = torch.randn(batch_size, 1, height, width).to(device)  # 红外图像
    vis_img = torch.randn(batch_size, 1, height, width).to(device)  # 可见光图像
    
    # 前向传播
    with torch.no_grad():
        fused_img = generator(ir_img, vis_img)
    
    print(f"输入红外图像形状: {ir_img.shape}")
    print(f"输入可见光图像形状: {vis_img.shape}")
    print(f"输出融合图像形状: {fused_img.shape}")
    
    # 检查输出范围
    print(f"输出范围: [{fused_img.min().item():.3f}, {fused_img.max().item():.3f}]")
    
    return generator

if __name__ == "__main__":
    test_generator()