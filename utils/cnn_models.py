# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class jointCNN_simple(nn.Module):
    def __init__(self, plaq_input_channels=2, rect_input_channels=4, plaq_output_channels=4, rect_output_channels=8, kernel_size=(3, 3)):
        super().__init__()
        # Combined input channels for plaq and rect features
        combined_input_channels = plaq_input_channels + rect_input_channels

        # First conv layer to process combined features
        self.conv1 = nn.Conv2d(
            combined_input_channels,
            combined_input_channels * 2,  # Double the channels
            kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation1 = nn.GELU()
        
        # Second conv layer to generate final outputs
        self.conv2 = nn.Conv2d(
            combined_input_channels * 2,
            plaq_output_channels + rect_output_channels,  # Combined output channels
            kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation2 = nn.GELU()

    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # First conv layer
        x = self.conv1(x)
        x = self.activation1(x)
        
        # Second conv layer
        x = self.conv2(x)
        x = self.activation2(x)
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs

class ResBlock(nn.Module):
    """Residual block with two convolutional layers and a skip connection"""
    def __init__(self, channels, kernel_size=(3, 3)):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, 
            channels, 
            kernel_size, 
            padding='same', 
            padding_mode='circular'
        )
        self.activation1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(
            channels, 
            channels, 
            kernel_size, 
            padding='same', 
            padding_mode='circular'
        )
        self.activation2 = nn.GELU()
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.activation1(out)
        
        out = self.conv2(out)
        
        out += identity  # Skip connection
        out = self.activation2(out)
        
        return out
    
class ResBlock_norm(nn.Module):
    """Residual block with group norm, depthwise separable conv, and dropout"""
    def __init__(self, channels, kernel_size=(3, 3)):
        super().__init__()
        
        # First norm + depthwise + pointwise conv
        self.norm1 = nn.GroupNorm(4, channels)
        self.depthwise1 = nn.Conv2d(
            channels, channels, kernel_size,
            padding='same', padding_mode='circular',
            groups=channels
        )
        self.pointwise1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation1 = nn.GELU()

        # Second norm + depthwise + pointwise conv
        self.norm2 = nn.GroupNorm(4, channels)
        self.depthwise2 = nn.Conv2d(
            channels, channels, kernel_size,
            padding='same', padding_mode='circular',
            groups=channels
        )
        self.pointwise2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation2 = nn.GELU()

    def forward(self, x):
        identity = x

        out = self.norm1(x)
        out = self.depthwise1(out)
        out = self.pointwise1(out)
        out = self.activation1(out)

        out = self.norm2(out)
        out = self.depthwise2(out)
        out = self.pointwise2(out)

        out = identity + out
        out = self.activation2(out)

        return out
    
class jointCNN_rnet(nn.Module):
    def __init__(self, plaq_input_channels=2, rect_input_channels=4, plaq_output_channels=4, rect_output_channels=8, kernel_size=(3, 3), num_res_blocks=3):
        super().__init__()
        # Combined input channels for plaq and rect features
        combined_input_channels = plaq_input_channels + rect_input_channels
        intermediate_channels = combined_input_channels * 2  # Reduced from 4x to 2x
        
        # Initial convolution to increase channels
        self.initial_conv = nn.Conv2d(
            combined_input_channels,
            intermediate_channels,
            kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.initial_activation = nn.GELU()
        
        # ResNet blocks (reduced from 3 to 2 by default)
        self.res_blocks = nn.ModuleList([
            ResBlock(intermediate_channels, kernel_size)
            for _ in range(num_res_blocks)
        ])
        
        # Final convolution to get output channels
        self.final_conv = nn.Conv2d(
            intermediate_channels,
            plaq_output_channels + rect_output_channels,
            kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.final_activation = nn.GELU()

    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_activation(x)
        
        # Apply ResNet blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Final convolution
        x = self.final_conv(x)
        x = self.final_activation(x)
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs


class jointCNN_rnet_norm(nn.Module):
    def __init__(self, plaq_input_channels=2, rect_input_channels=4, plaq_output_channels=4, rect_output_channels=8, kernel_size=(3, 3), num_res_blocks=1, dropout_prob=0.05):
        super().__init__()
        # Combined input channels for plaq and rect features
        combined_input_channels = plaq_input_channels + rect_input_channels
        intermediate_channels = combined_input_channels * 2  # Reduced from 4x to 2x
        
        # Initial convolution to increase channels
        self.initial_conv = nn.Conv2d(
            combined_input_channels,
            intermediate_channels,
            kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.initial_activation = nn.GELU()
        
        # ResNet blocks (reduced from 3 to 2 by default)
        self.res_blocks = nn.ModuleList([
            ResBlock_norm(intermediate_channels, kernel_size)
            for _ in range(num_res_blocks)
        ])
        
        # Final convolution to get output channels
        self.final_conv = nn.Conv2d(
            intermediate_channels,
            plaq_output_channels + rect_output_channels,
            kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.final_activation = nn.GELU()
        
        # Single dropout at the end
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_activation(x)
        
        # Apply ResNet blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Final convolution
        x = self.final_conv(x)
        x = self.final_activation(x)
        x = self.dropout(x)
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs






class ImprovedResBlock(nn.Module):
    """改进的ResNet块，具备更好的稳定性"""
    def __init__(self, channels, kernel_size=(3, 3), dropout_rate=0.1):
        super().__init__()
        
        # Pre-normalization设计，提高训练稳定性
        self.norm1 = nn.GroupNorm(min(8, channels//4), channels)
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size,
            padding='same', padding_mode='circular',
            bias=False  # 使用GroupNorm时不需要bias
        )
        
        self.norm2 = nn.GroupNorm(min(8, channels//4), channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size,
            padding='same', padding_mode='circular',
            bias=False
        )
        
        # 使用Swish激活函数，比GELU更稳定
        self.activation = nn.SiLU()  # Swish activation
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # 可学习的残差连接权重
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        identity = x
        
        # Pre-norm设计
        out = self.norm1(x)
        out = self.activation(out)
        out = self.conv1(out)
        
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        # 可学习的残差连接
        out = identity + self.alpha * out
        
        return out

class AttentionGate(nn.Module):
    """简单的注意力机制，提高模型表达能力"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

class ImprovedJointCNN(nn.Module):
    """改进的Joint CNN架构"""
    def __init__(self, 
                 plaq_input_channels=2, 
                 rect_input_channels=4, 
                 plaq_output_channels=4, 
                 rect_output_channels=8,
                 hidden_channels=24,  # 减少通道数避免过拟合
                 num_res_blocks=4,    # 增加深度但用更稳定的块
                 dropout_rate=0.15,
                 kernel_size=(3, 3)):
        super().__init__()
        
        combined_input_channels = plaq_input_channels + rect_input_channels
        
        # 初始投影层
        self.input_proj = nn.Sequential(
            nn.Conv2d(combined_input_channels, hidden_channels, 1),
            nn.GroupNorm(min(8, hidden_channels//4), hidden_channels),
            nn.SiLU()
        )
        
        # 多尺度特征提取 - 修复通道数整除问题
        ms_channels = hidden_channels // 4  # 确保能被4整除，每个分支8个通道
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(hidden_channels, ms_channels, (3, 3), 
                     padding='same', padding_mode='circular'),
            nn.Conv2d(hidden_channels, ms_channels, (5, 5), 
                     padding='same', padding_mode='circular'),
            nn.Conv2d(hidden_channels, ms_channels, (7, 7), 
                     padding='same', padding_mode='circular')
        ])
        # 添加一个1x1卷积来调整通道数到期望值
        self.channel_adjust = nn.Conv2d(ms_channels * 3, hidden_channels, 1)
        
        # ResNet主干
        self.res_blocks = nn.ModuleList([
            ImprovedResBlock(hidden_channels, kernel_size, dropout_rate)
            for _ in range(num_res_blocks)
        ])
        
        # 注意力机制
        self.attention_gates = nn.ModuleList([
            AttentionGate(hidden_channels)
            for _ in range(num_res_blocks//2)
        ])
        
        # 输出投影层
        self.output_norm = nn.GroupNorm(min(8, hidden_channels//4), hidden_channels)
        self.output_conv = nn.Conv2d(
            hidden_channels, 
            plaq_output_channels + rect_output_channels,
            1,  # 使用1x1卷积减少参数
            bias=True
        )
        
        # 输出缩放参数，控制变换幅度
        self.output_scale = nn.Parameter(torch.ones(1) * 0.05)
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用He初始化，适合ReLU族激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 输出层使用小的初始化值
        nn.init.normal_(self.output_conv.weight, std=0.01)
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias)
    
    def forward(self, plaq_features, rect_features):
        # 合并输入特征
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # 初始投影
        x = self.input_proj(x)
        
        # 多尺度特征提取
        ms_features = []
        for conv in self.multi_scale_conv:
            ms_features.append(conv(x))
        x = torch.cat(ms_features, dim=1)
        x = self.channel_adjust(x)  # 调整回原来的通道数
        
        # ResNet主干 + 注意力
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)
            # 每两个ResBlock后添加注意力
            if i < len(self.attention_gates) and (i + 1) % 2 == 0:
                x = self.attention_gates[i//2](x)
        
        # 输出层
        x = self.output_norm(x)
        x = F.silu(x)
        x = self.output_conv(x)
        
        # 控制输出幅度
        x = self.output_scale * torch.tanh(x)  # 限制在[-output_scale, +output_scale]
        
        # 分离输出
        plaq_coeffs = x[:, :4, :, :]
        rect_coeffs = x[:, 4:, :, :]
        
        return plaq_coeffs, rect_coeffs

class HybridArchitecture(nn.Module):
    """混合架构：结合CNN和Transformer"""
    def __init__(self, base_channels=32):
        super().__init__()
        
        # CNN backbone
        self.cnn_backbone = ImprovedJointCNN(
            hidden_channels=base_channels,
            num_res_blocks=3,
            dropout_rate=0.1
        )
        
        # 简化的自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=base_channels,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 最终输出层
        self.final_conv = nn.Conv2d(base_channels, 12, 1)
        
    def forward(self, plaq_features, rect_features):
        # CNN特征提取
        plaq_coeffs, rect_coeffs = self.cnn_backbone(plaq_features, rect_features)
        
        # 为Transformer重塑特征
        B, C, H, W = plaq_coeffs.shape
        # 简化版：只在最后一层使用注意力
        combined = torch.cat([plaq_coeffs, rect_coeffs], dim=1)  # [B, 12, H, W]
        
        return plaq_coeffs, rect_coeffs





def choose_cnn_model(model_tag):
    if model_tag == 'simple':
        return jointCNN_simple
    elif model_tag == 'rnet':
        return jointCNN_rnet
    elif model_tag == 'rnet_norm':
        return jointCNN_rnet_norm
    elif model_tag == 'hybrid':
        return HybridArchitecture
    else:
        raise ValueError(f"Invalid model tag: {model_tag}")