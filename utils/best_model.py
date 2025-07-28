import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass

@dataclass
class NetConfig:
    plaq_input_channels: int = 2
    rect_input_channels: int = 4
    plaq_output_channels: int = 4
    rect_output_channels: int = 8
    hidden_channels: int = 14
    kernel_size: tuple = (3, 3)
    
    
    
class LocalNet(nn.Module):
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels

        # First conv layer to process combined features
        self.conv1 = nn.Conv2d(
            combined_input_channels,
            combined_input_channels * 2,  # Double the channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation1 = nn.GELU()
        
        # Second conv layer to generate final outputs
        self.conv2 = nn.Conv2d(
            combined_input_channels * 2,
            config.plaq_output_channels + config.rect_output_channels,  # Combined output channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation2 = nn.GELU()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier normal for hidden layers and small normal for output layer"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Special initialization for output layer (conv2 acts as output)
        nn.init.normal_(self.conv2.weight, std=0.01)
        nn.init.zeros_(self.conv2.bias)

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


class LocalResAttnNetLite(nn.Module):
    def __init__(self):
        super().__init__()
        config = NetConfig()
        
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        
        # First conv layer to process combined features
        self.conv1 = nn.Conv2d(
            combined_input_channels,
            combined_input_channels * 2,  # Double the channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation1 = nn.GELU()
        
        # Residual block
        ch = combined_input_channels * 2
        self.res_conv1 = nn.Conv2d(ch, ch, config.kernel_size, padding='same', padding_mode='circular')
        self.res_conv2 = nn.Conv2d(ch, ch, config.kernel_size, padding='same', padding_mode='circular')
        
        # Channel attention
        # squeeze -> reduce -> expand -> sigmoid
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(2, ch // 4), 1),
            nn.GELU(),
            nn.Conv2d(max(2, ch // 4), ch, 1),
            nn.Sigmoid()
        )
        
        # Second conv layer to generate final outputs
        self.conv2 = nn.Conv2d(
            combined_input_channels * 2,
            config.plaq_output_channels + config.rect_output_channels,  # Combined output channels
            config.kernel_size,
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
        
        # Residual block
        res = self.res_conv1(x)
        res = self.activation1(res)
        res = self.res_conv2(res)
        x = x + res
        
        # Channel attention
        w = self.attn(x)
        x = x * w
        
        # Second conv layer
        x = self.conv2(x)
        x = self.activation2(x)
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs 


class LocalStableNet(nn.Module):
    """Stable hybrid architecture: memory-friendly but stronger than simple"""
    def __init__(self):
        super().__init__()
        config = NetConfig()
        
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        
        # Simplified input projection
        self.input_conv = nn.Conv2d(
            combined_input_channels, 
            config.hidden_channels, 
            config.kernel_size,
            padding='same', 
            padding_mode='circular'
        )
        self.input_norm = nn.GroupNorm(2, config.hidden_channels)  # Reduce groups
        
        # Only use 2 ResNet blocks, but with more stable design
        self.res_block1 = ResidualBlock(config.hidden_channels, config.kernel_size)
        self.res_block2 = ResidualBlock(config.hidden_channels, config.kernel_size)
        
        # Simplified channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(config.hidden_channels, max(2, config.hidden_channels // 4), 1),  # Conservative dimension reduction
            nn.ReLU(),
            nn.Conv2d(max(2, config.hidden_channels // 4), config.hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # Output layer
        self.output_conv = nn.Conv2d(
            config.hidden_channels,
            config.plaq_output_channels + config.rect_output_channels,
            1,
            bias=True
        )
        
        # Learnable output scaling
        self.output_scale = nn.Parameter(torch.ones(1) * 0.05)
        
        # Stable initialization
        self._init_weights()
        
    def _init_weights(self):
        """Stable weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.8)  # Moderate gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Small initialization for output layer
        nn.init.normal_(self.output_conv.weight, std=0.01)
        nn.init.zeros_(self.output_conv.bias)
    
    def forward(self, plaq_features, rect_features):
        # Merge inputs
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # Input processing
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        # ResNet blocks - add scaling factor for improved stability
        identity1 = x
        x = self.res_block1(x) * 0.3 + identity1  # Moderate scaling
        
        identity2 = x
        x = self.res_block2(x) * 0.3 + identity2
        
        # Channel attention
        attention = self.channel_attention(x)
        x = x * attention
        
        # Output - progressive limiting
        x = self.output_conv(x)
        x = torch.tanh(x) * self.output_scale
        
        # Separate outputs
        plaq_coeffs = x[:, :4, :, :]
        rect_coeffs = x[:, 4:, :, :]
        
        return plaq_coeffs, rect_coeffs
    




 

class LocalResAttnNet(nn.Module):
    def __init__(self):
        super().__init__()
        config = NetConfig()
        
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels

        # Input projection
        self.input_conv = nn.Conv2d(
            combined_input_channels, 
            config.hidden_channels, 
            config.kernel_size,
            padding='same', 
            padding_mode='circular'
        )
        self.input_norm = nn.GroupNorm(2, config.hidden_channels)

        # ResBlock 1 with pre-norm
        self.res_block1 = ResidualBlock(config.hidden_channels, config.kernel_size)
        self.res_block2 = ResidualBlock(config.hidden_channels, config.kernel_size)

        # Lightweight dual-attention (channel)
        self.channel_attention = ChannelAttention(config.hidden_channels)

        # Output projection
        self.output_conv = nn.Conv2d(
            config.hidden_channels,
            config.plaq_output_channels + config.rect_output_channels,
            1,
            bias=True
        )

        # Learnable output scaling
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier normal for conv layers, proper GroupNorm init, and small output layer"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Special initialization for output layer
        nn.init.normal_(self.output_conv.weight, std=0.01)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, plaq_features, rect_features):
        x = torch.cat([plaq_features, rect_features], dim=1)

        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.gelu(x)

        x = self.res_block1(x)
        x = self.res_block2(x)

        # Attention
        x = self.channel_attention(x)

        x = self.output_conv(x)
        x = torch.arctan(x) / math.pi / 2  # Range [-0.25, 0.25]
        x = self.output_scale * x

        plaq_coeffs = x[:, :4, :, :]
        rect_coeffs = x[:, 4:, :, :]
        return plaq_coeffs, rect_coeffs


class ResidualBlock(nn.Module):
    """Pre-norm + Dropout + Learnable scaling"""
    def __init__(self, channels, kernel_size=(3, 3)):
        super().__init__()
        self.norm1 = nn.GroupNorm(2, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 
                               padding='same', padding_mode='circular')
        
        self.norm2 = nn.GroupNorm(2, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 
                               padding='same', padding_mode='circular')

        self.activation = nn.SiLU()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable residual scaling

    def forward(self, x):
        identity = x

        out = self.norm1(x)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.activation(out)
        out = self.conv2(out)

        return identity + self.alpha * out


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(2, channels // reduction)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)
    
    
def choose_cnn_model(model_tag):
    if model_tag == 'simple':
        return LocalNet
    elif model_tag == 'lite':
        return LocalResAttnNetLite
    elif model_tag == 'rsat':
        return LocalResAttnNet
    elif model_tag == 'stable':
        return LocalStableNet
    else:
        raise ValueError(f"Invalid model tag: {model_tag}")