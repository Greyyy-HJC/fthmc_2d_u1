# %%
import math
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




########################################################################################·
# Below are test models


class LightweightHybrid(nn.Module):
    """Lightweight hybrid architecture: memory-friendly version"""
    def __init__(self, 
                 plaq_input_channels=2, 
                 rect_input_channels=4, 
                 plaq_output_channels=4, 
                 rect_output_channels=8,
                 hidden_channels=16,  # Significantly reduce channel count
                 kernel_size=(3, 3)):
        super().__init__()
        
        combined_input_channels = plaq_input_channels + rect_input_channels
        
        # Simple input projection
        self.input_conv = nn.Conv2d(
            combined_input_channels, 
            hidden_channels, 
            kernel_size,
            padding='same', 
            padding_mode='circular'
        )
        self.input_norm = nn.GroupNorm(4, hidden_channels)  # Fixed 4 groups
        
        # Lightweight ResNet blocks - only use 2
        self.res_block1 = ResBlock(hidden_channels, kernel_size)
        self.res_block2 = ResBlock(hidden_channels, kernel_size)
        
        # Simple channel attention (replace complex MultiheadAttention)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(hidden_channels, hidden_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 4, hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # Output layer
        self.output_conv = nn.Conv2d(
            hidden_channels,
            plaq_output_channels + rect_output_channels,
            1,  # 1x1 convolution
            bias=True
        )
        
        # Initialize with small weights
        self._init_small_weights()
        
    def _init_small_weights(self):
        """Initialize small weights to produce near-identity transformation"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, plaq_features, rect_features):
        # Merge inputs
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # Input processing
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        # ResNet blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Channel attention
        attention = self.channel_attention(x)
        x = x * attention
        
        # Output
        x = self.output_conv(x)
        x = torch.arctan(x) / torch.pi / 2  # Limit output range
        
        # Separate outputs
        plaq_coeffs = x[:, :4, :, :]
        rect_coeffs = x[:, 4:, :, :]
        
        return plaq_coeffs, rect_coeffs

class StableHybrid(nn.Module):
    """Stable hybrid architecture: memory-friendly but stronger than simple"""
    def __init__(self, 
                 plaq_input_channels=2, 
                 rect_input_channels=4, 
                 plaq_output_channels=4, 
                 rect_output_channels=8,
                 hidden_channels=14,  # Reduce channels but still more than simple's 12
                 kernel_size=(3, 3)):
        super().__init__()
        
        combined_input_channels = plaq_input_channels + rect_input_channels
        
        # Simplified input projection
        self.input_conv = nn.Conv2d(
            combined_input_channels, 
            hidden_channels, 
            kernel_size,
            padding='same', 
            padding_mode='circular'
        )
        self.input_norm = nn.GroupNorm(2, hidden_channels)  # Reduce groups
        
        # Only use 2 ResNet blocks, but with more stable design
        self.res_block1 = ResBlock(hidden_channels, kernel_size)
        self.res_block2 = ResBlock(hidden_channels, kernel_size)
        
        # Simplified channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, max(2, hidden_channels // 4), 1),  # Conservative dimension reduction
            nn.ReLU(),
            nn.Conv2d(max(2, hidden_channels // 4), hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # Output layer
        self.output_conv = nn.Conv2d(
            hidden_channels,
            plaq_output_channels + rect_output_channels,
            1,
            bias=True
        )
        
        # Learnable output scaling
        self.output_scale = nn.Parameter(torch.ones(1) * 0.05)
        
        # Stable initialization
        self._init_stable_weights()
        
    def _init_stable_weights(self):
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
    
    
########################################################################################

class StableHybridV2(nn.Module):
    def __init__(self, 
                 plaq_input_channels=2, 
                 rect_input_channels=4, 
                 plaq_output_channels=4, 
                 rect_output_channels=8,
                 hidden_channels=14,
                 kernel_size=(3, 3)):
        super().__init__()

        combined_input_channels = plaq_input_channels + rect_input_channels

        # Input projection
        self.input_conv = nn.Conv2d(
            combined_input_channels, 
            hidden_channels, 
            kernel_size,
            padding='same', 
            padding_mode='circular'
        )
        self.input_norm = nn.GroupNorm(2, hidden_channels)

        # ResBlock 1 with pre-norm
        self.res_block1 = ResidualBlockV2(hidden_channels, kernel_size)
        self.res_block2 = ResidualBlockV2(hidden_channels, kernel_size)

        # Lightweight dual-attention (channel)
        self.channel_attention = ChannelAttention(hidden_channels)

        # Output projection
        self.output_conv = nn.Conv2d(
            hidden_channels,
            plaq_output_channels + rect_output_channels,
            1,
            bias=True
        )

        # Learnable output scaling
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
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


class ResidualBlockV2(nn.Module):
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

    

class StableHybridV3(nn.Module):
    def __init__(self, 
                 plaq_input_channels=2, 
                 rect_input_channels=4, 
                 plaq_output_channels=4, 
                 rect_output_channels=8,
                 hidden_channels=14,
                 kernel_size=(3, 3)):
        super().__init__()

        combined_input_channels = plaq_input_channels + rect_input_channels

        # Input projection
        self.input_conv = nn.Conv2d(
            combined_input_channels, 
            hidden_channels, 
            kernel_size,
            padding='same', 
            padding_mode='circular'
        )
        self.input_norm = nn.GroupNorm(2, hidden_channels)

        # Squeeze-expand bottleneck
        self.squeeze = nn.Conv2d(hidden_channels, hidden_channels // 2, 1)
        self.expand = nn.Conv2d(hidden_channels // 2, hidden_channels, 1)

        # Residual blocks
        self.res_block1 = ResidualBlockV3(hidden_channels, kernel_size)
        self.res_block2 = ResidualBlockV3(hidden_channels, kernel_size)

        # Channel attention
        self.channel_attention = ChannelAttention(hidden_channels)

        # Output projection
        self.output_conv = nn.Conv2d(
            hidden_channels,
            plaq_output_channels + rect_output_channels,
            1,
            bias=True
        )

        # Learnable output scaling (scalar)
        self.output_scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.output_conv.weight, std=0.01)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, plaq_features, rect_features):
        x = torch.cat([plaq_features, rect_features], dim=1)

        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.gelu(x)

        x = self.squeeze(x)
        x = F.gelu(x)
        x = self.expand(x)

        x = self.res_block1(x)
        x = self.res_block2(x)

        x = self.channel_attention(x)

        x = self.output_conv(x)
        x = torch.arctan(x) / math.pi / 2  # Range [-0.25, 0.25]
        x = self.output_scale * x

        plaq_coeffs = x[:, :4, :, :]
        rect_coeffs = x[:, 4:, :, :]
        return plaq_coeffs, rect_coeffs


class ResidualBlockV3(nn.Module):
    def __init__(self, channels, kernel_size=(3, 3)):
        super().__init__()
        self.norm1 = nn.GroupNorm(2, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 
                               padding='same', padding_mode='circular')

        self.norm2 = nn.GroupNorm(2, channels)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size, 
                                padding='same', padding_mode='circular', groups=channels)
        self.pwconv = nn.Conv2d(channels, channels, 1)

        self.activation = nn.SiLU()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        identity = x

        out = self.norm1(x)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.activation(out)
        out = self.dwconv(out)
        out = self.pwconv(out)

        return identity + self.alpha * out



class StableHybridV4(nn.Module):
    def __init__(self, 
                 plaq_input_channels=2, 
                 rect_input_channels=4, 
                 plaq_output_channels=4, 
                 rect_output_channels=8,
                 hidden_channels=14,
                 kernel_size=(3, 3)):
        super().__init__()

        combined_input_channels = plaq_input_channels + rect_input_channels

        self.input_conv = WSConv2d(
            combined_input_channels, 
            hidden_channels, 
            kernel_size,
            padding='same', 
            padding_mode='circular'
        )
        self.input_norm = nn.GroupNorm(2, hidden_channels)

        self.squeeze = nn.Conv2d(hidden_channels, hidden_channels // 2, 1)
        self.expand = nn.Conv2d(hidden_channels // 2, hidden_channels, 1)

        self.res_block1 = ResidualBlockV4(hidden_channels, kernel_size)
        self.res_block2 = ResidualBlockV4(hidden_channels, kernel_size)

        self.channel_attention = ChannelAttention(hidden_channels)

        self.output_conv = nn.Conv2d(
            hidden_channels,
            plaq_output_channels + rect_output_channels,
            1,
            bias=True
        )

        self.output_scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.output_conv.weight, std=0.01)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, plaq_features, rect_features):
        x = torch.cat([plaq_features, rect_features], dim=1)

        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.gelu(x)

        x = self.squeeze(x)
        x = F.gelu(x)
        x = self.expand(x)

        x = self.res_block1(x)
        x = self.res_block2(x)

        x = self.channel_attention(x)

        x = self.output_conv(x)
        x = torch.arctan(x) / math.pi / 2
        x = self.output_scale * x

        plaq_coeffs = x[:, :4, :, :]
        rect_coeffs = x[:, 4:, :, :]
        return plaq_coeffs, rect_coeffs


class ResidualBlockV4(nn.Module):
    def __init__(self, channels, kernel_size=(3, 3)):
        super().__init__()
        self.norm1 = nn.GroupNorm(2, channels)
        self.gated_conv = GatedConv(channels, channels, kernel_size)

        self.norm2 = nn.GroupNorm(2, channels)
        self.dwconv = nn.Conv2d(
            channels, channels, kernel_size,
            padding='same', padding_mode='circular', groups=channels
        )
        self.pwconv = nn.Conv2d(channels, channels, 1)

        self.alpha = nn.Parameter(torch.ones(channels) * 1e-3)

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.gated_conv(out)

        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dwconv(out)
        out = self.pwconv(out)

        return identity + self.alpha.view(1, -1, 1, 1) * out


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels * 2, kernel_size, padding='same', padding_mode='circular')

    def forward(self, x):
        x_proj = self.proj(x)
        x, gate = x_proj.chunk(2, dim=1)
        return x * torch.sigmoid(gate)


class WSConv2d(nn.Conv2d):
    def forward(self, x):
        weight = self.weight
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        std = weight.std(dim=(1, 2, 3), keepdim=True) + 1e-5
        weight = (weight - mean) / std
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


########################################################################################
# StableHybridV5 with modern improvements

class Mish(nn.Module):
    """Mish activation function: usually performs better than ReLU/GELU"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ModernResBlock(nn.Module):
    """Modern ResBlock inspired by ConvNeXt design principles"""
    def __init__(self, channels, kernel_size=(3, 3)):
        super().__init__()
        
        # Medium kernel depth-wise conv (memory friendly)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=5, 
                               padding=2, groups=channels, 
                               padding_mode='circular')
        
        # GroupNorm for stable training (better for 4D tensors)
        self.norm = nn.GroupNorm(2, channels)
        
        # Inverted bottleneck with 4x expansion
        self.pwconv1 = nn.Conv2d(channels, 4 * channels, 1)
        self.act = Mish()
        self.pwconv2 = nn.Conv2d(4 * channels, channels, 1)
        
        # Learnable scale parameter
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-6)
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        # Apply GroupNorm directly to (N,C,H,W)
        x = self.norm(x)
        
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        x = self.gamma.view(1, -1, 1, 1) * x
        
        return input + x


class ECAAttention(nn.Module):
    """ECA-Net: More efficient channel attention than SE-Net"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log(channels, 2) + b) / gamma))
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        
    def forward(self, x):
        y = x.mean((2, 3), keepdim=True)  # Global avg pool
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(y)


class StableHybridV5(nn.Module):
    """Modern hybrid architecture with latest improvements but no spatial attention"""
    def __init__(self, 
                 plaq_input_channels=2, 
                 rect_input_channels=4, 
                 plaq_output_channels=4, 
                 rect_output_channels=8,
                 hidden_channels=12,
                 kernel_size=(3, 3)):
        super().__init__()

        combined_input_channels = plaq_input_channels + rect_input_channels

        # Modern stem with larger receptive field
        self.stem = nn.Sequential(
            nn.Conv2d(combined_input_channels, hidden_channels, kernel_size, 
                     padding='same', padding_mode='circular'),
            nn.GroupNorm(2, hidden_channels),
            Mish()
        )

        # Modern ResBlocks
        self.blocks = nn.ModuleList([
            ModernResBlock(hidden_channels) for _ in range(2)
        ])

        # Advanced channel attention (no spatial attention as requested)
        self.channel_attention = ECAAttention(hidden_channels)

        # Output with better design
        self.output_conv = nn.Conv2d(hidden_channels, 
                                   plaq_output_channels + rect_output_channels, 
                                   1, bias=True)
        
        # Learnable temperature scaling
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        self._init_weights()

    def _init_weights(self):
        """Modern weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Special init for output layer
        nn.init.trunc_normal_(self.output_conv.weight, std=0.01)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, plaq_features, rect_features):
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        x = self.stem(x)
        
        # Apply modern ResBlocks
        for block in self.blocks:
            x = block(x)
        
        # Apply channel attention
        x = self.channel_attention(x)
        
        # Output processing
        x = self.output_conv(x)
        x = torch.arctan(x * self.temperature) / math.pi / 2
        
        plaq_coeffs = x[:, :4, :, :]
        rect_coeffs = x[:, 4:, :, :]
        return plaq_coeffs, rect_coeffs




class ConvMixerBlock(nn.Module):
    """ConvMixer block: depthwise conv + pointwise conv + residual + LayerScale"""
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels, channels, kernel_size,
            padding=kernel_size // 2, padding_mode='circular',
            groups=channels
        )
        self.norm1 = nn.GroupNorm(2, channels)
        self.pwconv = nn.Conv2d(channels, channels, 1)
        self.norm2 = nn.GroupNorm(2, channels)
        self.act = nn.GELU()
        # LayerScale: small per-channel residual scaling
        self.scale = nn.Parameter(torch.ones(channels) * 1e-6)

    def forward(self, x):
        y = self.dwconv(x)
        y = self.norm1(y)
        y = self.pwconv(y)
        y = self.norm2(y)
        y = self.act(y)
        return x + self.scale.view(1, -1, 1, 1) * y


class StableHybridV6(nn.Module):
    """
    StableHybridV6: ConvMixer-based hybrid CNN with ECA channel attention,
    4 sequential ConvMixer blocks, and lightweight design.
    """
    def __init__(
        self,
        plaq_input_channels=2,
        rect_input_channels=4,
        plaq_output_channels=4,
        rect_output_channels=8,
        hidden_channels=12,
        mixer_kernel_size=5,
        num_blocks=4
    ):
        super().__init__()
        in_ch = plaq_input_channels + rect_input_channels
        out_ch = plaq_output_channels + rect_output_channels

        # Stem: single conv + GroupNorm + GELU
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden_channels, mixer_kernel_size,
                      padding=mixer_kernel_size // 2, padding_mode='circular'),
            nn.GroupNorm(2, hidden_channels),
            nn.GELU()
        )

        # ConvMixer blocks
        self.blocks = nn.Sequential(*[
            ConvMixerBlock(hidden_channels, kernel_size=mixer_kernel_size)
            for _ in range(num_blocks)
        ])

        # Channel attention
        self.attn = ECAAttention(hidden_channels)

        # Output projection
        self.output_conv = nn.Conv2d(hidden_channels, out_ch, 1, bias=True)
        self.temperature = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # special init for output
        nn.init.trunc_normal_(self.output_conv.weight, std=0.01)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, plaq_features, rect_features):
        # plaq_features: [B,2,H,W], rect_features: [B,4,H,W]
        x = torch.cat([plaq_features, rect_features], dim=1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.attn(x)
        x = self.output_conv(x)
        # apply learnable temperature before arctan
        x = torch.arctan(x * self.temperature) / math.pi / 2
        # split outputs
        plaq_coeffs = x[:, :4, :, :]
        rect_coeffs = x[:, 4:, :, :]
        return plaq_coeffs, rect_coeffs






########################################################################################


def choose_cnn_model(model_tag):
    if model_tag == 'simple':
        return jointCNN_simple
    elif model_tag == 'rnet':
        return jointCNN_rnet
    elif model_tag == 'lightweight':
        return LightweightHybrid
    elif model_tag == 'stable':
        return StableHybrid
    elif model_tag == 'stablev2':
        return StableHybridV2
    elif model_tag == 'stablev3':
        return StableHybridV3
    elif model_tag == 'stablev4':
        return StableHybridV4
    elif model_tag == 'stablev5':
        return StableHybridV5
    elif model_tag == 'stablev6':
        return StableHybridV6
    else:
        raise ValueError(f"Invalid model tag: {model_tag}")