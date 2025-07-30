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
    
    
class MicroChannelAttention(nn.Module):
    """
    Ultra-lightweight Squeeze-and-Excitation style channel attention.
    
    Architecture:
    - Global average pooling (squeeze)
    - Channel dimension reduction by factor of 8 (more aggressive than standard SE)
    - ReLU activation
    - Channel dimension expansion back to original
    - Sigmoid gating
    
    Parameters: ~ 62 for 12 input channels (much lighter than standard ChannelAttention)
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(2, channels // reduction)
        # Total parameters: channels * mid + mid + mid * channels + channels
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 0 parameters
            nn.Conv2d(channels, mid, 1),  # channels * mid + mid parameters
            nn.ReLU(),  # 0 parameters
            nn.Conv2d(mid, channels, 1),  # mid * channels + channels parameters
            nn.Sigmoid()  # 0 parameters
        )

    def forward(self, x):
        # Element-wise multiplication (0 parameters)
        return x * self.attention(x)  # attention module parameters used above


class LocalNetPlus(nn.Module):
    """
    Enhanced lightweight CNN with three minimal innovations over LocalNet.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Adaptive Feature Weighting: Learnable plaq/rect balance (2 parameters)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Micro Channel Attention: Ultra-lightweight SE-style attention (62 parameters)
    - Progressive Refinement: 1x1 conv with learnable mixing (156 parameters)
    - Conv2: 12 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Output: arctan normalization to [-1/4, 1/4] range
    
    Innovations:
    1. Adaptive plaq/rect fusion weights for optimal feature balance
    2. Micro channel attention for efficient feature recalibration
    3. Progressive refinement with learnable mixing coefficient
    
    Locality Properties:
    - Receptive field: 5x5 lattice sites (two 3x3 convolutions)
    - All innovations preserve spatial locality while adding global channel context
    
    Total parameters: ~ 2,220 (lightweight with targeted enhancements)
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
        
        # Innovation 1: Adaptive fusion weights (2 parameters)
        self.plaq_weight = nn.Parameter(torch.ones(1))  # 1 parameter
        self.rect_weight = nn.Parameter(torch.ones(1))  # 1 parameter
        
        # First conv layer to process combined features
        # Parameters: 6 * 12 * 3 * 3 + 12 = 660
        self.conv1 = nn.Conv2d(6, 12, 3, padding='same', padding_mode='circular')
        self.activation1 = nn.GELU()  # 0 parameters
        
        # Innovation 2: Micro channel attention (62 parameters)
        # For 12 channels with reduction=8: mid=2
        # Parameters: 12*2 + 2 + 2*12 + 12 = 62
        self.micro_attention = MicroChannelAttention(12, reduction=8)
        
        # Innovation 3: Progressive refinement (156 parameters)
        # Parameters: 12 * 12 * 1 * 1 + 12 = 156
        self.refine_conv = nn.Conv2d(12, 12, 1)  # 1x1 convolution for refinement
        self.refine_alpha = nn.Parameter(torch.tensor(0.1))  # 1 parameter - learnable mixing weight
        
        # Second conv layer to generate final outputs
        # Parameters: 12 * 12 * 3 * 3 + 12 = 1,308
        self.conv2 = nn.Conv2d(12, 12, 3, padding='same', padding_mode='circular')
        self.activation2 = nn.GELU()  # 0 parameters
        
    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Innovation 1: Adaptive feature fusion (2 parameters used)
        plaq_w = torch.sigmoid(self.plaq_weight)  # 1 parameter - learnable plaq weight
        rect_w = torch.sigmoid(self.rect_weight)  # 1 parameter - learnable rect weight
        x = torch.cat([plaq_features * plaq_w, rect_features * rect_w], dim=1)
        
        # First conv layer (660 parameters used)
        x = self.conv1(x)  # 660 parameters
        x = self.activation1(x)  # 0 parameters
        
        # Innovation 2: Micro channel attention (62 parameters used)
        x = self.micro_attention(x)  # 62 parameters - ultra-lightweight attention
        
        # Innovation 3: Progressive refinement (156 + 1 = 157 parameters used)
        refined = torch.sigmoid(self.refine_conv(x))  # 156 parameters - refinement features
        x = x * (1 + self.refine_alpha * refined)  # 1 parameter - learnable mixing coefficient
        
        # Second conv layer (1,308 parameters used)
        x = self.conv2(x)  # 1,308 parameters
        x = self.activation2(x)  # 0 parameters
        x = torch.arctan(x) / torch.pi / 2  # 0 parameters - range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients (0 parameters)
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
    """Pre-norm + SiLU + Conv + Pre-norm + SiLU + Conv + learnable scaling"""
    def __init__(self, channels, kernel_size=(3,3)):
        super().__init__()
        self.norm1 = nn.GroupNorm(2, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding='same', padding_mode='circular')
        self.norm2 = nn.GroupNorm(2, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding='same', padding_mode='circular')
        self.act = nn.SiLU()
        self.alpha = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)
        return identity + self.alpha * out

class LocalStableSimple(nn.Module):
    """
    CoordConv + 2×ResidualBlock + GN + GELU + tanh output scaling
    No global attention, fully local and periodic.
    """
    def __init__(self):
        super().__init__()
        cfg = NetConfig()
        cin = cfg.plaq_input_channels + cfg.rect_input_channels + 2  # +2 for coord channels
        hid = cfg.hidden_channels
        cout = cfg.plaq_output_channels + cfg.rect_output_channels

        # Input projection
        self.input_conv = nn.Conv2d(cin, hid, cfg.kernel_size, padding='same', padding_mode='circular')
        self.input_norm = nn.GroupNorm(2, hid)

        # Residual blocks
        self.res1 = ResidualBlock(hid, cfg.kernel_size)
        self.res2 = ResidualBlock(hid, cfg.kernel_size)

        # Output projection
        self.output_conv = nn.Conv2d(hid, cout, 1, bias=True)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.05)


    def forward(self, plaq_features, rect_features):
        B, _, H, W = plaq_features.shape
        device = plaq_features.device
        # CoordConv channels
        xs = torch.linspace(-1, 1, W, device=device).view(1,1,1,W).expand(B,1,H,W)
        ys = torch.linspace(-1, 1, H, device=device).view(1,1,H,1).expand(B,1,H,W)
        coords = torch.cat([xs, ys], dim=1)

        # Combine inputs
        x = torch.cat([plaq_features, rect_features, coords], dim=1)
        # Input proj
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.gelu(x)

        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)

        # Output proj + tanh scaling
        x = self.output_conv(x)
        x = torch.tanh(x) * self.output_scale

        # Split
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
        self.alpha = nn.Parameter(torch.tensor(0.3))  # learnable residual scaling

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
    elif model_tag == 'plus':
        return LocalNetPlus
    elif model_tag == 'lite':
        return LocalResAttnNetLite
    elif model_tag == 'rsat':
        return LocalResAttnNet
    elif model_tag == 'stable':
        return LocalStableNet
    elif model_tag == 'stable_simple':
        return LocalStableSimple
    else:
        raise ValueError(f"Invalid model tag: {model_tag}")