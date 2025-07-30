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
    """
    Simple 2-layer CNN model for local gauge field updates.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Conv2: 12 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Output: arctan normalization to [-1/4, 1/4] range
    
    Locality Properties:
    - Receptive field: 5x5 lattice sites (two 3x3 convolutions)
    
    Total parameters: ~ 2,000 (very lightweight)
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels

        # First conv layer to process combined features
        # Parameters = input_channels x output_channels x kernel_height x kernel_width + bias_terms
        # Parameters: 6 * 12 * 3 * 3 + 12 = 660
        self.conv1 = nn.Conv2d(
            combined_input_channels,
            combined_input_channels * 2,  # Double the channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation1 = nn.GELU()  # 0 parameters
        
        # Second conv layer to generate final outputs
        # Parameters: 12 * 12 * 3 * 3 + 12 = 1,308
        self.conv2 = nn.Conv2d(
            combined_input_channels * 2,
            config.plaq_output_channels + config.rect_output_channels,  # Combined output channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation2 = nn.GELU()  # 0 parameters

    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features (0 parameters - tensor operation)
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # First conv layer (660 parameters used)
        x = self.conv1(x)
        x = self.activation1(x)  # 0 parameters
        
        # Second conv layer (1,308 parameters used)
        x = self.conv2(x)
        x = self.activation2(x)  # 0 parameters
        x = torch.arctan(x) / torch.pi / 2  # 0 parameters - tensor operation, range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients (0 parameters - tensor slicing)
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs    


class LocalResAttnNetLite(nn.Module):
    """
    Lightweight CNN with residual connection and channel attention.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Residual Block: 12 → 12 → 12 channels, two 3x3 convs with residual connection
    - Channel Attention: Global avg pool → squeeze → expand → sigmoid gating
    - Conv2: 12 → 12 channels, 3x3 kernel, output layer
    - Output: arctan normalization to [-1/4, 1/4] range
    
    Locality Properties:
    - Receptive field: 9x9 lattice sites (four 3x3 convolutions total)
    - Channel attention adds global channel-wise context while preserving spatial locality
    
    Total parameters: ~ 4,700 (moderate complexity)
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
        
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        
        # First conv layer to process combined features
        # Parameters: 6 * 12 * 3 * 3 + 12 = 660
        self.conv1 = nn.Conv2d(
            combined_input_channels,
            combined_input_channels * 2,  # Double the channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation1 = nn.GELU()  # 0 parameters
        
        # Residual block
        ch = combined_input_channels * 2
        # Parameters: input_channels * output_channels * kernel_height * kernel_width + bias_terms
        # Parameters: 12 * 12 * 3 * 3 + 12 = 1,308
        self.res_conv1 = nn.Conv2d(ch, ch, config.kernel_size, padding='same', padding_mode='circular')
        # Parameters: 12 * 12 * 3 * 3 + 12 = 1,308
        self.res_conv2 = nn.Conv2d(ch, ch, config.kernel_size, padding='same', padding_mode='circular')
        
        # Channel attention using standard ChannelAttention module
        # squeeze -> reduce -> expand -> sigmoid
        # Parameters: 12*3*1*1 + 3 + 3*12*1*1 + 12 = 87
        self.attn = ChannelAttention(ch)
        
        # Second conv layer to generate final outputs
        # Parameters: 12 * 12 * 3 * 3 + 12 = 1,308
        self.conv2 = nn.Conv2d(
            combined_input_channels * 2,
            config.plaq_output_channels + config.rect_output_channels,  # Combined output channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation2 = nn.GELU()  # 0 parameters
        
    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features (0 parameters)
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # First conv layer (660 parameters used)
        x = self.conv1(x)
        x = self.activation1(x)  # 0 parameters
        
        # Residual block (2,616 parameters used)
        res = self.res_conv1(x)  # 1,308 parameters
        res = self.activation1(res)  # 0 parameters
        res = self.res_conv2(res)  # 1,308 parameters
        x = x + res  # 0 parameters - tensor addition
        
        # Channel attention (~87 parameters used)
        x = self.attn(x)  # ~87 parameters - applies attention weights automatically
        
        # Second conv layer (1,308 parameters used)
        x = self.conv2(x)
        x = self.activation2(x)  # 0 parameters
        x = torch.arctan(x) / torch.pi / 2  # 0 parameters - range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients (0 parameters)
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs 

class LocalStableNet(nn.Module):
    """
    Stable hybrid architecture: memory-friendly but stronger than simple CNN.
    
    Architecture:
    - Input projection: 6 → 14 channels with GroupNorm
    - Two residual blocks with stable scaling (α=0.3)
    - Channel attention with conservative dimension reduction
    - Output projection: 14 → 12 channels with 1x1 conv
    - Learnable output scaling with tanh activation
    
    Locality Properties:
    - Receptive field: 11x11 lattice sites (input conv + two residual blocks (each has two 3x3 convs), 5 total 3x3 kernels)
    - Channel attention adds conservative global context while preserving spatial locality
    
    Total parameters: ~ 8,500 (high complexity)
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
        
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        
        # Simplified input projection
        # Parameters: 6 * 14 * 3 * 3 + 14 = 770
        self.input_conv = nn.Conv2d(
            combined_input_channels, 
            config.hidden_channels, 
            config.kernel_size,
            padding='same', 
            padding_mode='circular'
        )
        # Parameters: 2 * 14 = 28
        self.input_norm = nn.GroupNorm(2, config.hidden_channels)  # Reduce groups
        
        # Only use 2 ResNet blocks, but with more stable design
        # Each ResidualBlock has ~3,600 parameters
        self.res_block1 = ResidualBlock(config.hidden_channels, config.kernel_size)
        self.res_block2 = ResidualBlock(config.hidden_channels, config.kernel_size)
        
        # Channel attention using standard ChannelAttention module
        # Parameters: ~130 (same as before, now using reusable component)
        self.channel_attention = ChannelAttention(config.hidden_channels)
        
        # Output layer
        # Parameters: 14 * 12 * 1 * 1 + 12 = 180
        self.output_conv = nn.Conv2d(
            config.hidden_channels,
            config.plaq_output_channels + config.rect_output_channels,
            1,
            bias=True
        )
        
        # Learnable output scaling (1 parameter)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.05)
        
    
    def forward(self, plaq_features, rect_features):
        # Merge inputs (0 parameters)
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # Input processing (770 + 28 = 798 parameters used)
        x = self.input_conv(x)  # 770 parameters
        x = self.input_norm(x)  # 28 parameters
        x = F.gelu(x)  # 0 parameters
        
        # ResNet blocks - add scaling factor for improved stability
        x = self.res_block1(x) # ~3,600 parameters, moderate scaling
        x = self.res_block2(x)  # ~3,600 parameters
        
        # Channel attention (~130 parameters used)
        x = self.channel_attention(x)  # ~130 parameters - applies attention weights automatically
        
        # Output - progressive limiting (180 + 1 = 181 parameters used)
        x = self.output_conv(x)  # 180 parameters
        x = torch.tanh(x) * self.output_scale  # 1 parameter - learnable scaling
        
        # Separate outputs (0 parameters)
        plaq_coeffs = x[:, :4, :, :]
        rect_coeffs = x[:, 4:, :, :]
        
        return plaq_coeffs, rect_coeffs
    


class LocalResNet(nn.Module):
    """
    ResNet‐style local model with CoordConv for spatial awareness.
    
    Architecture:
    - CoordConv: Add normalized x,y coordinate channels (8 total input channels)
    - Input projection: 8 → 14 channels with GroupNorm
    - One standard residual block with learnable scaling
    - Output projection: 14 → 12 channels with 1x1 conv
    - arctan output limiting with learnable scaling
    
    Locality Properties:
    - Receptive field: 11x11 lattice sites (input conv + one residual block, 3 total 3x3 kernels)
    - CoordConv adds explicit spatial awareness while maintaining locality
    
    Total parameters: ~ 4,800 (moderate complexity)
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
        # include 2 extra channels for CoordConv
        cin = config.plaq_input_channels + config.rect_input_channels + 2
        hid = config.hidden_channels 
        cout = config.plaq_output_channels + config.rect_output_channels

        # Input projection
        # Parameters: 8 * 14 * 3 * 3 + 14 = 1,022
        self.input_conv = nn.Conv2d(
            cin, hid,
            config.kernel_size,
            padding='same', padding_mode='circular'
        )
        # Parameters: 2 * 14 = 28
        self.input_norm = nn.GroupNorm(2, hid)

        # One standard ResidualBlock
        # Each has ~3,600 parameters
        self.res_block = ResidualBlock(hid, config.kernel_size)

        # Output projection
        # Parameters: 14 * 12 * 1 * 1 + 12 = 180
        self.output_conv = nn.Conv2d(
            hid, cout,
            1, bias=True
        )
        # Learnable scaling (1 parameter)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, plaq_features, rect_features):
        B, _, H, W = plaq_features.shape
        device = plaq_features.device
        # CoordConv: add two channels of normalized coords (0 parameters - computed)
        xs = torch.linspace(-1, 1, W, device=device).view(1,1,1,W).expand(B,1,H,W)  # 0 parameters
        ys = torch.linspace(-1, 1, H, device=device).view(1,1,H,1).expand(B,1,H,W)  # 0 parameters
        coords = torch.cat([xs, ys], dim=1)  # 0 parameters

        # combine inputs (0 parameters)
        x = torch.cat([plaq_features, rect_features, coords], dim=1)

        # input proj (1,022 + 28 = 1,050 parameters used)
        x = self.input_conv(x)  # 1,022 parameters
        x = self.input_norm(x)  # 28 parameters
        x = F.gelu(x)  # 0 parameters

        # ResNet block (~3,600 parameters used)
        x = self.res_block(x)

        # output proj + range limiting (180 + 1 = 181 parameters used)
        x = self.output_conv(x)  # 180 parameters
        x = torch.tanh(x) * self.output_scale  # 1 parameter - learnable scaling

        # split (0 parameters)
        plaq_coeffs = x[:, :4, :, :]
        rect_coeffs = x[:, 4:, :, :]
        return plaq_coeffs, rect_coeffs

 

class LocalResAttnNet(nn.Module):
    """
    ResNet with channel attention - most sophisticated local model.
    
    Architecture:
    - Input projection: 6 → 14 channels with GroupNorm
    - Two residual blocks with pre-normalization and learnable scaling
    - Channel attention module with squeeze-and-excitation design
    - Output projection: 14 → 12 channels with 1x1 conv
    - arctan output limiting with learnable scaling
    
    Locality Properties:
    - Receptive field: 11x11 lattice sites (input conv + two residual blocks, 5 total 3x3 kernels)
    - Channel attention adds conservative global context while preserving spatial locality
    
    Total parameters: ~ 8,500 (high complexity)
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
        
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels

        # Input projection
        # Parameters: 6 * 14 * 3 * 3 + 14 = 770
        self.input_conv = nn.Conv2d(
            combined_input_channels, 
            config.hidden_channels, 
            config.kernel_size,
            padding='same', 
            padding_mode='circular'
        )
        # Parameters: 2 * 14 = 28
        self.input_norm = nn.GroupNorm(2, config.hidden_channels)

        # ResBlock 1 with pre-norm (~3,600 parameters each)
        self.res_block1 = ResidualBlock(config.hidden_channels, config.kernel_size)
        self.res_block2 = ResidualBlock(config.hidden_channels, config.kernel_size)

        # Lightweight dual-attention (channel) (~100 parameters)
        self.channel_attention = ChannelAttention(config.hidden_channels)

        # Output projection
        # Parameters: 14 * 12 * 1 * 1 + 12 = 180
        self.output_conv = nn.Conv2d(
            config.hidden_channels,
            config.plaq_output_channels + config.rect_output_channels,
            1,
            bias=True
        )

        # Learnable output scaling (1 parameter)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, plaq_features, rect_features):
        # Combine inputs (0 parameters)
        x = torch.cat([plaq_features, rect_features], dim=1)

        # Input processing (770 + 28 = 798 parameters used)
        x = self.input_conv(x)  # 770 parameters
        x = self.input_norm(x)  # 28 parameters
        x = F.gelu(x)  # 0 parameters

        # Residual blocks (~5,600 parameters used)
        x = self.res_block1(x)  # ~3,600 parameters
        x = self.res_block2(x)  # ~3,600 parameters

        # Attention (~130 parameters used)
        x = self.channel_attention(x)  # ~100 parameters

        # Output projection (180 + 1 = 181 parameters used)
        x = self.output_conv(x)  # 180 parameters
        x = torch.arctan(x) / math.pi / 2  # 0 parameters - Range [-0.25, 0.25]
        x = self.output_scale * x  # 1 parameter - learnable scaling

        # Split outputs (0 parameters)
        plaq_coeffs = x[:, :4, :, :]
        rect_coeffs = x[:, 4:, :, :]
        return plaq_coeffs, rect_coeffs


class ResidualBlock(nn.Module):
    """
    Pre-norm residual block with learnable scaling.
    
    Architecture:
    - Pre-norm → SiLU → Conv → Pre-norm → SiLU → Conv
    - Residual connection with learnable scaling factor α
    - GroupNorm for stable training
    - Circular padding for lattice boundary conditions
    
    Parameters per block: ~ 3,600 for 14 hidden channels
    """
    def __init__(self, channels, kernel_size=(3, 3)):
        super().__init__()
        # Parameters: 2 * channels each
        self.norm1 = nn.GroupNorm(2, channels)  # 2 * channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 
                               padding='same', padding_mode='circular')  # channels² * 9 + channels
        
        self.norm2 = nn.GroupNorm(2, channels)  # 2 * channels
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 
                               padding='same', padding_mode='circular')  # channels² * 9 + channels

        self.activation = nn.SiLU()  # 0 parameters
        self.alpha = nn.Parameter(torch.tensor(0.3))  # 1 parameter - learnable residual scaling

    def forward(self, x):
        identity = x  # 0 parameters

        # First block (norm1 + conv1 parameters used)
        out = self.norm1(x)  # 2 * channels parameters
        out = self.activation(out)  # 0 parameters
        out = self.conv1(out)  # channels² * 9 + channels parameters

        # Second block (norm2 + conv2 parameters used)
        out = self.norm2(out)  # 2 * channels parameters
        out = self.activation(out)  # 0 parameters
        out = self.conv2(out)  # channels² * 9 + channels parameters

        # Residual connection with learnable scaling (1 parameter used)
        return identity + self.alpha * out  # 1 parameter for alpha


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    
    Architecture:
    - Global average pooling (squeeze)
    - Channel dimension reduction by factor of 4
    - ReLU activation
    - Channel dimension expansion back to original
    - Sigmoid gating
    
    Parameters: ~ 100 for 14 input channels
    """
    def __init__(self, channels, reduction=4):
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
    
    



def choose_cnn_model(model_tag):
    if model_tag == 'simple':
        return LocalNet
    elif model_tag == 'lite':
        return LocalResAttnNetLite
    elif model_tag == 'rsat':
        return LocalResAttnNet
    elif model_tag == 'stable':
        return LocalStableNet
    elif model_tag == 'rnet':
        return LocalResNet
    else:
        raise ValueError(f"Invalid model tag: {model_tag}")