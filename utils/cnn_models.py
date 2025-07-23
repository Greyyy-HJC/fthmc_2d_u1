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






########################################################################################·
# Below are test models



class ImprovedResBlock(nn.Module):
    """Improved ResNet block with better stability"""
    def __init__(self, channels, kernel_size=(3, 3), dropout_rate=0.1):
        super().__init__()
        
        # Pre-normalization design for improved training stability
        self.norm1 = nn.GroupNorm(min(8, channels//4), channels)
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size,
            padding='same', padding_mode='circular',
            bias=False  # No bias needed when using GroupNorm
        )
        
        self.norm2 = nn.GroupNorm(min(8, channels//4), channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size,
            padding='same', padding_mode='circular',
            bias=False
        )
        
        # Use Swish activation function, more stable than GELU
        self.activation = nn.SiLU()  # Swish activation
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Learnable residual connection weight
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        identity = x
        
        # Pre-norm design
        out = self.norm1(x)
        out = self.activation(out)
        out = self.conv1(out)
        
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        # Learnable residual connection
        out = identity + self.alpha * out
        
        return out

class AttentionGate(nn.Module):
    """Simple attention mechanism to improve model expressiveness"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

class ImprovedJointCNN(nn.Module):
    """Improved Joint CNN architecture"""
    def __init__(self, 
                 plaq_input_channels=2, 
                 rect_input_channels=4, 
                 plaq_output_channels=4, 
                 rect_output_channels=8,
                 hidden_channels=24,  # Reduce channels to avoid overfitting
                 num_res_blocks=4,    # Increase depth with more stable blocks
                 dropout_rate=0.15,
                 kernel_size=(3, 3)):
        super().__init__()
        
        combined_input_channels = plaq_input_channels + rect_input_channels
        
        # Initial projection layer
        self.input_proj = nn.Sequential(
            nn.Conv2d(combined_input_channels, hidden_channels, 1),
            nn.GroupNorm(min(8, hidden_channels//4), hidden_channels),
            nn.SiLU()
        )
        
        # Multi-scale feature extraction - fix channel divisibility issue
        ms_channels = hidden_channels // 4  # Ensure divisible by 4, 8 channels per branch
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(hidden_channels, ms_channels, (3, 3), 
                     padding='same', padding_mode='circular'),
            nn.Conv2d(hidden_channels, ms_channels, (5, 5), 
                     padding='same', padding_mode='circular'),
            nn.Conv2d(hidden_channels, ms_channels, (7, 7), 
                     padding='same', padding_mode='circular')
        ])
        # Add 1x1 conv to adjust channels back to expected value
        self.channel_adjust = nn.Conv2d(ms_channels * 3, hidden_channels, 1)
        
        # ResNet backbone
        self.res_blocks = nn.ModuleList([
            ImprovedResBlock(hidden_channels, kernel_size, dropout_rate)
            for _ in range(num_res_blocks)
        ])
        
        # Attention mechanism
        self.attention_gates = nn.ModuleList([
            AttentionGate(hidden_channels)
            for _ in range(num_res_blocks//2)
        ])
        
        # Output projection layer
        self.output_norm = nn.GroupNorm(min(8, hidden_channels//4), hidden_channels)
        self.output_conv = nn.Conv2d(
            hidden_channels, 
            plaq_output_channels + rect_output_channels,
            1,  # Use 1x1 conv to reduce parameters
            bias=True
        )
        
        # Output scaling parameter to control transformation amplitude
        self.output_scale = nn.Parameter(torch.ones(1) * 0.05)
        
        # Weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Improved weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use He initialization, suitable for ReLU family activation functions
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Use small initialization values for output layer
        nn.init.normal_(self.output_conv.weight, std=0.01)
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias)
    
    def forward(self, plaq_features, rect_features):
        # Merge input features
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Multi-scale feature extraction
        ms_features = []
        for conv in self.multi_scale_conv:
            ms_features.append(conv(x))
        x = torch.cat(ms_features, dim=1)
        x = self.channel_adjust(x)  # Adjust back to original channel count
        
        # ResNet backbone + attention
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)
            # Add attention after every two ResBlocks
            if i < len(self.attention_gates) and (i + 1) % 2 == 0:
                x = self.attention_gates[i//2](x)
        
        # Output layer
        x = self.output_norm(x)
        x = F.silu(x)
        x = self.output_conv(x)
        
        # Control output amplitude
        x = self.output_scale * torch.tanh(x)  # Limit to [-output_scale, +output_scale]
        
        # Separate outputs
        plaq_coeffs = x[:, :4, :, :]
        rect_coeffs = x[:, 4:, :, :]
        
        return plaq_coeffs, rect_coeffs

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

class HybridArchitecture(nn.Module):
    """Hybrid architecture: combines CNN and Transformer (maintains compatibility, actually uses lightweight version)"""
    def __init__(self, base_channels=16):  # Default reduced channel count
        super().__init__()
        
        # Use lightweight version
        self.lightweight_model = LightweightHybrid(
            hidden_channels=base_channels
        )
        
    def forward(self, plaq_features, rect_features):
        return self.lightweight_model(plaq_features, rect_features)



def choose_cnn_model(model_tag):
    if model_tag == 'simple':
        return jointCNN_simple
    elif model_tag == 'rnet':
        return jointCNN_rnet
    elif model_tag == 'rnet_norm':
        return jointCNN_rnet_norm
    elif model_tag == 'hybrid':
        return HybridArchitecture
    elif model_tag == 'lightweight':
        return LightweightHybrid
    elif model_tag == 'stable':
        return StableHybrid
    else:
        raise ValueError(f"Invalid model tag: {model_tag}")