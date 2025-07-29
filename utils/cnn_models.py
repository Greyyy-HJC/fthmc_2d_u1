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
    
    

class LocalNetv1(nn.Module): #! baseline
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


class LocalNetv2(nn.Module): #! add residual block and channel attention
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
        # self.conv1 = nn.utils.weight_norm(self.conv1)
        self.norm1 = nn.GroupNorm(2, combined_input_channels * 2)
        self.activation1 = nn.GELU()
        
        # Residual block
        ch = combined_input_channels * 2
        self.res_conv1 = nn.Conv2d(ch, ch, config.kernel_size, padding='same', padding_mode='circular')
        # self.res_norm1 = nn.GroupNorm(2, ch)
        self.res_conv2 = nn.Conv2d(ch, ch, config.kernel_size, padding='same', padding_mode='circular')
        # self.res_norm2 = nn.GroupNorm(2, ch)
        
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
        # self.conv2 = nn.utils.weight_norm(self.conv2)
        self.norm2 = nn.GroupNorm(2, config.plaq_output_channels + config.rect_output_channels)
        self.activation2 = nn.GELU()
        
        self.res_scale = nn.Parameter(torch.tensor(0.3))
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # First conv layer
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        
        # Residual block
        res = self.res_conv1(x)
        # res = self.res_norm1(res)
        res = self.activation1(res)
        res = self.res_conv2(res)
        # res = self.res_norm2(res)
        x = x + res * self.res_scale
        
        # Channel attention
        w = self.attn(x)
        x = x * w
        
        # Second conv layer
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        # x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        x = torch.tanh(x) * self.output_scale
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs 
    
    
class LocalNetv3(nn.Module): #! add weight_norm to conv1/conv2 and ReZero
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
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.activation1 = nn.GELU()
        
        # Second conv layer to generate final outputs
        self.conv2 = nn.Conv2d(
            combined_input_channels * 2,
            config.plaq_output_channels + config.rect_output_channels,  # Combined output channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.conv2 = nn.utils.weight_norm(self.conv2)
        self.activation2 = nn.GELU()
        
        # ReZero
        self.rezero_alpha = nn.Parameter(torch.zeros(1))
        

    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features
        x0 = torch.cat([plaq_features, rect_features], dim=1)
        
        # First conv layer
        x1 = self.conv1(x0)
        x1 = self.activation1(x1)
        
        # Second conv layer
        x2 = self.conv2(x1)
        x2 = self.activation2(x2)
        
        # ReZero
        x = x1 + self.rezero_alpha * x2
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs   
    


class LocalNetv4(nn.Module): #! CoordConv, helpful
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels + 2

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
        

    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features
        B, _, H, W = plaq_features.shape
        device = plaq_features.device
        
        xs = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        ys = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        coords = torch.cat([xs, ys], dim=1)
        
        x = torch.cat([plaq_features, rect_features, coords], dim=1)
        
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


class LocalNetv5(nn.Module): #! add residual block, helpful
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
        
        # Second conv layer
        x = self.conv2(x)
        x = self.activation2(x)
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs 
    

class LocalNetv6(nn.Module): #! add channel attention
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
        
        # Channel attention
        # squeeze -> reduce -> expand -> sigmoid
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(combined_input_channels * 2, max(2, combined_input_channels * 2 // 4), 1),
            nn.GELU(),
            nn.Conv2d(max(2, combined_input_channels * 2 // 4), combined_input_channels * 2, 1),
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
    
    
class LocalNetv7(nn.Module): #! dilation conv, useless
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
            dilation=2,
            padding=2,
            padding_mode='circular'
        )
        self.activation1 = nn.GELU()
        
        # Second conv layer to generate final outputs
        self.conv2 = nn.Conv2d(
            combined_input_channels * 2,
            config.plaq_output_channels + config.rect_output_channels,  # Combined output channels
            config.kernel_size,
            dilation=2,
            padding=2,
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



class LocalNetv8(nn.Module): #! add local self-attention, useless
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
        
        # Local self-attention
        self.attn = self._local_self_attention
        self.attn_window = 4
        
        # Second conv layer to generate final outputs
        self.conv2 = nn.Conv2d(
            combined_input_channels * 2,
            config.plaq_output_channels + config.rect_output_channels,  # Combined output channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation2 = nn.GELU()
        
    def _local_self_attention(self, x):
        B, C, H, W = x.shape
        ws = self.attn_window
        assert H % ws == 0 and W % ws == 0, "Height/width must be divisible by window size"

        # 1) unfold to patch of size ws×ws
        patches = F.unfold(x, kernel_size=ws, stride=ws)  # [B, C*ws*ws, nW]
        nW = patches.size(-1)

        # 2) reshape to [B, nW, M, C]
        patches = patches.view(B, C, ws*ws, nW).permute(0, 3, 2, 1)

        # 3) Q = K = V
        q = patches
        k = patches

        # 4) attention score, and softmax
        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(C), dim=-1)  # [B,nW,M,M]

        # 5) weighted V
        out = attn @ patches  # [B,nW,M,C]

        # 6) fold back to feature map
        out = out.permute(0, 3, 1, 2).reshape(B, C*ws*ws, nW)
        x = F.fold(out, output_size=(H, W), kernel_size=ws, stride=ws)
        return x
        
    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # First conv layer
        x = self.conv1(x)
        x = self.activation1(x)
        
        # Local self-attention
        x = self.attn(x)
        
        # Second conv layer
        x = self.conv2(x)
        x = self.activation2(x)
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs 
    

class LocalNetv9(nn.Module): #! add residual block and channel attention + weight_norm + CoordConv
    def __init__(self):
        super().__init__()
        config = NetConfig()
        
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels + 2
        
        # First conv layer to process combined features
        self.conv1 = nn.Conv2d(
            combined_input_channels,
            combined_input_channels * 2,  # Double the channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )   
        self.conv1 = nn.utils.weight_norm(self.conv1)
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
        self.conv2 = nn.utils.weight_norm(self.conv2)
        self.activation2 = nn.GELU()
        
    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # CoordConv
        B, _, H, W = plaq_features.shape
        device = plaq_features.device
        
        xs = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        ys = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        coords = torch.cat([xs, ys], dim=1)
        
        # Combine input features
        x = torch.cat([plaq_features, rect_features, coords], dim=1)
        
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
    
    
class LocalNetv10(nn.Module): #! add depthwise separable conv + pre-activation residual block on v9, useless
    def __init__(self):
        super().__init__()
        config = NetConfig()
        # add 2 channels from CoordConv
        cin = config.plaq_input_channels + config.rect_input_channels + 2
        cout = config.plaq_output_channels + config.rect_output_channels

        # === depthwise separable conv1 ===
        # Depthwise
        self.dw1 = nn.Conv2d(
            cin, cin, 
            kernel_size=config.kernel_size, 
            padding='same', padding_mode='circular',
            groups=cin
        )
        # Pointwise
        self.pw1 = nn.Conv2d(cin, cin*2, kernel_size=1)
        self.act1 = nn.GELU()
        # optional normalization (for stability)
        self.norm1 = nn.GroupNorm(2, cin*2)

        # === pre-activation residual block ===
        ch = cin*2
        self.pre_norm1 = nn.GroupNorm(2, ch)
        self.pre_conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding='same', padding_mode='circular')
        self.pre_norm2 = nn.GroupNorm(2, ch)
        self.pre_conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding='same', padding_mode='circular')

        # === Channel Attention ===
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(2, ch//4), 1),
            nn.GELU(),
            nn.Conv2d(max(2, ch//4), ch, 1),
            nn.Sigmoid()
        )

        # === depthwise separable conv2 ===
        self.dw2 = nn.Conv2d(
            ch, ch,
            kernel_size=config.kernel_size,
            padding='same', padding_mode='circular',
            groups=ch
        )
        self.pw2 = nn.Conv2d(ch, cout, kernel_size=1)
        self.act2 = nn.GELU()
        self.norm2 = nn.GroupNorm(2, cout)

    def forward(self, plaq, rect):
        B, _, H, W = plaq.shape
        device = plaq.device
        # CoordConv
        xs = torch.linspace(-1,1,W,device=device).view(1,1,1,W).expand(B,1,H,W)
        ys = torch.linspace(-1,1,H,device=device).view(1,1,H,1).expand(B,1,H,W)
        coords = torch.cat([xs, ys], dim=1)

        # combine
        x = torch.cat([plaq, rect, coords], dim=1)

        # depthwise separable conv1
        x = self.dw1(x)
        x = self.pw1(x)
        x = self.act1(x)
        x = self.norm1(x)

        # pre-activation residual block
        identity = x
        out = F.gelu(self.pre_norm1(x))
        out = self.pre_conv1(out)
        out = F.gelu(self.pre_norm2(out))
        out = self.pre_conv2(out)
        x = identity + out

        # channel attention
        w = self.attn(x)
        x = x * w

        # depthwise separable conv2
        x = self.dw2(x)
        x = self.pw2(x)
        x = self.act2(x)
        x = self.norm2(x)

        # project to [-1/4, 1/4]
        x = torch.arctan(x) / math.pi / 2

        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs 
    
    
class LocalNetv11(nn.Module): #! add residual block + weight_norm + CoordConv
    def __init__(self):
        super().__init__()
        config = NetConfig()
        
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels + 2
        
        # First conv layer to process combined features
        self.conv1 = nn.Conv2d(
            combined_input_channels,
            combined_input_channels * 2,  # Double the channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )   
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.activation1 = nn.GELU()
        
        # Residual block
        ch = combined_input_channels * 2
        self.res_conv1 = nn.Conv2d(ch, ch, config.kernel_size, padding='same', padding_mode='circular')
        self.res_conv2 = nn.Conv2d(ch, ch, config.kernel_size, padding='same', padding_mode='circular')
        
        # Second conv layer to generate final outputs
        self.conv2 = nn.Conv2d(
            combined_input_channels * 2,
            config.plaq_output_channels + config.rect_output_channels,  # Combined output channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.conv2 = nn.utils.weight_norm(self.conv2)
        self.activation2 = nn.GELU()
        
    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # CoordConv
        B, _, H, W = plaq_features.shape
        device = plaq_features.device
        
        xs = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        ys = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        coords = torch.cat([xs, ys], dim=1)
        
        # Combine input features
        x = torch.cat([plaq_features, rect_features, coords], dim=1)
        
        # First conv layer
        x = self.conv1(x)
        x = self.activation1(x)
        
        # Residual block
        res = self.res_conv1(x)
        res = self.activation1(res)
        res = self.res_conv2(res)
        x = x + res
        
        # Second conv layer
        x = self.conv2(x)
        x = self.activation2(x)
        
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs  
   
class LocalNetv12(nn.Module): #! add lightweight UNet, useless
    def __init__(self):
        super().__init__()
        config = NetConfig()
        cin = config.plaq_input_channels + config.rect_input_channels
        cout = config.plaq_output_channels + config.rect_output_channels
        hid = config.hidden_channels

        # 下采样路径
        self.enc1 = nn.Sequential(
            nn.Conv2d(cin, hid, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
        )
        self.pool1 = nn.MaxPool2d(2)  # 2x2 局部池化

        self.enc2 = nn.Sequential(
            nn.Conv2d(hid, hid*2, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(hid*2, hid*2, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        # 底部瓶颈
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hid*2, hid*4, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(hid*4, hid*2, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
        )

        # 上采样路径
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(hid*4, hid*2, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(hid*2, hid*2, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(hid*3, hid, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
        )

        # 输出层
        self.final = nn.Conv2d(hid, cout, 1)

    def forward(self, plaq, rect):
        # 拼接输入
        x = torch.cat([plaq, rect], dim=1)  # [B, cin, H, W]

        # Encoder
        e1 = self.enc1(x)     # [B, hid, H, W]
        p1 = self.pool1(e1)   # [B, hid, H/2, W/2]

        e2 = self.enc2(p1)    # [B, hid*2, H/2, W/2]
        p2 = self.pool2(e2)   # [B, hid*2, H/4, W/4]

        # Bottleneck
        b = self.bottleneck(p2)  # [B, hid*2, H/4, W/4]

        # Decoder
        u2 = self.up2(b)                        # [B, hid*2, H/2, W/2]
        # 跳跃连接：concat 底层特征
        d2 = torch.cat([u2, e2], dim=1)         # [B, hid*4, H/2, W/2]
        d2 = self.dec2(d2)                      # [B, hid*2, H/2, W/2]

        u1 = self.up1(d2)                       # [B, hid*2, H, W]
        d1 = torch.cat([u1, e1], dim=1)         # [B, hid*3, H, W]
        d1 = self.dec1(d1)                      # [B, hid, H, W]

        out = self.final(d1)                    # [B, cout, H, W]
        out = torch.arctan(out) / math.pi / 2   # 映射到 [-1/4,1/4]

        # Split output into plaq and rect coefficients
        plaq_coeffs = out[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = out[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs  
    

class LocalNetv13(nn.Module): #! add residual block + CoordConv + 3 conv layers
    def __init__(self):
        super().__init__()
        config = NetConfig()
        config.hidden_channels = 16
        config.kernel_size = (4, 4)
        # input channels + 2 CoordConv channels
        cin = config.plaq_input_channels + config.rect_input_channels + 2
        ch = cin * 2
        cout = config.plaq_output_channels + config.rect_output_channels

        # CoordConv is concatenated in forward

        # First conv layer
        self.conv1 = nn.Conv2d(cin, ch, config.kernel_size,
                      padding='same', padding_mode='circular')
        self.act1 = nn.GELU()

        # Residual block
        self.res_conv1 = nn.Conv2d(ch, ch, config.kernel_size,
                                   padding='same', padding_mode='circular')
        self.res_conv2 = nn.Conv2d(ch, ch, config.kernel_size,
                                   padding='same', padding_mode='circular')

        # Second conv layer (hidden)
        self.conv2 = nn.Conv2d(ch, ch, config.kernel_size,
                               padding='same', padding_mode='circular')
        self.act2 = nn.GELU()

        # Third conv layer (output)
        self.conv3 = nn.Conv2d(ch, cout, config.kernel_size,
                               padding='same', padding_mode='circular')
        self.act3 = nn.GELU()

    def forward(self, plaq_features, rect_features):
        B, _, H, W = plaq_features.shape
        device = plaq_features.device
        # CoordConv
        xs = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        ys = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        coords = torch.cat([xs, ys], dim=1)

        # Combine input features
        x = torch.cat([plaq_features, rect_features, coords], dim=1)

        # conv1
        x = self.act1(self.conv1(x))

        # ResBlock
        res = self.act1(self.res_conv1(x))
        res = self.res_conv2(res)
        x = x + res

        # conv2
        x = self.act2(self.conv2(x))

        # conv3
        x = self.act3(self.conv3(x))
        
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs 
    
    
class LocalNetv14(nn.Module): #! add residual block and channel attention + CoordConv + 3 conv layers
    def __init__(self):
        super().__init__()
        config = NetConfig()
        config.hidden_channels = 16
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels + 2
        
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
            combined_input_channels * 2,  # Combined output channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation2 = nn.GELU()
        
        # Third conv layer to generate final outputs
        self.conv3 = nn.Conv2d(
            combined_input_channels * 2,
            config.plaq_output_channels + config.rect_output_channels,  # Combined output channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation3 = nn.GELU()
        
    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # CoordConv
        B, _, H, W = plaq_features.shape
        device = plaq_features.device
        # CoordConv
        xs = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        ys = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        coords = torch.cat([xs, ys], dim=1)

        # Combine input features
        x = torch.cat([plaq_features, rect_features, coords], dim=1)
        
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
        
        # conv3
        x = self.conv3(x)
        x = self.activation3(x)
        
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
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
    

class LocalNetv15(nn.Module):  # CoordConv + two ResNet blocks + stable-style scaling
    def __init__(self):
        super().__init__()
        config = NetConfig()
        # Input channels + 2 for CoordConv
        cin = config.plaq_input_channels + config.rect_input_channels + 2
        hid = config.hidden_channels
        cout = config.plaq_output_channels + config.rect_output_channels

        # First conv projection
        self.conv1 = nn.Conv2d(
            cin, hid,
            config.kernel_size,
            padding='same', padding_mode='circular'
        )
        self.norm1 = nn.GroupNorm(2, hid)
        self.act1 = nn.GELU()

        # Two ResidualBlocks with fixed scaling
        self.res_block1 = ResidualBlock(hid, config.kernel_size)
        self.res_block2 = ResidualBlock(hid, config.kernel_size)

        # Output conv
        self.conv2 = nn.Conv2d(
            hid, cout,
            config.kernel_size,
            padding='same', padding_mode='circular'
        )
        self.act2 = nn.GELU()

        # Learnable output scaling
        self.output_scale = nn.Parameter(torch.ones(1) * 0.05)

    def forward(self, plaq_features, rect_features):
        B, _, H, W = plaq_features.shape
        device = plaq_features.device
        # CoordConv channels
        xs = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        ys = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        coords = torch.cat([xs, ys], dim=1)

        # Combine inputs
        x = torch.cat([plaq_features, rect_features, coords], dim=1)
        # Project & normalize
        x = self.act1(self.norm1(self.conv1(x)))

        # Two ResNet blocks with tunable scaling
        x = self.res_block1(x)
        x = self.res_block2(x)

        # Output projection
        x = self.act2(self.conv2(x))
        # Range limiting
        x = torch.tanh(x) * self.output_scale

        # Split output
        plaq_coeffs = x[:, :4, :, :]
        rect_coeffs = x[:, 4:, :, :]
        return plaq_coeffs, rect_coeffs   
    
    
def choose_cnn_model(model_tag):
    if model_tag == 'v1':
        return LocalNetv1
    elif model_tag == 'v2':
        return LocalNetv2
    elif model_tag == 'v3':
        return LocalNetv3
    elif model_tag == 'v4':
        return LocalNetv4
    elif model_tag == 'v5':
        return LocalNetv5
    elif model_tag == 'v6':
        return LocalNetv6
    elif model_tag == 'v7':
        return LocalNetv7
    elif model_tag == 'v8':
        return LocalNetv8
    elif model_tag == 'v9':
        return LocalNetv9
    elif model_tag == 'v10':
        return LocalNetv10
    elif model_tag == 'v11':
        return LocalNetv11
    elif model_tag == 'v12':
        return LocalNetv12
    elif model_tag == 'v13':
        return LocalNetv13
    elif model_tag == 'v14':
        return LocalNetv14
    elif model_tag == 'v15':
        return LocalNetv15
    else:
        raise ValueError(f"Invalid model tag: {model_tag}")