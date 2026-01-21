import torch
import torch.nn as nn
import torch.nn.functional as F

class ProgressiveExpansion(nn.Module):
    """
    Progressive Expansion Layer.
    Implements the series expansion: S_u = sum(c_n * x^p_n)
    For u=2, p=[1, 2]: S_2 = c1 * x + c2 * x^2
    """
    def __init__(self, channels, u=2):
        super(ProgressiveExpansion, self).__init__()
        self.u = u
        self.channels = channels
        # Learnable coefficients for each term i=1 to u. 
        # Using a separate coefficient for each channel? The paper says "c_n" are coefficients. 
        # Usually these are channel-wise to allow flexibility.
        # Shape: (u, channels, 1, 1) to support broadcasting over H,W
        self.coeffs = nn.Parameter(torch.Tensor(u, channels, 1, 1))
        nn.init.normal_(self.coeffs, mean=0.0, std=0.02) # Initialize small

    def forward(self, x):
        # x: (B, C, H, W)
        output = torch.zeros_like(x)
        for n in range(1, self.u + 1):
            # Term n: c_n * x^n
            # self.coeffs[n-1]: (C, 1, 1)
            term = self.coeffs[n-1] * torch.pow(x, n)
            output = output + term
        return output

class ConvPE(nn.Module):
    """
    Convolutional Progressive Expansion Layer.
    Structure: Conv2d(dilation) -> BN -> ReLU -> ProgressiveExpansion
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=2):
        super(ConvPE, self).__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2 # Same padding calculation
        # Adjust padding formula: 
        # K_eff = K + (K-1)(D-1)
        # Pad = (K_eff - 1) / 2
        # K=5, D=2 => K_eff = 5 + 4*1 = 9 => Pad = 4
        
        real_padding = (kernel_size - 1) * dilation // 2 # Dilation-based padding
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              padding=real_padding, dilation=dilation, bias=False) # Bias False for BN
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pe = ProgressiveExpansion(out_channels, u=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pe(x)
        return x

class MHSA(nn.Module):
    """
    Multi-Head Self-Attention (MHSA).
    Simplified implementation suitable for vision tasks (like ViT or specialized blocks).
    Here assuming standard attention on flattened spatial dimensions.
    """
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: (B, H*W, C) - Expecting flattened spatial
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, H, N, C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PENAttentionBlock(nn.Module):
    """
    PEN-Attention Block.
    """
    def __init__(self, in_channels, out_channels, patch_size=4):
        super(PENAttentionBlock, self).__init__()
        self.patch_size = patch_size
        
        # Two-branch ConvPE
        # Branch 1
        self.branch1 = ConvPE(in_channels, out_channels // 2)
        # Branch 2
        self.branch2 = ConvPE(in_channels, out_channels // 2)
        
        # MHSA
        self.mhsa = MHSA(dim=out_channels, num_heads=2)
        
        self.ln = nn.LayerNorm(out_channels)
        
        # Skip connection
        self.skip_conv = None
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
            
        # Parallel branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        
        # Concatenate
        x_p = torch.cat([b1, b2], dim=1) # (B, C, H, W)
        
        B, C, H, W = x_p.shape
        
        # --- MHSA with Patching/Downsampling ---
        # Downsample factor to reduce sequence length
        scale_factor = self.patch_size
        
        # Downsample features for Attention
        # Use simple Average Pooling acting as "Patching"
        if scale_factor > 1:
            x_down = F.avg_pool2d(x_p, kernel_size=scale_factor, stride=scale_factor)
        else:
            x_down = x_p
        
        # Flatten patched features
        x_flat = x_down.flatten(2).transpose(1, 2) 
        
        x_attn = self.mhsa(x_flat)
        
        # Layer Norm
        x_attn = self.ln(x_attn)
        
        # Reshape back to spatial (downsampled)
        if scale_factor > 1:
            H_new, W_new = H // scale_factor, W // scale_factor
            x_spatial = x_attn.transpose(1, 2).reshape(B, C, H_new, W_new)
            # Upsample back to original resolution
            x_out = F.interpolate(x_spatial, size=(H, W), mode='bilinear', align_corners=False)
        else:
            x_out = x_attn.transpose(1, 2).reshape(B, C, H, W)
        
        # Residual connection
        return x_out + residual

class MarsLSNet(nn.Module):
    def __init__(self, in_channels=7, num_classes=1, base_filters=48, patch_size=4):
        super(MarsLSNet, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # Stacked Blocks
        self.block1 = PENAttentionBlock(base_filters, base_filters, patch_size=patch_size)
        self.block2 = PENAttentionBlock(base_filters, base_filters, patch_size=patch_size)
        self.block3 = PENAttentionBlock(base_filters, base_filters, patch_size=patch_size)
        self.block4 = PENAttentionBlock(base_filters, base_filters, patch_size=patch_size)
        
        # Prediction Head
        self.head = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, 7, 128, 128)
        x = self.stem(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        logits = self.head(x)
        return logits
