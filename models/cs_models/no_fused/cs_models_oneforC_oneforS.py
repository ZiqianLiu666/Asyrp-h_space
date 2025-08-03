import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List

class EqualizedLinear(nn.Module):
    """
    Equalized Learning-Rate Linear Layer with a single shared weight across all positions.
    """
    def __init__(self, in_features: int, out_features: int, bias: float = 0.0):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])  # 2D weight now
        self.bias = nn.Parameter(torch.full((out_features,), bias))

    def forward(self, x: torch.Tensor):
        # x: [B, N, in_features] — applies same weight to all N positions
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedWeight(nn.Module):
    """
    Equalized learning-rate scaling for 2D weights: [out_features, in_features]
    """
    def __init__(self, shape: List[int]):
        super().__init__()
        self.c = 1 / math.sqrt(shape[1])  # in_features
        self.weight = nn.Parameter(torch.randn(shape))  # [out_features, in_features]

    def forward(self):
        return self.weight * self.c




class MappingNetwork_cs_Unet(nn.Module):
    """
    Dual-output Mapping Network for UNet features using shared EqualizedLinear layers.
    Input: [B, 512, 8, 8]
    Output: c, s ∈ [B, 512, 8, 8]
    """
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        features, n_layers = opts.latent_dim, opts.n_cs_layers  # e.g., 512

        self.net_c = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])
        self.net_s = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])
        
        self.c_embed = nn.Sequential(
            nn.Conv2d(features, features // 8, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)    # 输出 (B, C, 1, 1)


        # # fusion layer
        # self.compos = CompositionalLayer(opts)
        
        # —— 新增：属性级二分类 head —— #
        # 取任意 [B,C,H,W] latent 做全局池化，输出单个 logit
        self.attr_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # [B,C,1,1]
            nn.Flatten(),                # [B,C]
            nn.Linear(features, 1)       # [B,1]
        )

    def _layer(self, features):
        return nn.Sequential(
            EqualizedLinear(features, features),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, z: torch.Tensor, eval_visul=False, infer=False):
        # z: [B, 512, 8, 8]
        B, C, H, W = z.shape
        z_flat = z.view(B, C, H*W).permute(0, 2, 1).contiguous()

        # Apply shared MLP to each spatial location
        c = self.net_c(z_flat)  # [B, 64, 512]
        s = self.net_s(z_flat)
        
        if eval_visul:
            # 把扁平后的 c 还原为 [B, C, H, W]
            c_map = c.permute(0, 2, 1).contiguous().view(B, C, H, W)
            return c_map
        
        # Reshape back to [B, 512, 8, 8]
        c = c.permute(0, 2, 1).view(B, C, H, W)
        s = s.permute(0, 2, 1).view(B, C, H, W)

        if infer:
            return c, s
        
        # ↓ DAO 投影后的 logits
        c_map    = c           # [B, C, H, W]
        proj_map = self.c_embed(c_map)                                # [B, proj_dim, H, W]
        g_c_logits = self.global_pool(proj_map).view(B, proj_map.size(1))  # [B, proj_dim]

        # —— 属性 logit —— #
        logit_c = self.attr_classifier(c)   # [B,1]
        logit_s = self.attr_classifier(s)   # [B,1]

        return c, s, g_c_logits, logit_c, logit_s
    
    
    


class ConvMappingNetwork_cs_Unet(nn.Module):
    """
    Dual-output Mapping Network using Conv2D layers.
    Input: [B, 512, 8, 8]
    Output: c, s ∈ [B, 512, 8, 8]
    """
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        channels = opts.latent_dim  # typically 512
        n_layers = opts.n_cs_layers

        # —— 为了复用，我们先定义一个 ConvBlock —— #
        def conv_block():
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Output heads
        self.net_c = nn.Sequential(*[conv_block() for _ in range(n_layers)])
        self.net_s = nn.Sequential(*[conv_block() for _ in range(n_layers)])
        
        # DAO 投影头：把 shared 特征从 features 降到 features//8
        self.c_embed = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
            
        # —— 新增：属性级二分类 head —— #
        # 取任意 [B,C,H,W] latent 做全局池化，输出单个 logit
        self.attr_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # [B,C,1,1]
            nn.Flatten(),                # [B,C]
            nn.Linear(channels, 1)       # [B,1]
        )

    def forward(self, z, eval_visul=False, infer=False):
        B, C, H, W = z.shape
        # Input: [B, 512, 8, 8]
        c = self.net_c(z)           # [B, 512, 8, 8]
        s = self.net_s(z)           # [B, 512, 8, 8]
        
        if eval_visul:
            return c
        
        if infer:
            return c, s
        
        # ↓ DAO 投影后的 logits，用于后面 KL loss
        proj_map    = self.c_embed(c)                                # [B,proj_dim,H,W]
        g_c_logits  = self.global_pool(proj_map).contiguous().view(B, proj_map.size(1))
        
        # —— 属性 logit —— #
        logit_c = self.attr_classifier(c)   # [B,1]
        logit_s = self.attr_classifier(s)   # [B,1]
        return c, s, g_c_logits, logit_c, logit_s