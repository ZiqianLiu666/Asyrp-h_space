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

class one_encoder_mlp(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        features, n_layers = opts.latent_dim, opts.n_cs_layers  # e.g., 512

        layers = []
        for i in range(n_layers - 1):
            layers += [ EqualizedLinear(features, features),
                        nn.LeakyReLU(0.2, inplace=True) ]

        layers += [ EqualizedLinear(features, 3 * features),
                    nn.LeakyReLU(0.2, inplace=True) ]
        self.encoder = nn.Sequential(*layers)

        self.c_embed = nn.Sequential(
            nn.Conv2d(features, features // 8, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)  
        
        self.attr_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # [B,C,1,1]
            nn.Flatten(),                # [B,C]
            nn.Linear(features, 1)       # [B,1]
        )
    
    def forward(self, z: torch.Tensor, eval_visul=False):
        # z: [B, 512, 8, 8]
        B, C, H, W = z.shape
        z_flat = z.contiguous().view(B, C, H * W).permute(0, 2, 1)  # [B, 64, 512]

        # Apply shared MLP to each spatial location
        out_flat = self.encoder(z_flat)  # [B, 64, 512]
        c, s_bg, s_t = torch.split(out_flat, C, dim=2)
        
        if eval_visul:
            c_map = c.permute(0, 2, 1).contiguous().view(B, C, H, W)
            return c_map
        
        c_map       = c.permute(0, 2, 1).contiguous().view(B, C, H, W)       # [B,features,H,W]
        proj_map    = self.c_embed(c_map)                                    # [B,proj_dim,H,W]
        g_c_logits  = self.global_pool(proj_map).contiguous().view(B, proj_map.size(1))  # [B,proj_dim]


        # Reshape back to [B, 512, 8, 8]
        c = c.permute(0, 2, 1).contiguous().view(B, C, H, W)
        s_bg = s_bg.permute(0, 2, 1).contiguous().view(B, C, H, W)
        s_t = s_t.permute(0, 2, 1).contiguous().view(B, C, H, W)

        logit_c = self.attr_classifier(c)   # [B,1]
        logit_s_bg = self.attr_classifier(s_bg)   # [B,1]
        logit_s_t = self.attr_classifier(s_t)   # [B,1]
        
        return c, s_t, s_bg, g_c_logits, logit_c, logit_s_bg, logit_s_t





class one_encoder_conv(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        channels = opts.latent_dim  # typically 512
        n_layers = opts.n_cs_layers
        
        layers = []
        for _ in range(n_layers - 1):
            layers += [
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [
            nn.Conv2d(channels, 3 * channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.encoder = nn.Sequential(*layers)

        self.c_embed = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.attr_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # [B,C,1,1]
            nn.Flatten(),                # [B,C]
            nn.Linear(channels, 1)       # [B,1]
        )
    
    def forward(self, z: torch.Tensor, eval_visul=False):
        B, C, H, W = z.shape
        # z: [B, 512, 8, 8]
        out = self.encoder(z)
        c, s_bg, s_t = torch.split(out, C, dim=1)

        if eval_visul:
            return c

        proj_map   = self.c_embed(c)            # [B, C//8, H, W]
        g_c_logits = self.global_pool(proj_map)     # [B, C//8, 1, 1]
        g_c_logits = g_c_logits.view(B, -1)         # [B, C//8]

        logit_c = self.attr_classifier(c)   # [B,1]
        logit_s_bg = self.attr_classifier(s_bg)   # [B,1]
        logit_s_t = self.attr_classifier(s_t)   # [B,1]
        
        return c, s_t, s_bg, g_c_logits, logit_c, logit_s_bg, logit_s_t