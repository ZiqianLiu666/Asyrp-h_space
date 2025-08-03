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




class specific_mlp_fused(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        features, n_layers = opts.latent_dim, opts.n_cs_layers  # e.g., 512
        
        self.shared_c = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])
        self.specific_t_s = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])
        self.specific_bg_s = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])

        self.c_embed = nn.Sequential(
            nn.Conv2d(features, features // 8, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)    # 输出 (B, C, 1, 1)


        # fusion layer
        self.compos = CompositionalLayer(opts)
        
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
    
    def forward(self, z: torch.Tensor, is_bg=False, no_fused=False, eval_visul=False):
        # z: [B, 512, 8, 8]
        B, C, H, W = z.shape
        z_flat = z.contiguous().view(B, C, H * W).permute(0, 2, 1)  # [B, 64, 512]

        # Apply shared MLP to each spatial location
        c = self.shared_c(z_flat)  # [B, 64, 512]
        if eval_visul:
            # 把扁平后的 c 还原为 [B, C, H, W]
            c_map = c.permute(0, 2, 1).contiguous().view(B, C, H, W)
            return c_map

        if is_bg:
            s = self.specific_bg_s(z_flat)
        else:
            s = self.specific_t_s(z_flat)
        
        if no_fused:
            c_map = c.permute(0, 2, 1).view(B, C, H, W)
            s_map = s.permute(0, 2, 1).view(B, C, H, W)
            return c_map, s_map
        
        # ↓ DAO 投影后的 logits
        c_map       = c.permute(0, 2, 1).contiguous().view(B, C, H, W)       # [B,features,H,W]
        proj_map    = self.c_embed(c_map)                                    # [B,proj_dim,H,W]
        g_c_logits  = self.global_pool(proj_map).contiguous().view(B, proj_map.size(1))  # [B,proj_dim]
                
        # 3) fuse shared + specific via residual-MLP
        fused_flat = self.compos(c, s)   # [B, HW, feats]

        # Reshape back to [B, 512, 8, 8]
        c = c.permute(0, 2, 1).contiguous().view(B, C, H, W)
        s = s.permute(0, 2, 1).contiguous().view(B, C, H, W)
        fused = fused_flat.permute(0, 2, 1).contiguous().view(B, C, H, W)
        
        # —— 属性 logit —— #
        logit_c = self.attr_classifier(c)   # [B,1]
        logit_s = self.attr_classifier(s)   # [B,1]
        return c, s, fused, g_c_logits, logit_c, logit_s
    
        
# 这里实现了 cs 融合
class CompositionalLayer(nn.Module):
    def __init__(self, opts, normalization_sign=True):
        super().__init__()
        self.normalization = normalization_sign
        self.opts = opts
        features, n_layers = opts.latent_dim, opts.n_cs_layers  # e.g., 512
        
        # 只保留一层：输入 2F，输出 F，再接 LeakyReLU
        self.blend_mlp = nn.Sequential(
            EqualizedLinear(2 * features, features),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, f1, f2):
        """
        :param f1: shared-modality fts
        :param f2: specific-modality fts
        :return:
        """
        # if self.normalization:
        #     f1 = F.normalize(f1, dim=-1)
        #     f2 = F.normalize(f2, dim=-1)
        residual = torch.cat((f1, f2), dim=-1) # [B, HW, 2F]

        #### compose two modalities by residual learning
        # f_cita proj
        residual = self.blend_mlp(residual)
        features = f1 + residual  # other modality + residual (default)

        return features
        
        
        
class specific_conv_fused(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        channels = opts.latent_dim  # typically 512
        n_layers = opts.n_cs_layers
        
        self.shared_c = self.make_layers(n_layers, channels)
        self.specific_t_s = self.make_layers(n_layers, channels)
        self.specific_bg_s = self.make_layers(n_layers, channels)
        
        # DAO 投影头：把 shared 特征从 features 降到 features//8
        self.c_embed = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # fusion layer
        self.compos = CompositionalLayer_conv(opts)
        
        # —— 新增：属性级二分类 head —— #
        # 取任意 [B,C,H,W] latent 做全局池化，输出单个 logit
        self.attr_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # [B,C,1,1]
            nn.Flatten(),                # [B,C]
            nn.Linear(channels, 1)       # [B,1]
        )
    
    def make_layers(self, n_layers, channels):
        layers = []
        for _ in range(n_layers):
            layers += [
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        return nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, is_bg=False, no_fused=False, eval_visul=False):
        B, C, H, W = z.shape
        # z: [B, 512, 8, 8]
        c = self.shared_c(z)

        if eval_visul:
            return c

        if is_bg:
            s = self.specific_bg_s(z)
        else:
            s = self.specific_t_s(z)
        
        if no_fused:
            return c, s
            
        # ↓ DAO 投影后的 logits，用于后面 KL loss
        proj_map    = self.c_embed(c)                                # [B,proj_dim,H,W]
        g_c_logits  = self.global_pool(proj_map).contiguous().view(B, proj_map.size(1))
        
        # 3) fuse shared + specific via residual-MLP
        fused = self.compos(c, s)
        
        # —— 属性 logit —— #
        logit_c = self.attr_classifier(c)   # [B,1]
        logit_s = self.attr_classifier(s)   # [B,1]
        return c, s, fused, g_c_logits, logit_c, logit_s
        
# 这里实现了 cs 融合
class CompositionalLayer_conv(nn.Module):

    def __init__(self, opts, normalization_sign=True):
        super().__init__()
        self.normalization = normalization_sign
        self.opts = opts
        channels = opts.latent_dim  # typically 512
        
        # 仅保留第一层：2F -> F，再接 LeakyReLU
        self.blend_conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, f1, f2):
        """
        :param f1: shared-modality fts
        :param f2: specific-modality fts
        :return:
        """
        # if self.normalization:
        #     f1 = F.normalize(f1, dim=1)
        #     f2 = F.normalize(f2, dim=1)
        residual = torch.cat((f1, f2), dim=1)

        #### compose two modalities by residual learning
        # f_cita proj
        residual = self.blend_conv(residual)
        features = f1 + residual  # other modality + residual (default)

        return features