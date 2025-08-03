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

class one_encoder_mlp_fused(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        features, n_layers = opts.latent_dim, opts.n_cs_layers  # e.g., 512
        
        # build shared MLP，注意最后一层输出 3*feats
        layers = []
        for i in range(n_layers - 1):
            layers += [ EqualizedLinear(features, features),
                        nn.LeakyReLU(0.2, inplace=True) ]
        # 最后一层把维度扩到 3*feats
        layers += [ EqualizedLinear(features, 3 * features),
                    nn.LeakyReLU(0.2, inplace=True) ]
        self.encoder = nn.Sequential(*layers)

        self.c_embed = nn.Sequential(
            nn.Conv2d(features, features // 8, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)    # 输出 (B, C, 1, 1)
        
        # fusion layer
        self.compos = CompositionalLayer_mlp(opts)
        
        # —— 新增：属性级二分类 head —— #
        # 取任意 [B,C,H,W] latent 做全局池化，输出单个 logit
        self.attr_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # [B,C,1,1]
            nn.Flatten(),                # [B,C]
            nn.Linear(features, 1)       # [B,1]
        )
    
    def forward(self, z: torch.Tensor, no_fused=False, eval_visul=False):
        # z: [B, 512, 8, 8]
        B, C, H, W = z.shape
        z_flat = z.contiguous().view(B, C, H * W).permute(0, 2, 1)  # [B, 64, 512]

        # Apply shared MLP to each spatial location
        out_flat = self.encoder(z_flat)  # [B, 64, 512]
        # split 成三份 每份都是 [B, HW, C]
        c, s_t, s_bg = torch.split(out_flat, C, dim=2)
        
        if eval_visul:
            # 把扁平后的 c 还原为 [B, C, H, W]
            c_map = c.permute(0, 2, 1).contiguous().view(B, C, H, W)
            return c_map
        
        if no_fused:
            c_map = c.permute(0, 2, 1).view(B, C, H, W)
            s_bg_map = s_bg.permute(0, 2, 1).view(B, C, H, W)
            s_t_map = s_t.permute(0, 2, 1).view(B, C, H, W)
            return c_map, s_t_map, s_bg_map
            
        # ↓ DAO 投影后的 logits
        c_map       = c.permute(0, 2, 1).contiguous().view(B, C, H, W)       # [B,features,H,W]
        proj_map    = self.c_embed(c_map)                                    # [B,proj_dim,H,W]
        g_c_logits  = self.global_pool(proj_map).contiguous().view(B, proj_map.size(1))  # [B,proj_dim]
                
        # 3) fuse shared + specific via residual-MLP
        fused_flat = self.compos(c, s_t, s_bg)   # [B, HW, feats]

        # Reshape back to [B, 512, 8, 8]
        c = c.permute(0, 2, 1).contiguous().view(B, C, H, W)
        s_bg = s_bg.permute(0, 2, 1).contiguous().view(B, C, H, W)
        s_t = s_t.permute(0, 2, 1).contiguous().view(B, C, H, W)
        fused = fused_flat.permute(0, 2, 1).contiguous().view(B, C, H, W)

        logit_c = self.attr_classifier(c)   # [B,1]
        logit_s_bg = self.attr_classifier(s_bg)   # [B,1]
        logit_s_t = self.attr_classifier(s_t)   # [B,1]
        
        return c, s_t, s_bg, fused, g_c_logits, logit_c, logit_s_t, logit_s_bg
    
class CompositionalLayer_mlp(nn.Module):
    def __init__(self, opts, normalization_sign=True):
        super().__init__()
        self.normalization = normalization_sign
        features = opts.latent_dim
        
        self.blend_mlp = nn.Sequential(
            EqualizedLinear(3 * features, features),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, f1, f2, f3):
        # if self.normalization:
        #     f1 = F.normalize(f1, dim=-1)
        #     f2 = F.normalize(f2, dim=-1)
        #     f3 = F.normalize(f3, dim=-1)
        # 拼接成 [B, HW, 3F]
        residual = torch.cat((f1, f2, f3), dim=-1)
        # 先做 3F->F，再做若干 F->F
        residual = self.blend_mlp(residual)
        # 残差连接
        return f1 + residual





class one_encoder_conv_fused(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        channels = opts.latent_dim  # typically 512
        n_layers = opts.n_cs_layers
        
        # 共享编码器：输出 3 * channels
        layers = []
        for _ in range(n_layers - 1):
            layers += [
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        # 最后一层输出 3*channels
        layers += [
            nn.Conv2d(channels, 3 * channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.encoder = nn.Sequential(*layers)

        # DAO 投影头：把 shared 特征从 channels 降到 channels//8
        self.c_embed = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # fusion layer，接收 3 路特征
        self.compos = CompositionalLayer_conv(opts)
        
        # —— 新增：属性级二分类 head —— #
        # 取任意 [B,C,H,W] latent 做全局池化，输出单个 logit
        self.attr_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # [B,C,1,1]
            nn.Flatten(),                # [B,C]
            nn.Linear(channels, 1)       # [B,1]
        )
    
    def forward(self, z: torch.Tensor, no_fused=False, eval_visul=False):
        B, C, H, W = z.shape
        # z: [B, 512, 8, 8]
        # 编码并 split 成三路 [B, 3C, H, W] -> 3 x [B, C, H, W]
        out = self.encoder(z)
        c, s_t, s_bg = torch.split(out, C, dim=1)

        if eval_visul:
            return c
        
        if no_fused:
            return c, s_t, s_bg

        # DAO 投影后的 logits，用于 KL loss
        proj_map   = self.c_embed(c)            # [B, C//8, H, W]
        g_c_logits = self.global_pool(proj_map)     # [B, C//8, 1, 1]
        g_c_logits = g_c_logits.view(B, -1)         # [B, C//8]

        # 3) 融合三路特征
        fused = self.compos(c, s_t, s_bg)
        
        # —— 属性 logit —— #
        logit_c = self.attr_classifier(c)   # [B,1]
        logit_s_bg = self.attr_classifier(s_bg)   # [B,1]
        logit_s_t = self.attr_classifier(s_t)   # [B,1]
        
        return c, s_t, s_bg, fused, g_c_logits, logit_c, logit_s_t, logit_s_bg
        
# 这里实现了 cs 融合
class CompositionalLayer_conv(nn.Module):

    def __init__(self, opts, normalization_sign=True):
        super().__init__()
        self.normalization = normalization_sign
        self.opts = opts
        channels = opts.latent_dim  # typically 512
        
        self.blend_conv = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, f1, f2, f3):
        """
        :param f1: shared-modality fts
        :param f2: specific-modality fts
        :return:
        """
        # if self.normalization:
        #     f1 = F.normalize(f1, dim=1)
        #     f2 = F.normalize(f2, dim=1)
        #     f3 = F.normalize(f3, dim=1)
        residual = torch.cat((f1, f2, f3), dim=1)

        #### compose two modalities by residual learning
        # f_cita proj
        residual = self.blend_conv(residual)
        features = f1 + residual  # other modality + residual (default)

        return features