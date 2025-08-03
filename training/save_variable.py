#!/usr/bin/env python
import os
import sys
import argparse
import torch

# 如果要从 training/ 目录直接运行，就把项目根加入 path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.mics import seed_all
from utils.config_utils import load_diffusion_config
from utils.model_utils import load_diffusion_model
from utils.data_utils import build_dataloaders


# —— 自己定义 argparse，把训练中 load_diffusion_model 和 build_dataloaders 会用到的参数都加上 —— 
def get_opts():
    parser = argparse.ArgumentParser(description="Save diffusion encoder cache")
    parser.add_argument('--diffu_config_path', type=str, required=False, default='configs/ffhq.yml')
    parser.add_argument('--cache_dir', type=str, required=False, default='ffhq_cached')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset_type', type=str, required=False, default='ffhq_glasses')
    parser.add_argument('--diffu_weights', type=str, required=False, default='pretrained_models/ffhq_p2.pt')
    parser.add_argument('--n_inv_step', type=int, default=40)
    parser.add_argument('--t_0', type=int, default=999)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample_type', type=str, default='ddim')
    parser.add_argument('--dataset_root', default='/home/ids/ziliu-24/diffu_asyrp_CA_revised', type=str)
    
    
    return parser.parse_args()

def main():
    opts = get_opts()
    seed_all(opts.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 加载 diffusion 模型与配置
    diff_configs = load_diffusion_config(opts.diffu_config_path)
    _, diffu_encoder, diffu_decoder, diffu_info = load_diffusion_model(
        opts=opts, config=diff_configs, device=device
    )
    diffu_encoder.eval()
    diffu_decoder.eval()

    # 构建 train & test dataloader（batch_size=1）
    opts.batch_size = 1
    train_bg_loader, train_t_loader, test_bg_loader, test_t_loader = build_dataloaders(opts)

    # 创建缓存目录
    train_cache = os.path.join(opts.cache_dir, 'train')
    test_cache  = os.path.join(opts.cache_dir, 'test')
    os.makedirs(train_cache, exist_ok=True)
    os.makedirs(test_cache,  exist_ok=True)

    seq_inv      = list(reversed(diffu_info['seq_inv']))
    seq_inv_next = list(reversed(diffu_info['seq_inv_next']))

    # 缓存训练集
    print(f"[save_cache] Caching TRAIN set to {train_cache}")
    for idx, (bg, t) in enumerate(zip(train_bg_loader, train_t_loader)):
        x_bg = bg.to(device).float()
        x_t  = t.to(device).float()

        # 注入噪声
        xT_bg = diffu_info['recon'].inject_noise_batch(
            diffu_encoder, diffu_decoder,
            x_bg, diffu_info['seq_inv'], diffu_info['seq_inv_next']
        )
        xT_t  = diffu_info['recon'].inject_noise_batch(
            diffu_encoder, diffu_decoder,
            x_t, diffu_info['seq_inv'], diffu_info['seq_inv_next']
        )

        # 准备保存结构
        enc_bg = {'h': [], 'hs': [], 'emb': []}
        enc_t  = {'h': [], 'hs': [], 'emb': []}
        x_bg_next, x_t_next = xT_bg, xT_t

        # 循环每个时间步，保存 encoder 输出
        for i, j in zip(seq_inv, seq_inv_next):
            t_i = torch.full((x_bg_next.size(0),), i, device=device)

            h_bg, hs_bg, emb_bg = diffu_encoder(x_bg_next, t_i)
            h_t,  hs_t,  emb_t  = diffu_encoder(x_t_next,  t_i)

            enc_bg['h'].append(h_bg.detach().cpu())
            enc_bg['hs'].append([h.detach().cpu() for h in hs_bg])
            enc_bg['emb'].append(emb_bg.detach().cpu())

            enc_t['h'].append(h_t.detach().cpu())
            enc_t['hs'].append([h.detach().cpu() for h in hs_t])
            enc_t['emb'].append(emb_t.detach().cpu())

            # decoder + reverse to get next x
            et_bg = diffu_decoder(h_bg, hs_bg, emb_bg, x_bg_next)
            et_t  = diffu_decoder(h_t,  hs_t,  emb_t,  x_t_next)

            x_bg_next, _ = diffu_info['recon'].reverse_denoising(
                xt=x_bg_next, t=t_i,
                t_next=torch.full((x_bg_next.size(0),), j, device=device),
                et=et_bg,
                logvars=diffu_info['logvar'],
                b=diffu_info['betas'],
                sampling_type=diffu_info['sample_type'],
                learn_sigma=diffu_info['learn_sigma']
            )
            x_t_next, _ = diffu_info['recon'].reverse_denoising(
                xt=x_t_next, t=t_i,
                t_next=torch.full((x_t_next.size(0),), j, device=device),
                et=et_t,
                logvars=diffu_info['logvar'],
                b=diffu_info['betas'],
                sampling_type=diffu_info['sample_type'],
                learn_sigma=diffu_info['learn_sigma']
            )

        # 保存文件
        cache = {
            'xT_bg': xT_bg.cpu(),
            'xT_t' : xT_t.cpu(),
            'enc_bg': enc_bg,
            'enc_t' : enc_t,
        }
        torch.save(cache, os.path.join(train_cache, f"{idx}.pt"))
        if idx % 100 == 0:
            print(f"[save_cache] saved TRAIN {idx}.pt")

    # 缓存测试集
    print(f"[save_cache] Caching TEST set to {test_cache}")
    for idx, (bg, t) in enumerate(zip(test_bg_loader, test_t_loader)):
        x_bg = bg.to(device).float()
        x_t  = t.to(device).float()

        xT_bg = diffu_info['recon'].inject_noise_batch(
            diffu_encoder, diffu_decoder,
            x_bg, diffu_info['seq_inv'], diffu_info['seq_inv_next']
        )
        xT_t  = diffu_info['recon'].inject_noise_batch(
            diffu_encoder, diffu_decoder,
            x_t, diffu_info['seq_inv'], diffu_info['seq_inv_next']
        )

        enc_bg = {'h': [], 'hs': [], 'emb': []}
        enc_t  = {'h': [], 'hs': [], 'emb': []}
        x_bg_next, x_t_next = xT_bg, xT_t

        for i, j in zip(seq_inv, seq_inv_next):
            t_i = torch.full((x_bg_next.size(0),), i, device=device)

            h_bg, hs_bg, emb_bg = diffu_encoder(x_bg_next, t_i)
            h_t,  hs_t,  emb_t  = diffu_encoder(x_t_next,  t_i)

            enc_bg['h'].append(h_bg.detach().cpu())
            enc_bg['hs'].append([h.detach().cpu() for h in hs_bg])
            enc_bg['emb'].append(emb_bg.detach().cpu())

            enc_t['h'].append(h_t.detach().cpu())
            enc_t['hs'].append([h.detach().cpu() for h in hs_t])
            enc_t['emb'].append(emb_t.detach().cpu())

            et_bg = diffu_decoder(h_bg, hs_bg, emb_bg, x_bg_next)
            et_t  = diffu_decoder(h_t,  hs_t,  emb_t,  x_t_next)

            x_bg_next, _ = diffu_info['recon'].reverse_denoising(
                xt=x_bg_next, t=t_i,
                t_next=torch.full((x_bg_next.size(0),), j, device=device),
                et=et_bg,
                logvars=diffu_info['logvar'],
                b=diffu_info['betas'],
                sampling_type=diffu_info['sample_type'],
                learn_sigma=diffu_info['learn_sigma']
            )
            x_t_next, _ = diffu_info['recon'].reverse_denoising(
                xt=x_t_next, t=t_i,
                t_next=torch.full((x_t_next.size(0),), j, device=device),
                et=et_t,
                logvars=diffu_info['logvar'],
                b=diffu_info['betas'],
                sampling_type=diffu_info['sample_type'],
                learn_sigma=diffu_info['learn_sigma']
            )

        cache = {
            'xT_bg': xT_bg.cpu(),
            'xT_t' : xT_t.cpu(),
            'enc_bg': enc_bg,
            'enc_t' : enc_t,
        }
        torch.save(cache, os.path.join(test_cache, f"{idx}.pt"))
        if idx % 50 == 0:
            print(f"[save_cache] saved TEST {idx}.pt")

if __name__ == '__main__':
    main()
