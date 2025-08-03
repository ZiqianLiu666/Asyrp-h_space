import os, sys 
# 保证项目根目录（utils 的上一级）在模块搜索路径中
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, proj_root)

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from utils.model_utils import load_diffusion_model, load_cs_model, load_cs_model_specific
from utils.data_utils import build_dataloaders
from utils.config_utils import load_diffusion_config
from PIL import Image
from options.train_options import TrainOptions
from tqdm import tqdm
import time

class inference:
    def __init__(self, opts):
        self.opts = opts
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # ---------------- Diffusion ---------------- #
        self.diff_configs = load_diffusion_config(opts.diffu_config_path)
        self.diffu_model, self.diffu_encoder, self.diffu_decoder, self.diffu_info = \
            load_diffusion_model(opts=opts, config=self.diff_configs, device=self.device)
    
        for p in self.diffu_encoder.parameters(): p.requires_grad = False
        for p in self.diffu_decoder.parameters(): p.requires_grad = False
  
        self.diffu_model.eval(); self.diffu_encoder.eval(); self.diffu_decoder.eval()

        # -------------  时间步区间设置 ------------- #
        # T（最大时间步）＝ reversed(seq_inv)[0] ；默认编辑区间 [t_start_edit, t_end_edit]
        self.t_start_edit = list(reversed(self.diffu_info['seq_inv']))[0]   # ≈999
        # 直接从 opts 读取，可在命令行传入 --t_end_edit N，否则用 500
        self.t_end_edit   = opts.t_end_edit

        self.logvar       = self.diffu_info['logvar']
        self.betas        = self.diffu_info['betas']
        self.sample_type  = self.diffu_info['sample_type']
        self.learn_sigma  = self.diffu_info['learn_sigma']

        # ---------------- 其它网络 ---------------- #
        # load CS model
        if opts.cs_net_type.startswith('specific'):
            self.cs_mlp_net = load_cs_model_specific(opts.cs_model_weights, opts, self.device)
            print("using specific network")
        else:
            print("using baseline network")
            self.cs_mlp_net = load_cs_model(opts.cs_model_weights, opts, self.device)
        
        if opts.cs_net_type.endswith('fused'):
            print("and also is fused network")
            self.swap_blend_layer = self.cs_mlp_net.compos
        else:
            print("no fused!!")    
            
        print(opts.exp_scheme)
        print(opts.cs_net_type)
        print(opts.n_cs_layers)
        print(opts.t_end_edit)
        self.cs_mlp_net.eval()
        print(opts.dataset_type)
        # ---------------- 数据 & 优化器 ----------- #
        if opts.dataset_type in ["infer_special_glasses", "infer_special_age", "infer_special_smile", "infer_special_gender"]:
            self.test_bg_dataloader,  self.test_t_dataloader = build_dataloaders(opts)
        else:
            _, _, self.test_bg_dataloader,  self.test_t_dataloader = build_dataloaders(opts)

        # ---------------- 日志目录 ---------------- #
        self.seq_inv = list(reversed(self.diffu_info['seq_inv']))
        self.seq_inv_next = list(reversed(self.diffu_info['seq_inv_next']))
        
    @torch.no_grad()
    def eval_recon_batch_swap(self, batch_xT_bg, batch_xT_t):
        # for swap
        x_bg = batch_xT_bg
        x_t = batch_xT_t
        
        with torch.no_grad():
            for step, (i, j) in enumerate(zip(self.seq_inv, self.seq_inv_next)):
                t      = torch.full((x_bg.size(0),), i, device=x_bg.device)
                t_next = torch.full((x_bg.size(0),), j, device=x_bg.device)

                middle_h_bg, hs_bg, emb_bg = self.diffu_encoder(x_bg, t)
                middle_h_t, hs_t, emb_t = self.diffu_encoder(x_t, t)

                if self.t_end_edit <= i <= self.t_start_edit:
                    if self.opts.cs_net_type.endswith('fused'):
                        if 'mlp' in self.opts.cs_net_type:
                            if self.opts.cs_net_type=="specific_mlp_fused":
                                c_bg, s_bg = self.cs_mlp_net(middle_h_bg, is_bg=True, no_fused=True)
                                c_t, s_t = self.cs_mlp_net(middle_h_t, is_bg=False, no_fused=True)
                            elif self.opts.cs_net_type=="equalized_mlp_fused":
                                c_bg, s_bg = self.cs_mlp_net(middle_h_bg, no_fused=True)
                                c_t, s_t = self.cs_mlp_net(middle_h_t, no_fused=True)
                            
                            B, C, H, W = c_bg.shape
                            c_bg = c_bg.contiguous().view(B, C, H*W).permute(0,2,1)  
                            s_bg = s_bg.contiguous().view(B, C, H*W).permute(0,2,1)
                            c_t = c_t.contiguous().view(B, C, H*W).permute(0,2,1) 
                            s_t = s_t.contiguous().view(B, C, H*W).permute(0,2,1)
                            fused_bg2t = self.swap_blend_layer(c_bg, s_t).permute(0, 2, 1).contiguous().view(B, C, H, W)
                            fused_t2bg = self.swap_blend_layer(c_t, s_bg).permute(0, 2, 1).contiguous().view(B, C, H, W)
                            
                            middle_h_bg = fused_bg2t
                            middle_h_t = fused_t2bg
                            
                        elif 'conv' in self.opts.cs_net_type:
                            if self.opts.cs_net_type=="specific_conv_fused":
                                c_bg, s_bg = self.cs_mlp_net(middle_h_bg, is_bg=True, no_fused=True)
                                c_t, s_t = self.cs_mlp_net(middle_h_t, is_bg=False, no_fused=True)
                            elif self.opts.cs_net_type=="conv_fused":
                                c_bg, s_bg = self.cs_mlp_net(middle_h_bg, no_fused=True)
                                c_t, s_t = self.cs_mlp_net(middle_h_t, no_fused=True)
                                
                            fused_bg2t = self.swap_blend_layer(c_bg, s_t)
                            fused_t2bg = self.swap_blend_layer(c_t, s_bg)
                            
                            middle_h_bg = fused_bg2t
                            middle_h_t = fused_t2bg
                    
                    else:
                        if 'mlp' in self.opts.cs_net_type:
                            if self.opts.cs_net_type == "specific_mlp":
                                c_bg, s_bg = self.cs_mlp_net(middle_h_bg, is_bg=True, infer=True)
                                c_t, s_t = self.cs_mlp_net(middle_h_t, is_bg=False, infer=True)
                            elif self.opts.cs_net_type == "equalized_mlp":
                                c_bg, s_bg = self.cs_mlp_net(middle_h_bg, infer=True)
                                c_t, s_t = self.cs_mlp_net(middle_h_t, infer=True)
                                
                        elif 'conv' in self.opts.cs_net_type:
                            if self.opts.cs_net_type=="specific_conv":
                                c_bg, s_bg = self.cs_mlp_net(middle_h_bg, is_bg=True, infer=True)
                                c_t, s_t = self.cs_mlp_net(middle_h_t, is_bg=False, infer=True)
                            elif self.opts.cs_net_type=="conv":
                                c_bg, s_bg = self.cs_mlp_net(middle_h_bg, infer=True)
                                c_t, s_t = self.cs_mlp_net(middle_h_t, infer=True)
                            
                        middle_h_bg = c_bg + s_t
                        middle_h_t = c_t + s_bg
                
                et_bg2t = self.diffu_decoder(middle_h_bg, hs_bg, emb_bg, x_bg)       
                et_t2bg = self.diffu_decoder(middle_h_t, hs_t, emb_t, x_t)

                x_bg, _ = self.diffu_info['recon'].reverse_denoising(
                    xt=x_bg, t=t, t_next=t_next,
                    et=et_bg2t,
                    logvars=self.logvar,
                    b=self.betas,
                    sampling_type='ddim',
                    learn_sigma=self.learn_sigma
                )
                
                x_t, _ = self.diffu_info['recon'].reverse_denoising(
                    xt=x_t, t=t, t_next=t_next,
                    et=et_t2bg,
                    logvars=self.logvar,
                    b=self.betas,
                    sampling_type='ddim',
                    learn_sigma=self.learn_sigma
                )
                
        # 但是理想情况下这里的x_bg已经戴上眼镜变为target了
        return x_bg, x_t
    
    @torch.no_grad()
    def eval_recon_batch_rec(self, batch_xT_bg, batch_xT_t):
        # for recon
        x_bg = batch_xT_bg
        x_t = batch_xT_t
        
        with torch.no_grad():
            for step, (i, j) in enumerate(zip(self.seq_inv, self.seq_inv_next)):
                t      = torch.full((x_bg.size(0),), i, device=x_bg.device)
                t_next = torch.full((x_bg.size(0),), j, device=x_bg.device)

                middle_h_bg, hs_bg, emb_bg = self.diffu_encoder(x_bg, t)
                middle_h_t, hs_t, emb_t = self.diffu_encoder(x_t, t)

                if self.t_end_edit <= i <= self.t_start_edit:
                    if self.opts.cs_net_type.endswith('fused'):
                        if 'mlp' in self.opts.cs_net_type:
                            if self.opts.cs_net_type=="specific_mlp_fused":
                                _, _, fused_bg, _, _, _ = self.cs_mlp_net(middle_h_bg, is_bg=True)
                                _, _, fused_t, _, _, _ = self.cs_mlp_net(middle_h_t, is_bg=False)
                            elif self.opts.cs_net_type=="equalized_mlp_fused":
                                _, _, fused_bg, _, _, _ = self.cs_mlp_net(middle_h_bg)
                                _, _, fused_t, _, _, _ = self.cs_mlp_net(middle_h_t)

                        elif 'conv' in self.opts.cs_net_type:
                            if self.opts.cs_net_type=="specific_conv_fused":
                                _, _, fused_bg, _, _, _ = self.cs_mlp_net(middle_h_bg, is_bg=True)
                                _, _, fused_t, _, _, _ = self.cs_mlp_net(middle_h_t, is_bg=False)

                            elif self.opts.cs_net_type=="conv_fused":
                                _, _, fused_bg, _, _, _ = self.cs_mlp_net(middle_h_bg)
                                _, _, fused_t, _, _, _ = self.cs_mlp_net(middle_h_t)

                            
                        middle_h_bg = fused_bg
                        middle_h_t = fused_t
                    
                    else:
                        if 'mlp' in self.opts.cs_net_type:
                            if self.opts.cs_net_type == "specific_mlp":
                                c_bg, s_bg, _, _ , _ = self.cs_mlp_net(middle_h_bg, is_bg=True)
                                c_t, s_t, _, _ , _ = self.cs_mlp_net(middle_h_t, is_bg=False)

                            elif self.opts.cs_net_type == "equalized_mlp":
                                c_bg, s_bg, _, _ , _ = self.cs_mlp_net(middle_h_bg)
                                c_t, s_t, _, _ , _ = self.cs_mlp_net(middle_h_t)

                                
                        elif 'conv' in self.opts.cs_net_type:
                            if self.opts.cs_net_type=="specific_conv":
                                c_bg, s_bg, _, _ , _ = self.cs_mlp_net(middle_h_bg, is_bg=True)
                                c_t, s_t, _, _ , _ = self.cs_mlp_net(middle_h_t, is_bg=False)

                            elif self.opts.cs_net_type=="conv":
                                c_bg, s_bg, _, _ , _ = self.cs_mlp_net(middle_h_bg)
                                c_t, s_t, _, _ , _ = self.cs_mlp_net(middle_h_t)

                        middle_h_bg = c_bg + s_bg
                        middle_h_t = c_t + s_t
                
                et_bg = self.diffu_decoder(middle_h_bg, hs_bg, emb_bg, x_bg)       
                et_t = self.diffu_decoder(middle_h_t, hs_t, emb_t, x_t)

                x_bg, _ = self.diffu_info['recon'].reverse_denoising(
                    xt=x_bg, t=t, t_next=t_next,
                    et=et_bg,
                    logvars=self.logvar,
                    b=self.betas,
                    sampling_type='ddim',
                    learn_sigma=self.learn_sigma
                )
                
                x_t, _ = self.diffu_info['recon'].reverse_denoising(
                    xt=x_t, t=t, t_next=t_next,
                    et=et_t,
                    logvars=self.logvar,
                    b=self.betas,
                    sampling_type='ddim',
                    learn_sigma=self.learn_sigma
                )
                
        # 重建
        return x_bg, x_t
    
    def hspace_rec(self, batch_xT_bg, batch_xT_t):
        # for recon
        x_bg = batch_xT_bg
        x_t = batch_xT_t
        
        with torch.no_grad():
            for step, (i, j) in enumerate(zip(self.seq_inv, self.seq_inv_next)):
                t      = torch.full((x_bg.size(0),), i, device=x_bg.device)
                t_next = torch.full((x_bg.size(0),), j, device=x_bg.device)

                middle_h_bg, hs_bg, emb_bg = self.diffu_encoder(x_bg, t)
                middle_h_t, hs_t, emb_t = self.diffu_encoder(x_t, t)
                
                et_bg = self.diffu_decoder(middle_h_bg, hs_bg, emb_bg, x_bg)       
                et_t = self.diffu_decoder(middle_h_t, hs_t, emb_t, x_t)

                x_bg, _ = self.diffu_info['recon'].reverse_denoising(
                    xt=x_bg, t=t, t_next=t_next,
                    et=et_bg,
                    logvars=self.logvar,
                    b=self.betas,
                    sampling_type='ddim',
                    learn_sigma=self.learn_sigma
                )
                
                x_t, _ = self.diffu_info['recon'].reverse_denoising(
                    xt=x_t, t=t, t_next=t_next,
                    et=et_t,
                    logvars=self.logvar,
                    b=self.betas,
                    sampling_type='ddim',
                    learn_sigma=self.learn_sigma
                )
                
        # h-space重建
        return x_bg, x_t
                
                
                
    
    def noise_injection(self, batch_image):
        batch_xT = self.diffu_info['recon'].inject_noise_batch(
            self.diffu_encoder,
            self.diffu_decoder,
            batch_image,
            self.diffu_info['seq_inv'],
            self.diffu_info['seq_inv_next']
        )
        return batch_xT
    
    def save_image(self, tensor, path):
        img = tensor.clamp(-1, 1).add(1).div(2).mul(255).byte()
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img).save(path)

    def inference(self):
        self.cs_mlp_net.eval()
        # 创建输出目录
        folders = ['real_x','real_y','recon_x','recon_y','swap_x2y','swap_y2x', 'hspace_x','hspace_y']
        # folders = ['recon_x','recon_y','swap_x2y','swap_y2x']
        for f in folders:
            os.makedirs(os.path.join(self.opts.results_dir, f), exist_ok=True)
            
        swap_time_total       = 0.0
        recon_cs_time_total   = 0.0
        recon_hspace_time_total = 0.0
        img_count             = 0

        total = len(self.test_bg_dataloader)
        with torch.no_grad():
            for batch_idx, ((batch_bg, names_bg), (batch_t, names_t)) in enumerate(
                    tqdm(zip(self.test_bg_dataloader, self.test_t_dataloader),
                         total=total, desc="Inference")):

                # ——【1】先筛一次索引，只保留“还没做完”的样本
                idxs = []
                for i, (nx, ny) in enumerate(zip(names_bg, names_t)):
                    base = self.opts.results_dir
                    paths = [
                        os.path.join(base, 'real_x',   nx),
                        os.path.join(base, 'real_y',   ny),
                        os.path.join(base, 'recon_x',  nx),
                        os.path.join(base, 'recon_y',  ny),
                        os.path.join(base, 'hspace_x',  nx),
                        os.path.join(base, 'hspace_y',  ny),
                        os.path.join(base, 'swap_x2y', nx),
                        os.path.join(base, 'swap_y2x', ny),
                    ]
                    if not all(os.path.exists(p) for p in paths):
                        idxs.append(i)
                if not idxs:
                    continue  # 整个 batch 全做完，跳过

                # 按 idxs 抽子 batch
                x_bg = batch_bg[idxs].to(self.device).float()
                x_t  = batch_t[idxs].to(self.device).float()
                sub_names_bg = [names_bg[i] for i in idxs]
                sub_names_t  = [names_t[i] for i in idxs]

                # ——【2】只对子 batch 跑一次推理
                xt_bg = self.noise_injection(x_bg)
                xt_t  = self.noise_injection(x_t)
                
                img_count += x_bg.size(0)  # 其实就是 +=1
                
                t0 = time.time()
                swap_bg2t, swap_t2bg = self.eval_recon_batch_swap(xt_bg, xt_t)
                swap_time_total += time.time() - t0
                
                t1 = time.time()
                rec_bg,    rec_t    = self.eval_recon_batch_rec(xt_bg, xt_t)
                recon_cs_time_total += time.time() - t1
                
                t2 = time.time()
                rec_h_bg, rec_h_t = self.hspace_rec(xt_bg, xt_t)
                recon_hspace_time_total += time.time() - t2
                
                # ——【3】最后再逐个保存
                for k, (nb, nt) in enumerate(zip(sub_names_bg, sub_names_t)):
                    # 原图
                    self.save_image(x_bg[k:k+1], os.path.join(self.opts.results_dir, 'real_x',   nb))
                    self.save_image(x_t[k:k+1],  os.path.join(self.opts.results_dir, 'real_y',   nt))
                    # 重建
                    self.save_image(rec_bg[k:k+1], os.path.join(self.opts.results_dir, 'recon_x',  nb))
                    self.save_image(rec_t[k:k+1],  os.path.join(self.opts.results_dir, 'recon_y',  nt))
                    # h-space 重建
                    self.save_image(rec_h_bg[k:k+1], os.path.join(self.opts.results_dir, 'hspace_x', nb))
                    self.save_image(rec_h_t[k:k+1],  os.path.join(self.opts.results_dir, 'hspace_y', nt))
                    # 交换
                    self.save_image(swap_bg2t[k:k+1], os.path.join(self.opts.results_dir, 'swap_x2y', nb))
                    self.save_image(swap_t2bg[k:k+1], os.path.join(self.opts.results_dir, 'swap_y2x', nt))
                    
        avg_swap     = swap_time_total       / img_count
        avg_recon_cs = recon_cs_time_total   / img_count
        avg_hspace   = recon_hspace_time_total / img_count

        print(f"Processed {img_count} images.")
        print(f"Average per-image times (s): swap = {avg_swap:.4f}, recon_cs = {avg_recon_cs:.4f}, hspace = {avg_hspace:.4f}")

if __name__ == "__main__":
    # 使用已有的 TrainOptions
    parser = TrainOptions()
    opts   = parser.parser.parse_args()

    inf = inference(opts)
    inf.inference()

