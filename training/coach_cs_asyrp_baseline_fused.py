import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from training.ranger import Ranger
from utils.mics import seed_all, aggregate_loss_dict
from utils.model_utils import load_diffusion_model, load_cs_model, load_id_lpips_models
from utils.data_utils import build_dataloaders
from utils.visual_utils import visualize_batch_grid
from utils.config_utils import load_diffusion_config
import torch.nn.functional as F
import math 

class Coach:
	def __init__(self, opts):
		self.opts = opts
		self.global_step = 1
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
		self.cs_mlp_net = load_cs_model(opts.cs_model_weights, opts, self.device)
		self.swap_blend_layer = self.cs_mlp_net.compos
		with torch.no_grad():
			self.lpips_loss, self.id_loss = load_id_lpips_models(opts, self.device)
		# ---------- 冻结 LPIPS & ID-loss 的参数 ----------
		self.lpips_loss.eval()
		for p in self.lpips_loss.parameters():
			p.requires_grad = False
		self.id_loss.eval()
		for p in self.id_loss.parameters():
			p.requires_grad = False

		# ---------------- 数据 & 优化器 ----------- #
		self.train_bg_dataloader, self.train_t_dataloader, \
		self.test_bg_dataloader,  self.test_t_dataloader = build_dataloaders(opts)

		self.optimizer = self.configure_optimizers()

		# ---------------- 日志目录 ---------------- #
		self.log_dir        = os.path.join(opts.results_dir, 'logs')
		self.checkpoint_dir = os.path.join(opts.results_dir, 'checkpoints')
		os.makedirs(self.log_dir, exist_ok=True)
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		os.makedirs(os.path.join(self.log_dir, 'images'), exist_ok=True)
  
		self.seq_inv = list(reversed(self.diffu_info['seq_inv']))
		self.seq_inv_next = list(reversed(self.diffu_info['seq_inv_next']))
		
		self.n_print      = 0
		self.best_val_loss = None
		seed_all(opts.seed)
  
		# 记录时间步总数
		self.T = len(self.seq_inv)              # e.g. 999  或  self.opts.t_max
		# 余弦权 w_t = cos((1-t/(T-1))*π/2)  与原实现保持一致
		w = torch.cos(torch.linspace(0, 1, steps=self.T) * math.pi/2)
		self.w_sum = float(w.sum())             # 一个标量常数

	def eval_recon_batch(self, batch_xT,
						latent_s_list=None, replace=False):
		"""
		replace=False  → 完全不改 middle_h   (h-space 原样重建)
		replace=True   → 在编辑区/或 swap 时替换 middle_h
		"""
		x = batch_xT
  
		with torch.no_grad():
			for step, (i, j) in enumerate(zip(self.seq_inv, self.seq_inv_next)):
				t      = torch.full((x.size(0),), i, device=x.device)
				t_next = torch.full((x.size(0),), j, device=x.device)

				middle_h, hs, emb = self.diffu_encoder(x, t)

				if replace:                                     # 只有打开开关才考虑替换
					if self.t_end_edit <= i <= self.t_start_edit:
						latent_c = self.cs_mlp_net(middle_h, eval_visul=True)
	  
						B, C, H, W = latent_c.shape
						if self.opts.cs_net_type == "equalized_mlp_fused":
							f1 = latent_c.contiguous().view(B, C, H*W).permute(0,2,1)   # [B, HW, C]
							f2 = latent_s_list[step].contiguous().view(B, C, H*W).permute(0,2,1)
						elif self.opts.cs_net_type == "conv_fused":
							f1 = latent_c
							f2 = latent_s_list[step]
						else:
							raise ValueError(f"非法的 cs_net_type: {self.opts.cs_net_type}")
						
						fused = self.swap_blend_layer(f1, f2)
						
						if self.opts.cs_net_type == "equalized_mlp_fused":
							# reshape 回 [B,C,H,W]
							fused_flat = fused.permute(0, 2, 1).contiguous().view(B, C, H, W)
						elif self.opts.cs_net_type == "conv_fused":
							fused_flat = fused
							
						middle_h = fused_flat

				et = self.diffu_decoder(middle_h, hs, emb, x)
	
				x, _ = self.diffu_info['recon'].reverse_denoising(
					xt=x, t=t, t_next=t_next,
					et=et,
					logvars=self.logvar,
					b=self.betas,
					sampling_type='ddim',
					learn_sigma=self.learn_sigma
				)

		return x
		
	def noise_injection(self, batch_image):
		with torch.no_grad():
			batch_xT = self.diffu_info['recon'].inject_noise_batch(
				self.diffu_encoder,
				self.diffu_decoder,
				batch_image,
				self.diffu_info['seq_inv'],
				self.diffu_info['seq_inv_next']
			)
		return batch_xT
	
	def configure_optimizers(self):
		param_groups = []

		# Default learning rate
		lr_main = self.opts.learning_rate

		# Learning rates for each group (can be the same or different)
		lr_mlp = self.opts.lr_mlp if hasattr(self.opts, 'lr_mlp') else lr_main
		lr_enc = self.opts.lr_enc if hasattr(self.opts, 'lr_enc') else lr_main
		lr_dec = self.opts.lr_dec if hasattr(self.opts, 'lr_dec') else lr_main

		# Add cs_mlp_net
		param_groups.append({
			'params': self.cs_mlp_net.parameters(),
			'lr': lr_mlp
		})

		# Add encoder (if training)
		if self.opts.train_diffu_encoder:
			param_groups.append({
				'params': self.diffu_encoder.parameters(),
				'lr': lr_enc
			})

		# Add decoder (if training)
		if self.opts.train_diffu_decoder:
			param_groups.append({
				'params': self.diffu_decoder.parameters(),
				'lr': lr_dec
			})

		# Choose optimizer
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(param_groups)
		else:
			optimizer = Ranger(param_groups)

		return optimizer
		
	def train(self):
		self.cs_mlp_net.train()
		# self.diffu_decoder.train()

		while self.global_step < self.opts.max_steps:

			for batch_idx, ((batch_bg, _), (batch_t, _)) in enumerate(zip(self.train_bg_dataloader, self.train_t_dataloader)):
				batch_bg = batch_bg.to(self.device).float()
				batch_t  = batch_t.to(self.device).float()
				with torch.no_grad():
					batch_xT_bg = self.noise_injection(batch_bg)
					batch_xT_t = self.noise_injection(batch_t)
	 
				x_bg_next = batch_xT_bg
				x_t_next = batch_xT_t
	
				latent_t_s_list = []
				latent_bg_s_list = []
				# for logging the training 
				lat_d_list, img_bg_d_list, img_t_d_list = [], [], []
		
				for step, (i, j) in enumerate(zip(self.seq_inv, self.seq_inv_next)):
					t      = torch.full((batch_xT_bg.size(0),), i, device=self.device)
					t_next = torch.full((batch_xT_bg.size(0),), j, device=self.device)
	 
					# ⚠️ 每步都清梯度
					self.optimizer.zero_grad()
     
					middle_h_bg, hs_bg, emb_bg = self.diffu_encoder(x_bg_next, t)
					w_bg_pSp = middle_h_bg
					middle_h_t, hs_t, emb_t = self.diffu_encoder(x_t_next, t)
					w_t_pSp = middle_h_t
			
					# -------- 是否在编辑区间？ -------- #
					if self.t_end_edit <= i <= self.t_start_edit:
						latent_c_bg, latent_s_bg, fused_bg, g_c_bg_logits, logit_c_bg, logit_s_bg = self.cs_mlp_net(middle_h_bg)
						latent_c_t, latent_s_t, fused_t, g_c_t_logits, logit_c_t, logit_s_t = self.cs_mlp_net(middle_h_t)
	  
						middle_h_bg = fused_bg
						middle_h_t = fused_t
	  
						latent_t_s_list.append(latent_s_t)
						latent_bg_s_list.append(latent_s_bg)
	  
					et_bg = self.diffu_decoder(middle_h_bg, hs_bg, emb_bg, x_bg_next)
					et_t = self.diffu_decoder(middle_h_t, hs_t, emb_t, x_t_next)

					x_bg_next, x0_t_bg = self.diffu_info['recon'].reverse_denoising(
						xt=x_bg_next, t=t, t_next=t_next,
						et=et_bg,
						logvars=self.logvar,
						b=self.betas,
						sampling_type='ddim',
						learn_sigma=self.learn_sigma
					)
		
					x_t_next, x0_t_t = self.diffu_info['recon'].reverse_denoising(
						xt=x_t_next, t=t, t_next=t_next,
						et=et_t,
						logvars=self.logvar,
						b=self.betas,
						sampling_type='ddim',
						learn_sigma=self.learn_sigma
					)
     
					x_bg_next = x_bg_next.detach()
					x_t_next = x_t_next.detach()
     
					# Calculate loss
					if self.t_end_edit <= i <= self.t_start_edit:
						loss_lat, loss_lat_dict = self.calc_latent_loss(latent_c_bg, latent_s_bg, latent_c_t, latent_s_t, w_bg_pSp, w_t_pSp, \
																	fused_bg, fused_t, \
																	g_c_bg_logits, g_c_t_logits, logit_c_bg, logit_c_t, logit_s_bg, logit_s_t, \
																	)

						loss_img_bg, loss_img_dict_bg = self.calc_image_loss_step(batch_bg, x0_t_bg, step)
						loss_img_t, loss_img_dict_t = self.calc_image_loss_step(batch_t, x0_t_t, step)

						# 累积字典
						lat_d_list.append(loss_lat_dict)
						img_bg_d_list.append(loss_img_dict_bg)
						img_t_d_list.append(loss_img_dict_t)
	  
						loss = loss_lat + loss_img_bg + loss_img_t

						loss.backward()
						self.optimizer.step()

				rec_img_bg = x_bg_next
				rec_img_t = x_t_next
	
				# Logging related
				train_loss_dict = None
				if self.global_step % self.opts.image_interval == 0:
					
					with torch.no_grad():
						diffu_x_bg = self.eval_recon_batch(batch_xT_bg)
						diffu_x_t  = self.eval_recon_batch(batch_xT_t)
						swap_x_bg  = self.eval_recon_batch(batch_xT_bg, latent_s_list=latent_t_s_list, replace=True)
						swap_x_t   = self.eval_recon_batch(batch_xT_t, latent_s_list=latent_bg_s_list, replace=True)
					visualize_batch_grid(
						image_batches=[
							torch.stack([batch_bg[0], batch_t[0]], dim=0),       # Originals
							torch.stack([diffu_x_bg[0], diffu_x_t[0]], dim=0),   # diffu recon
							torch.stack([rec_img_bg[0], rec_img_t[0]], dim=0),   # Reconstructed
							torch.stack([swap_x_bg[0], swap_x_t[0]], dim=0),   # swap
						],
						titles=["Input", "h-space", "Reconstruction", "Swap"],
						save_path=f"{self.log_dir}/images/train_step_{self.global_step}.png"
					)

				if self.global_step % 25 == 0:

					avg_lat_dict   = aggregate_loss_dict(lat_d_list)
					avg_img_bg_dict= aggregate_loss_dict(img_bg_d_list)
					avg_img_t_dict = aggregate_loss_dict(img_t_d_list)

					train_loss_dict = self.merge_loss_dict(
						avg_lat_dict,
						avg_img_bg_dict,
						avg_img_t_dict
					)
					self.print_metrics(train_loss_dict, prefix='train')
				if self.global_step % 25 == 0 and self.n_print < 100:
					self.write_metrics_to_txt(train_loss_dict, prefix='train', filename='loss_for_check.txt')
					self.n_print += 1

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss_sum'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss_sum']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						# save based on validation
						self.checkpoint_me(val_loss_dict, is_best=False)
					elif train_loss_dict is not None:
						# fall back to training loss only if it was computed
						self.checkpoint_me(train_loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self):
		self.cs_mlp_net.eval()

		agg_loss_lat_dict = []
		agg_loss_img_dict_bg = []
		agg_loss_img_dict_t = []

		# for batch_idx, batch in enumerate(self.test_dataloader):
		# 	x, y = batch
		for batch_idx, (batch_bg, batch_t) in enumerate(zip(self.test_bg_dataloader, self.test_t_dataloader)):

			with torch.no_grad():

				x_bg, x_t = batch_bg.to(self.device).float(), batch_t.to(self.device).float()

				batch_xT_bg = self.noise_injection(x_bg)
				batch_xT_t = self.noise_injection(x_t)

				rec_x_bg, middle_h_bg, latent_bg_c, latent_bg_s = self.reverse_edit_batch(batch_xT_bg, t_edit=self.t_edit, is_salient_bg=True)
				rec_x_t, middle_h_t, latent_t_c, latent_t_s = self.reverse_edit_batch(batch_xT_t, t_edit=self.t_edit, is_salient_bg=False)	
				# Calculate loss
				_, loss_lat_dict = self.calc_latent_loss(latent_bg_c, latent_bg_s, latent_t_c, latent_t_s, middle_h_bg, middle_h_t)
				_, loss_img_dict_bg, _ = self.calc_image_loss(x_bg, rec_x_bg)
				_, loss_img_dict_t, _ = self.calc_image_loss(x_t, rec_x_t)

			agg_loss_lat_dict.append(loss_lat_dict)
			agg_loss_img_dict_bg.append(loss_img_dict_bg)
			agg_loss_img_dict_t.append(loss_img_dict_t)

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				#self.pSp_net.train()
				self.cs_mlp_net.train()
				return None  # Do not log, inaccurate in first batch

		loss_lat_dict = aggregate_loss_dict(agg_loss_lat_dict)
		loss_img_dict_bg = aggregate_loss_dict(agg_loss_img_dict_bg)
		loss_img_dict_t = aggregate_loss_dict(agg_loss_img_dict_t)

		loss_dict = self.merge_loss_dict(loss_lat_dict, loss_img_dict_bg, loss_img_dict_t)

		#self.log_metrics(loss_dict, prefix='test')
		# self.print_metrics(loss_dict, prefix='test')

		self.cs_mlp_net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def write_metrics_to_txt(self, metrics_dict, prefix, filename):

		with open(os.path.join(self.checkpoint_dir, filename), 'a') as f:
			f.write(f'Metrics for {prefix}, Step - {self.global_step}')
			f.write(f'\n{metrics_dict}\n')

	
	def calc_image_loss_step(self, x: torch.Tensor, x0_t: torch.Tensor, t: int):
		"""
		Single-step image loss:
		  x      : ground truth image (B,C,H,W)
		  x0_t   : model's reconstruction estimate at time step t
		  t      : current time index (0..T-1)
		"""
		# 权重可选：你可以沿用 cos weight
		T = len(self.seq_inv)
		weight = math.cos((1 - t/(T-1)) * math.pi/2)

		total = 0.0
		loss_dict = {}

		if self.opts.id_lambda > 0:
			loss_id, sim, _ = self.id_loss(x0_t, x, x)
			loss_dict['loss_id'] = float(loss_id * self.opts.id_lambda * weight)
			loss_dict['id_improve'] = float(sim)
			total += loss_id * self.opts.id_lambda * weight

		if self.opts.pix_lambda > 0:
			loss_pix = F.mse_loss(x0_t, x)
			loss_dict['loss_pix'] = float(loss_pix * self.opts.pix_lambda * weight)
			total += loss_pix * self.opts.pix_lambda * weight

		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(x0_t, x)
			loss_dict['loss_lpips'] = float(loss_lpips * self.opts.lpips_lambda * weight)
			total += loss_lpips * self.opts.lpips_lambda * weight

		loss_dict['loss'] = float(total)
		return total, loss_dict


	def calc_latent_loss(self, latent_bg_c, latent_bg_s, latent_t_c, latent_t_s, w_bg_pSp, w_t_pSp, \
						fused_bg, fused_t, \
						g_c_bg_logits, g_c_t_logits, logit_c_bg, logit_c_t, logit_s_bg, logit_s_t, \
						   ):

		loss_dict = {}
		total_loss = 0.0

		# Background silence loss
		if self.opts.sbg_lambda > 0:
			loss_sbg = F.mse_loss(latent_bg_s, torch.zeros_like(latent_bg_s))

			loss_dict['loss_silent_bg'] = float(loss_sbg)
			total_loss += loss_sbg * self.opts.sbg_lambda

		# Latent reconstruction distance loss
		if self.opts.lat_lambda > 0:
			loss_bg = F.mse_loss(fused_bg, w_bg_pSp.detach())
			loss_t = F.mse_loss(fused_t, w_t_pSp.detach())

			loss_dict['loss_distance_bg'] = float(loss_bg)
			loss_dict['loss_distance_t'] = float(loss_t)
			total_loss += (loss_bg + loss_t) * self.opts.lat_lambda

		# 3) DAO: bi‐directional KL on pooled shared features
		if self.opts.dao_lambda > 0:
			T = 0.5  # temperature
   
			log_p_bg = F.log_softmax(g_c_bg_logits / T, dim=-1)
			p_t      = F.softmax   (g_c_t_logits / T, dim=-1)
			log_p_t  = F.log_softmax(g_c_t_logits / T, dim=-1)
			p_bg     = F.softmax   (g_c_bg_logits / T, dim=-1)

			# KL(p_bg || p_t) and KL(p_t || p_bg)
			kl_bg2t = F.kl_div(log_p_bg, p_t, reduction='none').sum(dim=-1)  # [B]
			kl_t2bg = F.kl_div(log_p_t,  p_bg, reduction='none').sum(dim=-1)  # [B]

			loss_dict['loss_dao_bg'] = float(kl_bg2t.mean().detach())
			loss_dict['loss_dao_t']  = float(kl_t2bg.mean().detach())

			loss_dao = (kl_bg2t + kl_t2bg).mean() * self.opts.dao_lambda
			total_loss += loss_dao

		# 4) Orthogonal projection loss
		if self.opts.ortho_lambda > 0:
			# normalize per‐channel
			cbn_bg = F.normalize(latent_bg_c, dim=1)   # [B,C,H,W]
			sbn_bg = F.normalize(latent_bg_s, dim=1)
			cbn_t  = F.normalize(latent_t_c, dim=1)
			sbn_t  = F.normalize(latent_t_s, dim=1)

			# inner‐product squared, then mean over (B,H,W)
			ortho_bg = ((cbn_bg * sbn_bg).sum(dim=1).pow(2)).mean()
			ortho_t  = ((cbn_t  * sbn_t ).sum(dim=1).pow(2)).mean()

			loss_dict['loss_ortho_bg'] = float(ortho_bg)
			loss_dict['loss_ortho_t']  = float(ortho_t)
			total_loss += self.opts.ortho_lambda * (ortho_bg + ortho_t)

		# 5) Attribute binary classification loss
		if self.opts.attr_lambda > 0:
			zeros = torch.zeros_like(logit_c_bg)
			ones  = torch.ones_like (logit_s_t)
			pos_w = torch.tensor([2.0], device=logit_s_t.device, dtype=logit_s_t.dtype)

			loss_attr_bg = 0.5 * (
				F.binary_cross_entropy_with_logits(logit_c_bg, zeros, reduction='mean') +
				F.binary_cross_entropy_with_logits(logit_s_bg, zeros, reduction='mean')
			)
			loss_attr_t = 0.5 * (
				F.binary_cross_entropy_with_logits(logit_c_t,  zeros, reduction='mean') +
				F.binary_cross_entropy_with_logits(logit_s_t,  ones,  reduction='mean', pos_weight=pos_w)
			)

			loss_dict['loss_attr_bg'] = float(loss_attr_bg)
			loss_dict['loss_attr_t']  = float(loss_attr_t)
			total_loss += (loss_attr_bg + loss_attr_t) * self.opts.attr_lambda
				
		loss_dict['loss'] = float(total_loss)
		return total_loss, loss_dict



	def merge_loss_dict(self, loss_lat_dict, loss_img_dict_bg, loss_img_dict_t):
		
		# loss_dict = loss_cs_dict | loss_dict_bg_pSp | loss_dict_t_pSp
		loss_dict = {}
		loss_dict['loss_lat'] = loss_lat_dict
		loss_dict['loss_img_bg'] = loss_img_dict_bg
		loss_dict['loss_img_t'] = loss_img_dict_t
		loss_dict['loss_sum'] = loss_lat_dict['loss'] + loss_img_dict_bg['loss'] + loss_img_dict_t['loss']
		
		return loss_dict

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)


	def __get_save_dict(self):
		save_dict = {
			'state_dict_cs_enc': self.cs_mlp_net.state_dict(),
			'opts': vars(self.opts),
			'global_step': self.global_step,
			'best_val_loss': self.best_val_loss,
		}

		if self.opts.train_diffu_encoder:
			save_dict['state_dict_encoder'] = self.diffu_encoder.state_dict()

		if self.opts.train_diffu_decoder:
			save_dict['state_dict_decoder'] = self.diffu_decoder.state_dict()

		# Optional: save EMA versions too, if using EMA
		if hasattr(self, 'ema_cs_mlp_net'):
			save_dict['state_dict_ema_cs_enc'] = self.ema_cs_mlp_net.state_dict()
		if self.opts.train_diffu_encoder and hasattr(self, 'ema_encoder'):
			save_dict['state_dict_ema_encoder'] = self.ema_encoder.state_dict()
		if self.opts.train_diffu_decoder and hasattr(self, 'ema_decoder'):
			save_dict['state_dict_ema_decoder'] = self.ema_decoder.state_dict()

		return save_dict