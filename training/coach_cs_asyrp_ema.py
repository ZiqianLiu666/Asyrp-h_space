import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import os
from training.ranger import Ranger
from utils.mics import seed_all, aggregate_loss_dict
from utils.model_utils import load_diffusion_model, load_cs_model, load_id_lpips_models
from utils.data_utils import build_dataloaders
from utils.visual_utils import visualize_batch_grid
from utils.config_utils import load_diffusion_config
import torch.nn.functional as F
import copy


class Coach:

	def __init__(self, opts):
		self.opts = opts
		self.global_step = 0
		self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.diff_configs = load_diffusion_config(opts.diffu_config_path)
		self.n_print = 0
		seed_all(opts.seed)
		self.best_val_loss = None
		###### Load pretrained diffusion models ######
		self.diffu_model, self.diffu_encoder, self.diffu_decoder, self.diffu_info = load_diffusion_model(
			opts=opts,
			config=self.diff_configs,
			device=self.device,
			is_eval=True
		)

		###### Load other networks ######
		self.cs_mlp_net = load_cs_model(opts.cs_model_weights, opts, self.device)
		self.lpips_loss, self.id_loss = load_id_lpips_models(opts, self.device)
		self.initialize_ema_models()
		
		###### Load datasets & optimizers ######
		self.train_bg_dataloader, self.train_t_dataloader, self.test_bg_dataloader, self.test_t_dataloader = build_dataloaders(opts)
		self.optimizer = self.configure_optimizers()

		###### create folders for logging ######
		self.log_dir = os.path.join(opts.results_dir, 'logs')
		self.checkpoint_dir = os.path.join(opts.results_dir, 'checkpoints')
		os.makedirs(self.log_dir, exist_ok=True)
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		os.makedirs(os.path.join(self.log_dir, 'images'), exist_ok=True)


	def noise_injection(self, batch_image):
		batch_xT = self.diffu_info['recon'].inject_noise_batch(
			self.diffu_model,
			batch_image,
			self.diffu_info['seq_inv'],
			self.diffu_info['seq_inv_next']
		)
		t_start = torch.full((batch_xT.size(0),), self.diffu_info['seq_inv'][-1], device=self.device)
		middle_h, _, _ = self.diffu_encoder(batch_xT, t_start)
		return batch_xT, middle_h

	def forward_generation(self, batch_xT, middle_h_modified=None):
		x = batch_xT
		recon = self.diffu_info['recon']

		with torch.no_grad():
			for step_idx, (i, j) in enumerate(zip(
				reversed(self.diffu_info['seq_inv']),
				reversed(self.diffu_info['seq_inv_next']))):
				
				t = torch.full((x.size(0),), i, device=self.device)
				t_next = torch.full((x.size(0),), j, device=self.device)

				if step_idx == 0 and middle_h_modified is not None:
					_, hs, emb = self.diffu_encoder(x, t)
					middle_h = middle_h_modified
				else:
					middle_h, hs, emb = self.diffu_encoder(x, t)

				et, et_modified = self.diffu_decoder(middle_h, hs, emb, x)

				x, _ = recon.denoising_step_enc_dec(
					xt=x, t=t, t_next=t_next, models=self.diffu_model,
					et=et, et_modified=et_modified,
					logvars=self.diffu_info['logvar'],
					sampling_type=self.diffu_info['sample_type'],
					b=self.diffu_info['betas'],
					learn_sigma=self.diffu_info['learn_sigma']
				)

		return x

	def initialize_ema_models(self):
		self.ema_decay = 0.999  # You can make this configurable via opts if needed

		# EMA for cs_mlp_net
		self.ema_cs_mlp_net = copy.deepcopy(self.cs_mlp_net)
		for p in self.ema_cs_mlp_net.parameters():
			p.requires_grad = False

		# EMA for encoder
		if self.opts.train_diffu_encoder:
			self.ema_encoder = copy.deepcopy(self.diffu_encoder)
			for p in self.ema_encoder.parameters():
				p.requires_grad = False

		# EMA for decoder
		if self.opts.train_diffu_decoder:
			self.ema_decoder = copy.deepcopy(self.diffu_decoder)
			for p in self.ema_decoder.parameters():
				p.requires_grad = False


	@torch.no_grad()
	def update_ema_model(self, model, ema_model, decay):
		for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
			ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
				
	def configure_optimizers(self):
		params = list(self.cs_mlp_net.parameters())
		if self.opts.train_diffu_encoder:
			params += list(self.diffu_encoder.parameters())
		if self.opts.train_diffu_decoder:
			params += list(self.diffu_decoder.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_optimizers_multilr(self):
		param_groups = [
			{"params": self.cs_mlp_net.parameters(), "lr": self.opts.learning_rate},
		]

		if self.opts.train_diffu_encoder:
			param_groups.append({
				"params": self.diffu_encoder.parameters(),
				"lr": self.opts.diffu_lr,
				"weight_decay": 0.0
			})

		if self.opts.train_diffu_decoder:
			param_groups.append({
				"params": self.diffu_decoder.parameters(),
				"lr": self.opts.diffu_lr,
				"weight_decay": 0.0
			})

		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(param_groups)
		else:
			optimizer = Ranger(param_groups)

		return optimizer

	def configure_multi_optimizers(self):
		param_groups = [
			{"params": self.cs_mlp_net.parameters(), "lr": self.opts.learning_rate},
		]

		if self.opts.train_diffu_encoder:
			param_groups.append({
				"params": self.diffu_encoder.parameters(),
				"lr": self.opts.encoder_lr if hasattr(self.opts, 'encoder_lr') else self.opts.diffu_lr
			})

		if self.opts.train_diffu_decoder:
			param_groups.append({
				"params": self.diffu_decoder.parameters(),
				"lr": self.opts.decoder_lr if hasattr(self.opts, 'decoder_lr') else self.opts.diffu_lr
			})

		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(param_groups)
		else:
			optimizer = Ranger(param_groups)

		return optimizer
	

	
	def print_cuda_memory(self, tag=''):
		torch.cuda.synchronize()
		allocated = torch.cuda.memory_allocated(self.device) / 1024**2
		reserved = torch.cuda.memory_reserved(self.device) / 1024**2
		print(f"[{tag}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

	def train(self):
		self.cs_mlp_net.train()
		while self.global_step < self.opts.max_steps:

			for batch_idx, (batch_bg, batch_t) in enumerate(zip(self.train_bg_dataloader, self.train_t_dataloader)):

				self.optimizer.zero_grad()

				x_bg, x_t = batch_bg.to(self.device).float(), batch_t.to(self.device).float()

				batch_xT_bg, middle_h_bg = self.noise_injection(x_bg)
				batch_xT_t, middle_h_t = self.noise_injection(x_t)

				diffu_x_bg = self.forward_generation(batch_xT_bg, middle_h_modified=None)
				diffu_x_t = self.forward_generation(batch_xT_t, middle_h_modified=None)	

				latent_bg_c, latent_bg_s = self.cs_mlp_net(middle_h_bg)
				latent_t_c, latent_t_s = self.cs_mlp_net(middle_h_t) 

				rec_x_bg = self.forward_generation(batch_xT_bg, middle_h_modified=latent_bg_c)
				rec_x_t = self.forward_generation(batch_xT_t, middle_h_modified=latent_t_c + latent_t_s)	

				# Calculate loss
				loss_lat, loss_lat_dict = self.calc_latent_loss(latent_bg_c, latent_bg_s, latent_t_c, latent_t_s, middle_h_bg, middle_h_t)
				loss_img_bg, loss_img_dict_bg, id_logs_bg = self.calc_image_loss(x_bg, rec_x_bg)
				loss_img_t, loss_img_dict_t, id_logs_t = self.calc_image_loss(x_t, rec_x_t)

				train_loss_dict = self.merge_loss_dict(loss_lat_dict, loss_img_dict_bg, loss_img_dict_t)
				loss = loss_lat + loss_img_bg + loss_img_t

				loss.backward()
				self.optimizer.step()
				# EMA updates
				self.update_ema_model(self.cs_mlp_net, self.ema_cs_mlp_net, self.ema_decay)	

				if self.opts.train_diffu_encoder:
					self.update_ema_model(self.diffu_encoder, self.ema_encoder, self.ema_decay)

				if self.opts.train_diffu_decoder:
					self.update_ema_model(self.diffu_decoder, self.ema_decoder, self.ema_decay)	

				# Logging related
				if self.global_step % self.opts.image_interval == 0 :

					# with torch.no_grad():
					# 	swap_x_bg = self.forward_generation(batch_xT_bg, middle_h_modified=latent_bg_c + latent_t_s)
					# 	swap_x_t = self.forward_generation(batch_xT_t, middle_h_modified=latent_t_c )	

					visualize_batch_grid(
						image_batches=[
							torch.stack([x_bg[0], x_t[0]], dim=0),       # Originals
							torch.stack([diffu_x_bg[0], diffu_x_t[0]], dim=0),   # diffu recon
							torch.stack([rec_x_bg[0], rec_x_t[0]], dim=0),   # Reconstructed
							# torch.stack([swap_x_bg[0], swap_x_t[0]], dim=0),   # swap
						],
						titles=["Input", "h-space", "c + s"],
						save_path=f"{self.log_dir}/images/train_step_{self.global_step}.png"
					)

				if self.global_step < 300 and self.global_step % 25 == 0 :
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
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(train_loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self):
		# Temporarily use EMA models
		original_cs_mlp = self.cs_mlp_net
		original_encoder = self.diffu_encoder
		original_decoder = self.diffu_decoder

		self.cs_mlp_net = self.ema_cs_mlp_net
		if self.opts.train_diffu_encoder:
			self.diffu_encoder = self.ema_encoder
		if self.opts.train_diffu_decoder:
			self.diffu_decoder = self.ema_decoder

		self.cs_mlp_net.eval()

		agg_loss_lat_dict = []
		agg_loss_img_dict_bg = []
		agg_loss_img_dict_t = []

		# for batch_idx, batch in enumerate(self.test_dataloader):
		# 	x, y = batch
		for batch_idx, (batch_bg, batch_t) in enumerate(zip(self.test_bg_dataloader, self.test_t_dataloader)):

			with torch.no_grad():

				x_bg, x_t = batch_bg.to(self.device).float(), batch_t.to(self.device).float()

				batch_xT_bg, middle_h_bg = self.noise_injection(x_bg)
				batch_xT_t, middle_h_t = self.noise_injection(x_t)

				latent_bg_c, latent_bg_s = self.cs_mlp_net(middle_h_bg)
				latent_t_c, latent_t_s = self.cs_mlp_net(middle_h_t) 

				rec_x_bg = self.forward_generation(batch_xT_bg, middle_h_modified=latent_bg_c)
				rec_x_t = self.forward_generation(batch_xT_t, middle_h_modified=latent_t_c + latent_t_s)	

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

		# Restore original models
		self.cs_mlp_net = original_cs_mlp
		self.diffu_encoder = original_encoder
		self.diffu_decoder = original_decoder

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

	
	def calc_image_loss(self, x, x_hat):
		"""
		Compute image-based losses: ID loss, pixel-wise MSE, and LPIPS perceptual loss.

		Args:
			x: Ground truth images (B, C, H, W)
			x_hat: Reconstructed/generated images (B, C, H, W)

		Returns:
			total_loss: Weighted sum of selected image losses
			loss_dict: Dict with each individual loss component
			id_logs: Logs from ID loss for additional inspection
		"""
		loss_dict = {}
		id_logs = None
		total_loss = 0.0

		if self.opts.id_lambda > 0:
			loss_id, sim_improvement, id_logs = self.id_loss(x_hat, x, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			total_loss += loss_id * self.opts.id_lambda

		if self.opts.pix_lambda > 0:
			loss_pix = F.mse_loss(x_hat, x)
			loss_dict['loss_pix'] = float(loss_pix)
			total_loss += loss_pix * self.opts.pix_lambda

		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(x_hat, x)
			loss_dict['loss_lpips'] = float(loss_lpips)
			total_loss += loss_lpips * self.opts.lpips_lambda

		loss_dict['loss'] = float(total_loss)
		return total_loss, loss_dict, id_logs



	def calc_latent_loss(self, latent_bg_c, latent_bg_s, latent_t_c, latent_t_s, w_bg_pSp, w_t_pSp):

		loss_dict = {}
		total_loss = 0.0

		# Background silence loss
		if self.opts.sbg_lambda > 0:
			loss_sbg = F.mse_loss(latent_bg_s, torch.zeros_like(latent_bg_s))

			loss_dict['loss_silent_bg'] = float(loss_sbg)
			total_loss += loss_sbg * self.opts.sbg_lambda

		# Latent reconstruction distance loss
		if self.opts.lat_lambda > 0:
			loss_bg = F.mse_loss(latent_bg_c, w_bg_pSp)
			loss_t = F.mse_loss(latent_t_c + latent_t_s, w_t_pSp)

			loss_dict['loss_distance_bg'] = float(loss_bg)
			loss_dict['loss_distance_t'] = float(loss_t)
			total_loss += (loss_bg + loss_t) * self.opts.lat_lambda

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

	
		
