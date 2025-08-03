import os
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np

matplotlib.use('Agg')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from torch import nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# import tqdm
from utils import common, train_utils
from criteria import id_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from training.ranger import Ranger
from models.mlp3D import MappingNetwork_cs_noact_lastlayer, EqualizedLinear

class Coach_csmlp:
	def __init__(self, opts, previous_train_ckpt=None):
		self.opts = opts
		self.global_step = 0
		self.n_print = 0
		self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		self.opts.device = self.device

		self.seed_experiments(opts.seed)

		# Initialize network
		self.init_networks(previous_train_ckpt)

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.pSp_net.latent_avg is None:
			self.pSp_net.latent_avg = self.pSp_net.decoder.mean_latent(int(1e5))[0].detach()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:	
			self.id_loss = id_loss.IDLoss().to(self.device).eval()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_bg_dataset, self.train_glass_dataset, self.test_bg_dataset, self.test_glass_dataset = self.configure_datasets()

		self.train_bg_dataloader = DataLoader(self.train_bg_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		
		self.train_glass_dataloader = DataLoader(self.train_glass_dataset,
									batch_size=self.opts.batch_size,
									shuffle=True,
									num_workers=int(self.opts.workers),
									drop_last=True)
		
		self.test_bg_dataloader = DataLoader(self.test_bg_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)
		
		self.test_glass_dataloader = DataLoader(self.test_glass_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)		

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		#self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

		self.init_additional_params(previous_train_ckpt)
		
	def seed_experiments(self, seed):
		# Set the random seed for reproducibility
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)  # If you use multi-GPU.

		# Ensures deterministic behavior for some PyTorch operations
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		
	def init_networks(self, prev_train_ckpt):
		pSp_ckpt = None
		cs_ckpt = None
		if prev_train_ckpt is not None:
			pSp_ckpt = prev_train_ckpt['state_dict_pSp']
			cs_ckpt  = prev_train_ckpt['state_dict_cs_enc']	
			self.global_step = prev_train_ckpt['global_step'] + 1

		self.pSp_net = pSp(self.opts, pSp_ckpt).to(self.device)

		self.cs_mlp_net = MappingNetwork_cs_noact_lastlayer(self.opts).to(self.device)	
		
		if cs_ckpt is not None:
			print('Loading cs encoder from previous checkpoint...')
			self.cs_mlp_net.load_state_dict(cs_ckpt)
			print(f'Resuming training from step {self.global_step}')

	def	init_additional_params(self, prev_train_ckpt):
		if self.opts.save_training_data and prev_train_ckpt is not None:
			self.best_val_loss = prev_train_ckpt['best_val_loss']
			if self.opts.keep_optimizer :
				self.optimizer.load_state_dict(prev_train_ckpt['optimizer'])

	def shift_with_avg(self, codes):
			# normalize with respect to the center of an average face
		shifted_codes = codes + self.pSp_net.latent_avg.repeat(codes.shape[0], 1, 1)
		return shifted_codes

	def center_with_avg(self, codes):
			# normalize with respect to the center of an average face

		shifted_codes = codes - self.pSp_net.latent_avg.repeat(codes.shape[0], 1, 1)
		return shifted_codes

	def train(self):
		#self.pSp_net.train()
		self.cs_mlp_net.train()
		while self.global_step < self.opts.max_steps:

			for batch_idx, (batch_bg, batch_t) in enumerate(zip(self.train_bg_dataloader, self.train_glass_dataloader)):
				
				self.optimizer.zero_grad()

				x_bg, _ = batch_bg
				x_t, _ = batch_t

				x_bg, x_t = x_bg.to(self.device).float(), x_t.to(self.device).float()

				rec_x_bg_pSp, w_bg_pSp = self.pSp_net.forward(x_bg, return_latents=True)
				rec_x_t_pSp, w_t_pSp = self.pSp_net.forward(x_t, return_latents=True) 

				latent_bg_c, latent_bg_s = self.cs_mlp_net(w_bg_pSp, zero_out_silent=self.opts.zero_out_silent_bg)
				latent_t_c, latent_t_s = self.cs_mlp_net(w_t_pSp, zero_out_silent=self.opts.zero_out_silent_t) 
				
				latent_bg = latent_bg_c 
				latent_t = latent_t_c + latent_t_s

				rec_x_bg = self.pSp_net.forward(latent_bg, input_code=True, randomize_noise=True, recon_modle=True)
				rec_x_t = self.pSp_net.forward(latent_t, input_code=True, randomize_noise=True, recon_modle=True)	

				# Calculate loss 
				loss_lat, loss_lat_dict = self.calc_latent_loss(latent_bg_s, latent_t_s, latent_bg, latent_t, w_bg_pSp, w_t_pSp)
				loss_img_bg, loss_img_dict_bg, id_logs_bg = self.calc_image_loss(x_bg, rec_x_bg)
				loss_img_t, loss_img_dict_t, id_logs_t = self.calc_image_loss(x_t, rec_x_t)
				
				train_loss_dict = self.merge_loss_dict(loss_lat_dict, loss_img_dict_bg, loss_img_dict_t)

				loss = loss_lat + loss_img_bg + loss_img_t

				loss.backward()
				self.optimizer.step()
				
				# # Logging related
				# if self.global_step % self.opts.image_interval == 0 :
				# 	self.parse_and_log_images(id_logs_bg, x_bg, rec_x_bg_pSp, rec_x_bg, title='images/train/faces_bg')
				# 	self.parse_and_log_images(id_logs_t, x_t, rec_x_t_pSp, rec_x_t, title='images/train/faces_t')
				if self.global_step < 100 and self.global_step % 25 == 0 :
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
		#self.pSp_net.eval()
		self.cs_mlp_net.eval()

		agg_loss_lat_dict = []
		agg_loss_img_dict_bg = []
		agg_loss_img_dict_t = []

		# for batch_idx, batch in enumerate(self.test_dataloader):
		# 	x, y = batch
		for batch_idx, (batch_bg, batch_t) in enumerate(zip(self.test_bg_dataloader, self.test_glass_dataloader)):
			
			x_bg, _ = batch_bg
			x_t, _ = batch_t

			with torch.no_grad():

				x_bg, x_t = x_bg.to(self.device).float(), x_t.to(self.device).float()
				
				_, w_bg_pSp = self.pSp_net.forward(x_bg, return_latents=True)
				_, w_t_pSp = self.pSp_net.forward(x_t, return_latents=True) 

				latent_bg_c, latent_bg_s = self.cs_mlp_net(w_bg_pSp, zero_out_silent=self.opts.zero_out_silent_bg)
				latent_t_c, latent_t_s = self.cs_mlp_net(w_t_pSp, zero_out_silent=self.opts.zero_out_silent_t) 
				
				latent_bg = latent_bg_c 
				latent_t = latent_t_c + latent_t_s

				rec_x_bg = self.pSp_net.forward(latent_bg, input_code=True, randomize_noise=True, recon_modle=True)
				rec_x_t = self.pSp_net.forward(latent_t, input_code=True, randomize_noise=True, recon_modle=True)	

				# Calculate loss 
				_, loss_lat_dict = self.calc_latent_loss(latent_bg_s, latent_t_s, latent_bg, latent_t, w_bg_pSp, w_t_pSp)
				_, loss_img_dict_bg, id_logs_bg = self.calc_image_loss(x_bg, rec_x_bg)
				_, loss_img_dict_t, id_logs_t = self.calc_image_loss(x_t, rec_x_t)
				
			agg_loss_lat_dict.append(loss_lat_dict)
			agg_loss_img_dict_bg.append(loss_img_dict_bg)
			agg_loss_img_dict_t.append(loss_img_dict_t)

			# # # Logging related
			# if self.global_step % self.opts.image_interval == 0 :
			# 	self.parse_and_log_images(id_logs_bg, x_bg, rec_x_bg_pSp, rec_x_bg, title='images/test/faces_bg', subscript='{:04d}'.format(batch_idx))
			# 	self.parse_and_log_images(id_logs_t, x_t, rec_x_t_pSp, rec_x_t, title='images/test/faces_t', subscript='{:04d}'.format(batch_idx))

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				#self.pSp_net.train()
				self.cs_mlp_net.train()
				return None  # Do not log, inaccurate in first batch

		loss_lat_dict = train_utils.aggregate_loss_dict(agg_loss_lat_dict)
		loss_img_dict_bg = train_utils.aggregate_loss_dict(agg_loss_img_dict_bg)
		loss_img_dict_t = train_utils.aggregate_loss_dict(agg_loss_img_dict_t)

		loss_dict = self.merge_loss_dict(loss_lat_dict, loss_img_dict_bg, loss_img_dict_t)

		#self.log_metrics(loss_dict, prefix='test')
		# self.print_metrics(loss_dict, prefix='test')

		#self.pSp_net.train()
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
				if self.opts.use_wandb:
					self.wb_logger.log_best_model()
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def write_metrics_to_txt(self, metrics_dict, prefix, filename):

		with open(os.path.join(self.checkpoint_dir, filename), 'a') as f:
			f.write(f'Metrics for {prefix}, Step - {self.global_step}')
			f.write(f'\n{metrics_dict}\n')		

	# def print_metrics(self, metrics_dict, prefix):
	# 	print(f'Metrics for {prefix}, step {self.global_step}')
	# 	for key, value in metrics_dict.items():
	# 		print(f'\t{key} = ', value)

	def configure_optimizers(self):
		params = list(self.cs_mlp_net.parameters())
		if self.opts.train_pSp_encoder:
			params += list(self.pSp_net.encoder.parameters())
		if self.opts.train_stylegan_decoder:
			params += list(self.pSp_net.decoder.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()

		train_bg_dataset = ImagesDataset(source_root=dataset_args['train_bg_source_root'],
									  target_root=dataset_args['train_bg_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
		
		train_glass_dataset = ImagesDataset(source_root=dataset_args['train_glass_source_root'],
									  target_root=dataset_args['train_glass_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
				
		test_bg_dataset = ImagesDataset(source_root=dataset_args['test_bg_source_root'],
									 target_root=dataset_args['test_bg_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts)
		
		test_glass_dataset = ImagesDataset(source_root=dataset_args['test_glass_source_root'],
									 target_root=dataset_args['test_glass_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts)
				
		if self.opts.use_wandb:
			self.wb_logger.log_dataset_wandb(train_bg_dataset, dataset_name="Train_bg")
			self.wb_logger.log_dataset_wandb(train_glass_dataset, dataset_name="Train_glass")
			self.wb_logger.log_dataset_wandb(test_bg_dataset, dataset_name="Test_bg")
			self.wb_logger.log_dataset_wandb(test_glass_dataset, dataset_name="Test_glass")

		print(f"Number of training samples: {len(train_bg_dataset)}")
		print(f"Number of test samples: {len(test_bg_dataset)}")
		return train_bg_dataset, train_glass_dataset, test_bg_dataset, test_glass_dataset
	
	def output_regularization(self, output, direction='element', reg_type='traditional', lambda_reg=0.1, a=3.7, gamma=3, threshold=0.0, epsilon=1e-8):
		"""
		Apply different types of regularization (Traditional, Logarithmic, Adaptive, Thresholded, SCAD, MCP) on the output tensor.

		Args:
			output (torch.Tensor): The tensor on which to apply regularization.
			direction (str): The direction for regularization ('row', 'column', or 'element').
			reg_type (str): Type of regularization ('traditional', 'logarithmic', 'adaptive', 'thresholded', 'SCAD', 'MCP').
			lambda_reg (float): Regularization strength (threshold for shrinkage).
			a (float): SCAD parameter controlling the threshold for minimal penalization (only used if reg_type='SCAD').
			gamma (float): MCP parameter controlling the threshold for minimal penalization (only used if reg_type='MCP').
			threshold (float): Threshold for Thresholded L1 regularization (only used if reg_type='thresholded').
			epsilon (float): Small constant to prevent division by zero in Adaptive L1 regularization.

		Returns:
			torch.Tensor: The computed regularization penalty.
		"""
		abs_output = torch.abs(output)

		if reg_type == 'traditional':
			regularization_fn = torch.abs
		elif reg_type == 'logarithmic':
			regularization_fn = lambda x: torch.log(1 + torch.abs(x))
		elif reg_type == 'adaptive':
			regularization_fn = lambda x: (1.0 / (torch.abs(x) + epsilon)) * torch.abs(x)
		elif reg_type == 'thresholded':
			regularization_fn = lambda x: torch.where(torch.abs(x) < threshold, torch.abs(x), torch.tensor(0.0, device=x.device))
		elif reg_type == 'SCAD':
			# SCAD penalty function
			def scad_penalty(x):
				penalty = torch.zeros_like(x)
				mask1 = abs_output <= lambda_reg
				penalty[mask1] = lambda_reg * abs_output[mask1]
				mask2 = (abs_output > lambda_reg) & (abs_output <= a * lambda_reg)
				penalty[mask2] = (-x[mask2]**2 + 2 * a * lambda_reg * abs_output[mask2] - lambda_reg**2) / (2 * (a - 1))
				mask3 = abs_output > a * lambda_reg
				penalty[mask3] = (a + 1) * lambda_reg**2 / 2
				return penalty
			regularization_fn = scad_penalty
		elif reg_type == 'MCP':
			# MCP penalty function
			def mcp_penalty(x):
				penalty = torch.zeros_like(x)
				mask1 = abs_output <= gamma * lambda_reg
				penalty[mask1] = lambda_reg * abs_output[mask1] - (x[mask1]**2) / (2 * gamma)
				mask2 = abs_output > gamma * lambda_reg
				penalty[mask2] = (gamma * lambda_reg**2) / 2
				return penalty
			regularization_fn = mcp_penalty
		else:
			raise ValueError(f"Invalid regularization type: '{reg_type}'.")

		# Apply the selected regularization function based on direction
		if direction == 'row':
			return torch.sum(regularization_fn(output).sum(dim=2))
		elif direction == 'column':
			return torch.sum(regularization_fn(output).sum(dim=1))
		elif direction == 'element':
			return torch.sum(regularization_fn(output))
		else:
			raise ValueError(f"Invalid direction: '{direction}'.")


	def elastic_net_regularization(self, model, network='net_s', reg_type='all', alpha=0.5):
		"""
		Apply Elastic Net regularization on the specified layers of either `net_s` or `net_c` in the model, 
		handling both 2D and 3D weight matrices.

		Args:
			model (MappingNetwork_cs_sparsity): The model with the network to regularize.
			network (str): The network to apply Elastic Net regularization ('net_s' or 'net_c').
			reg_type (str): Type of weight regularization ('all' or 'last').
				- 'all': Applies Elastic Net to all EqualizedLinear layers in the specified network.
				- 'last': Applies Elastic Net only to the last EqualizedLinear layer in the specified network.
			alpha (float): The balance between L1 and L2 regularization. 0.0 corresponds to pure L2 (Ridge),
						1.0 to pure L1 (Lasso), and values in between apply both.

		Returns:
			torch.Tensor: Computed Elastic Net regularization penalty for the selected layers.
		"""
		elastic_net_penalty = 0.0

		# Select the target network based on the network argument
		target_network = getattr(model, network, None)
		if target_network is None:
			raise ValueError(f"Invalid network '{network}'. Use 'net_s' or 'net_c'.")

		# Get layers of the selected network
		layers = list(target_network.children())

		# Apply Elastic Net regularization based on reg_type
		if reg_type == 'all':
			# Apply Elastic Net to all EqualizedLinear layers in the target network
			for layer in layers:
				if isinstance(layer[0], EqualizedLinear):
					weights = layer[0].weight.weight
					l1_penalty = torch.sum(torch.abs(weights))
					l2_penalty = torch.sum(weights ** 2)
					elastic_net_penalty += alpha * l1_penalty + (1 - alpha) * l2_penalty
		elif reg_type == 'last':
			# Apply Elastic Net only to the last EqualizedLinear layer in the target network
			last_layer = layers[-1][0]
			if isinstance(last_layer, EqualizedLinear):
				weights = last_layer.weight.weight
				l1_penalty = torch.sum(torch.abs(weights))
				l2_penalty = torch.sum(weights ** 2)
				elastic_net_penalty += alpha * l1_penalty + (1 - alpha) * l2_penalty
		else:
			raise ValueError(f"Invalid reg_type '{reg_type}'. Use 'all' or 'last'.")

		return elastic_net_penalty

	
	def calc_image_loss(self, x, x_hat):
		loss_dict = {}
		loss = 0.0
		id_logs = None

		# Calculate pSp id, rec, lpips losses for images
		if self.opts.id_lambda > 0:			
			loss_id, sim_improvement, id_logs = self.id_loss(x_hat, x, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda

		if self.opts.pix_lambda > 0:
			loss_pix = F.mse_loss(x, x_hat)
			loss_dict['loss_pix'] = float(loss_pix)
			loss += loss_pix * self.opts.pix_lambda

		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(x_hat, x)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda

		loss_dict['loss'] = float(loss)

		return loss, loss_dict, id_logs
	
	def calc_latent_loss(self, latent_bg_s, latent_t_s, latent_bg, latent_t, latent_bg_target, latent_t_target):
		"""
		Calculate total loss, including L1 regularization for output and weights.

		Args:
			latent_bg_s: Background latent tensor.
			latent_t_s: Target latent tensor.
			latent_bg: Background tensor.
			latent_t: Target tensor.
			latent_bg_target: Target tensor for background.
			latent_t_target: Target tensor for output.
			lasso_sbg_lambda: Weight for output L1 regularization.
			lasso_st_lambda: Weight for output L1 regularization.
			lasso_weight_lambda: Weight for weight L1 regularization.
			model: The model on which to apply regularization.
			lasso_output_type: Type of L1 regularization for output ('row', 'column', 'element').
			lasso_weight_type: Type of L1 regularization for weights ('all', 'last').
			network: Specifies 'net_s' or 'net_c' for weight regularization.

		Returns:
			total_loss: Computed total loss.
			loss_dict: Dictionary with individual loss components.
		"""
		loss_dict = {}
		loss = 0.0

		if self.opts.sbg_lambda > 0:
			if self.opts.sbg_type == 'l2':
				loss_silent_bg = F.mse_loss(latent_bg_s, torch.zeros_like(latent_bg_s).to(self.device))
			elif self.opts.sbg_type == 'fro':
				loss_silent_bg = torch.norm(latent_bg_s, p='fro')  # Frobenius norm
			loss_dict['loss_silent_bg'] = float(loss_silent_bg)
			loss += loss_silent_bg * self.opts.sbg_lambda
			

		if self.opts.w_dist_lambda > 0:
			loss_distance_bg = F.mse_loss(latent_bg, latent_bg_target)
			loss_distance_t = F.mse_loss(latent_t, latent_t_target)
			loss_dict['loss_distance_bg'] = float(loss_distance_bg)
			loss_dict['loss_distance_t'] = float(loss_distance_t)
			loss += (loss_distance_bg + loss_distance_t) * self.opts.w_dist_lambda	

		if self.opts.lasso_sbg_lambda > 0:
			lasso_sbg_loss = self.output_regularization(latent_bg_s, direction=self.opts.lasso_direction, reg_type=self.opts.lasso_reg_type, threshold=self.opts.lasso_threshold)
			loss_dict['lasso_sbg_loss'] = float(lasso_sbg_loss)  #  direction=self.opts.lasso_direction, reg_type=self.opts.lasso_reg_type, threshold=self.opts.lasso_threshold
			loss += lasso_sbg_loss * self.opts.lasso_sbg_lambda

		if self.opts.lasso_st_lambda > 0:
			lasso_st_loss = self.output_regularization(latent_t_s, direction=self.opts.lasso_direction, reg_type=self.opts.lasso_reg_type, threshold=self.opts.lasso_threshold)
			loss_dict['lasso_st_loss'] = float(lasso_st_loss)
			loss += lasso_st_loss * self.opts.lasso_st_lambda

		with torch.no_grad():	
			zero_count_bg = (latent_bg_s.abs() <= 1e-7).sum().item()	
			zero_count_t = (latent_t_s.abs() <= 1e-7).sum().item()
			total_count = latent_t_s.numel()
			loss_dict['zero_count_sbg'] = int(zero_count_bg)
			loss_dict['zero_count_st'] = int(zero_count_t)
			loss_dict['total_count_s'] = int(total_count)

		if self.opts.lasso_weight_lambda > 0:
			lasso_weight_loss = (self.elastic_net_regularization(self.cs_mlp_net, network='net_s', reg_type=self.opts.lasso_weight_type, alpha=self.opts.elastic_alpha) 
				if self.cs_mlp_net else 0.0)
			loss_dict['lasso_weight_loss'] = float(lasso_weight_loss)
			loss += lasso_weight_loss * self.opts.lasso_weight_lambda

		loss_dict['loss'] = float(loss)
		return loss, loss_dict		


	def merge_loss_dict(self, loss_lat_dict, loss_img_dict_bg, loss_img_dict_t):
		
		# loss_dict = loss_cs_dict | loss_dict_bg_pSp | loss_dict_t_pSp
		loss_dict = {}
		loss_dict['loss_lat'] = loss_lat_dict
		loss_dict['loss_img_bg'] = loss_img_dict_bg
		loss_dict['loss_img_t'] = loss_img_dict_t
		loss_dict['loss_sum'] = loss_lat_dict['loss'] + loss_img_dict_bg['loss'] + loss_img_dict_t['loss']
		
		return loss_dict
		

	# def log_metrics(self, metrics_dict, prefix):
	# 	for key, value in metrics_dict.items():
	# 		self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
	# 	if self.opts.use_wandb:
	# 		self.wb_logger.log(prefix, metrics_dict, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	# def parse_and_log_images(self, id_logs, x, y_pSp, y_ours, title, subscript=None, display_count=2):
	# 	im_data = []
	# 	for i in range(display_count):
	# 		cur_im_data = {
	# 			'input_face': common.log_input_image(x[i], self.opts),
	# 			'pSp_output_face': common.tensor2im(y_pSp[i]),
	# 			'Our_output_face': common.tensor2im(y_ours[i])
	# 		}
	# 		if id_logs is not None:
	# 			for key in id_logs[i]:
	# 				cur_im_data[key] = id_logs[i][key]
	# 		im_data.append(cur_im_data)
	# 	self.log_images(title, im_data=im_data, subscript=subscript)

	# def log_images(self, name, im_data, subscript=None, log_latest=False):
	# 	fig = common.vis_faces(im_data)
	# 	step = self.global_step
	# 	if log_latest:
	# 		step = 0
	# 	if subscript:
	# 		path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
	# 	else:
	# 		path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
	# 	os.makedirs(os.path.dirname(path), exist_ok=True)
	# 	fig.savefig(path)
	# 	plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			#'state_dict_pSp': self.pSp_net.state_dict(),
			'state_dict_cs_enc': self.cs_mlp_net.state_dict(),
			'opts': vars(self.opts)
		}
		if self.opts.save_training_data:
			save_dict['global_step'] = self.global_step
			save_dict['optimizer'] = self.optimizer.state_dict()
			save_dict['best_val_loss'] = self.best_val_loss
				
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.opts.start_from_latent_avg:
			save_dict['latent_avg'] = self.pSp_net.latent_avg
		return save_dict
	
		
