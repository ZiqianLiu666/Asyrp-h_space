"""
This file runs the main training/val loop
"""
import os
import shutil
import json
import sys
import pprint
import torch
sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach import Coach
from configs.paths_config import model_paths
from argparse import Namespace

def main():
	if model_paths['previous_train_ckpt_path']:
		previous_train_ckpt = torch.load(model_paths['previous_train_ckpt_path'], map_location='cpu', weights_only=True)
		opts = previous_train_ckpt['opts']
		if 'learn_in_w' not in opts:
			opts['learn_in_w'] = False
		if 'output_size' not in opts:
			opts['output_size'] = 1024
		opts = Namespace(**opts)	
	else:
		opts = TrainOptions().parse()
		create_initial_experiment_dir(opts)

def create_initial_experiment_dir(opts):
	if os.path.exists(opts.exp_dir):
		shutil.rmtree(opts.exp_dir)
		#raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)


	coach = Coach(opts)
	coach.train()


if __name__ == '__main__':
	main()
