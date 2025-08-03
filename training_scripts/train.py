"""
Main training and validation loop.
"""

import os
import sys
import shutil
import json
import pprint
import torch
from argparse import Namespace

# Setup environment and system paths
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(".")
sys.path.append("..")
from options.train_options import TrainOptions
from configs.paths_config import MODEL_PATHS


def main():
    # Load checkpoint or start fresh
    if MODEL_PATHS['previous_ckpt_path'] is not None:
        ckpt_path = MODEL_PATHS['previous_ckpt_path']
        print(f"Resuming training from checkpoint: {ckpt_path}")
        prev_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        opts = load_opts_from_checkpoint(prev_ckpt)
    else:
        opts = TrainOptions().parse()
        create_initial_experiment_dir(opts)

    # Load appropriate training coach
    if opts.exp_scheme == 'baseline':
        print("Using coach_csmlp_baseline.py...")
        from training.coach_cs_asyrp_baseline import Coach
    elif opts.exp_scheme == 'baseline_fused':
        print("Using coach_csmlp_baseline.py...")
        from training.coach_cs_asyrp_baseline_fused import Coach
        
    elif opts.exp_scheme == 'specific_encoder':
        print("Using coach_csmlp_baseline.py...")
        from training.specific_single_salient import Coach
    elif opts.exp_scheme == 'specific_encoder_fused':
        print("Using coach_csmlp_baseline.py...")
        from training.specific_single_salient_fused import Coach
        
    elif opts.exp_scheme == 'one_encoder':
        print("Using coach_csmlp_baseline.py...")
        from training.one_encoder_single_salient import Coach
    elif opts.exp_scheme == 'one_encoder_fused':
        print("Using coach_csmlp_baseline.py...")
        from training.one_encoder_single_salient_fused import Coach
        
    elif opts.exp_scheme == 'test_denoise_step':
        print("Using coach_csmlp_baseline.py...")
        from training.test_denoise_step import Coach
        
    else:
        raise ValueError(f"Unsupported experiment scheme: {opts.exp_scheme}")


    # Start training
    coach = Coach(opts)
    coach.train()


def load_opts_from_checkpoint(previous_train_ckpt):
    opts = Namespace(**previous_train_ckpt['opts'])
    print("Loaded options from checkpoint:")
    pprint.pprint(vars(opts))
    return opts


def create_initial_experiment_dir(opts):
    if os.path.exists(opts.results_dir):
        shutil.rmtree(opts.results_dir)
    os.makedirs(opts.results_dir)

    print("Created experiment directory with options:")
    opts_dict = vars(opts)
    pprint.pprint(opts_dict)

    with open(os.path.join(opts.results_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
