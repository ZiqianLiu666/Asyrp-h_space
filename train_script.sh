#!/bin/bash
#SBATCH --job-name=400_12sf
#SBATCH --nodes=1
#SBATCH --partition=A40
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40GB
#SBATCH --time=999:00:00
#SBATCH --output=./reports/save_output.txt
#SBATCH --error=./reports/save_error.txt

source ./set_cuda_IDS.sh # Set cuda dependencies for Jeanzay

### source environments ###
conda activate stylegan_liu  # if using IDS cluster

# --exp_scheme=ema
python training_scripts/train.py \
--exp_scheme=specific_encoder_fused \
--results_dir=results/Single_Salient/eachpred_weighted_imageloss/glasses/t_edit=400/inv=100/fused/specific_encoder/layer_12/conv/lr0.00001 \
--cs_net_type=specific_conv_fused \
--n_cs_layers=12 \
--t_end_edit 400 \
--n_inv_step=100 \
--lr_mlp=0.00001 \
--dataset_type=ffhq_glasses \
--diffu_weights=pretrained_models/ffhq_p2.pt \
--diffu_config_path=configs/ffhq.yml \
--image_interval=50 \
--val_interval=9999999 \
--save_interval=100 \
--batch_size 16 \
--dataset_root /home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN

# results/Single_Salient/eachpred_weighted_imageloss/t_edit=400/no_fused/oneforC_oneforS
                    # b # baseline # equalized_mlp # conv
# results/Single_Salient/eachpred_weighted_imageloss/t_edit=400/fused/oneforC_oneforS
                    # bf # baseline_fused # equalized_mlp_fused # conv_fused

# results/Single_Salient/eachpred_weighted_imageloss/glasses/t_edit=999/inv=100/no_fused/specific_encoder
                    # s # specific_encoder # specific_mlp # specific_conv
# results/Single_Salient/eachpred_weighted_imageloss/glasses/t_edit=999/inv=100/fused/specific_encoder
                    # sf # specific_encoder_fused # specific_mlp_fused # specific_conv_fused

# 400 or 0