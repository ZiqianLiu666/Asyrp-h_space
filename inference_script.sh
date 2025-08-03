#!/bin/bash
#SBATCH --job-name=3500_999_12sf
#SBATCH --nodes=1
#SBATCH --partition=A100,H100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40GB
#SBATCH --time=24:00:00
#SBATCH --output=./reports/infer111_out.txt
#SBATCH --error=./reports/infer111_error.txt

conda activate stylegan

# --exp_scheme=ema
python training_scripts/inference.py \
--exp_scheme=specific_encoder_fused \
--results_dir=final_results_infer/ffhq_glasses/specific_encoder/inv=100/fused/t_edit999/conv/layer_12/lr0.00001/check_3500 \
--cs_net_type=specific_conv_fused \
--cs_model_weights /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/results/Single_Salient/eachpred_weighted_imageloss/glasses/t_edit=999/inv=100/fused/specific_encoder/layer_12/conv/lr0.00001/checkpoints/iteration_3500.pt \
--n_cs_layers=12 \
--t_end_edit 0 \
--dataset_type=ffhq_glasses \
--diffu_weights=pretrained_models/ffhq_p2.pt \
--diffu_config_path=configs/ffhq.yml \
--n_inv_step=100 \
--batch_size=16 \
--dataset_root=/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN

# --dataset_root=/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN

# final_results_infer/oneforC_oneforS/no_fused/t_edit400/conv
                    # b # baseline # equalized_mlp # conv
# final_results_infer/oneforC_oneforS/fused/t_edit400/conv
                    # bf # baseline_fused # equalized_mlp_fused # conv_fused

# final_results_infer/specific_encoder/no_fused/t_edit400/conv
                    # s # specific_encoder # specific_mlp # specific_conv
# final_results_infer/specific_encoder/fused/t_edit400/conv
                    # sf # specific_encoder_fused # specific_mlp_fused # specific_conv_fused


# age
# dataset_type=ffhq_age
# diffu_weights=pretrained_models/ffhq_p2.pt
# diffu_config_path=configs/ffhq.yml

# gender
# dataset_type=celeba_gender
# diffu_weights=pretrained_models/celebahq_p2.pt
# diffu_config_path=configs/celeba_p2.yml

# smile
# dataset_type=celeba_smile
# diffu_weights=pretrained_models/celebahq_p2.pt
# diffu_config_path=configs/celeba_p2.yml