#!/bin/bash
#SBATCH --job-name=999_12s
#SBATCH --nodes=1
#SBATCH --partition=L40S,A100,H100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40GB
#SBATCH --time=24:00:00
#SBATCH --output=./reports/infer1111_out.txt
#SBATCH --error=./reports/infer1111_error.txt

python training_scripts/inference_pick_best.py \
--exp_scheme=specific_encoder \
--results_dir=results_infer_special/ffhq-glasses/t_edit=999/inv=100/no_fused/specific/layer=12/conv/lr0.00001 \
--cs_net_type=specific_conv \
--cs_model_weights /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/results/Single_Salient/eachpred_weighted_imageloss/glasses/t_edit=999/inv=100/no_fused/specific_encoder/layer_12/conv/lr0.00001/checkpoints \
--n_cs_layers=12 \
--t_end_edit 0 \
--dataset_type=infer_special_glasses \
--diffu_weights=pretrained_models/ffhq_p2.pt \
--diffu_config_path=configs/ffhq.yml \
--classifier_path /home/ids/ziliu-24/TIME_densenet/ffhq_glasses/best_densenet121.pth \
--n_inv_step=100 \
--batch_size=16 \
--dataset_root=/home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised

# results_infer/t_edit=400/no_fused/oneforC_oneforS/
                    # b # baseline # equalized_mlp # conv
# results_infer/t_edit=400/fused/oneforC_oneforS/
                    # bf # baseline_fused # equalized_mlp_fused # conv_fused

# results_infer/t_edit=400/no_fused/specific_encoder/
                    # s # specific_encoder # specific_mlp # specific_conv
# results_infer/t_edit=400/fused/specific_encoder/
                    # sf # specific_encoder_fused # specific_mlp_fused # specific_conv_fused

# age
# dataset_type=infer_special_age
# diffu_weights=pretrained_models/ffhq_p2.pt
# diffu_config_path=configs/ffhq.yml

# gender
# dataset_type=infer_special_gender
# diffu_weights=pretrained_models/celebahq_p2.pt
# diffu_config_path=configs/celeba_p2.yml