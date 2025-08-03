#!/bin/bash
#SBATCH --job-name=image_metrics3
#SBATCH --nodes=1
#SBATCH --partition=L40S,A100,A40,H100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40GB
#SBATCH --time=24:00:00
#SBATCH --output=./reports/save_output.txt
#SBATCH --error=./reports/save_error.txt


### source environments ###
conda activate stylegan  # if using IDS cluster

# age
# python calc_image_metrics.py \
# --model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/ffhq_age/specific_encoder/fused/t_edit400/conv/layer_8/lr0.00001 \
# --save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/ffhq_age/specific_encoder/fused/t_edit400/conv/layer_8/lr0.00001 \
# --resize

# python calc_image_metrics.py \
# --model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/ffhq_age/specific_encoder/fused/t_edit400/conv/layer_12/lr0.00001 \
# --save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/ffhq_age/specific_encoder/fused/t_edit400/conv/layer_12/lr0.00001 \
# --resize

# python calc_image_metrics.py \
# --model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/ffhq_age/specific_encoder/fused/t_edit999/conv/layer_8/lr0.00001 \
# --save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/ffhq_age/specific_encoder/fused/t_edit999/conv/layer_8/lr0.00001 \
# --resize

# python calc_image_metrics.py \
# --model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/ffhq_age/specific_encoder/fused/t_edit999/conv/layer_12/lr0.00001 \
# --save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/ffhq_age/specific_encoder/fused/t_edit999/conv/layer_12/lr0.00001 \
# --resize

# # gender
# python calc_image_metrics.py \
# --model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/celeba-gender/specific_encoder/fused/t_edit400/conv/layer_8/lr0.00001 \
# --save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/celeba_gender/specific_encoder/fused/t_edit400/conv/layer_8/lr0.00001 \
# --resize

# python calc_image_metrics.py \
# --model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/celeba-gender/specific_encoder/fused/t_edit400/conv/layer_12/lr0.00001 \
# --save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/celeba_gender/specific_encoder/fused/t_edit400/conv/layer_12/lr0.00001 \
# --resize

# python calc_image_metrics.py \
# --model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/celeba-gender/specific_encoder/fused/t_edit999/conv/layer_8/lr0.00001 \
# --save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/celeba_gender/specific_encoder/fused/t_edit999/conv/layer_8/lr0.00001 \
# --resize

# python calc_image_metrics.py \
# --model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/celeba-gender/specific_encoder/fused/t_edit999/conv/layer_12/lr0.00001 \
# --save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/celeba_gender/specific_encoder/fused/t_edit999/conv/layer_12/lr0.00001 \
# --resize

# # smile
# python calc_image_metrics.py \
# --model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/celeba-smile/specific_encoder/fused/t_edit400/conv/layer_8/lr0.00001 \
# --save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/celeba_smile/specific_encoder/fused/t_edit400/conv/layer_8/lr0.00001 \
# --resize

# python calc_image_metrics.py \
# --model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/celeba-smile/specific_encoder/fused/t_edit400/conv/layer_12/lr0.00001 \
# --save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/celeba_smile/specific_encoder/fused/t_edit400/conv/layer_12/lr0.00001 \
# --resize

# python calc_image_metrics.py \
# --model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/celeba-smile/specific_encoder/fused/t_edit999/conv/layer_8/lr0.00001 \
# --save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/celeba_smile/specific_encoder/fused/t_edit999/conv/layer_8/lr0.00001 \
# --resize

# python calc_image_metrics.py \
# --model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/celeba-smile/specific_encoder/fused/t_edit999/conv/layer_12/lr0.00001 \
# --save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/celeba_smile/specific_encoder/fused/t_edit999/conv/layer_12/lr0.00001 \
# --resize

# glasses
# python calc_image_metrics.py \
# --model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/ffhq_glasses/specific_encoder/inv=100/fused/t_edit400/conv/layer_12/lr0.00001 \
# --save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/ffhq_glasses/specific_encoder/inv=100/fused/t_edit400/conv/layer_8/lr0.00001 \
# --resize

python calc_image_metrics.py \
--model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/ffhq_glasses/specific_encoder/inv=100/fused/t_edit999/conv/layer_12/lr0.00001/check_2600 \
--save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/ffhq_glasses/specific_encoder/inv=100/fused/t_edit999/conv/layer_12/lr0.00001/check_2600 \
--resize

python calc_image_metrics.py \
--model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/ffhq_glasses/specific_encoder/inv=100/fused/t_edit999/conv/layer_12/lr0.00001/check_2800 \
--save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/ffhq_glasses/specific_encoder/inv=100/fused/t_edit999/conv/layer_12/lr0.00001/check_2800 \
--resize

python calc_image_metrics.py \
--model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/ffhq_glasses/specific_encoder/inv=100/fused/t_edit999/conv/layer_12/lr0.00001/check_3500 \
--save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/ffhq_glasses/specific_encoder/inv=100/fused/t_edit999/conv/layer_12/lr0.00001/check_3500 \
--resize

python calc_image_metrics.py \
--model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/ffhq_glasses/specific_encoder/inv=100/fused/t_edit999/conv/layer_12/lr0.00001/check_3800 \
--save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/ffhq_glasses/specific_encoder/inv=100/fused/t_edit999/conv/layer_12/lr0.00001/check_3800 \
--resize

python calc_image_metrics.py \
--model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/ffhq_glasses/specific_encoder/inv=100/fused/t_edit999/conv/layer_12/lr0.00001/check_4400 \
--save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/ffhq_glasses/specific_encoder/inv=100/fused/t_edit999/conv/layer_12/lr0.00001/check_4400 \
--resize

# python calc_image_metrics.py \
# --model_path /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_results_infer/ffhq_glasses/specific_encoder/inv=100/no_fused/t_edit999/conv/layer_12/lr0.00001 \
# --save_dir /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/final_image_metrics/ffhq_glasses/specific_encoder/inv=100/no_fused/t_edit999/conv/layer_12/lr0.00001 \
# --resize