import os
import torch
import lpips
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pytorch_msssim import ms_ssim
import torch.nn.functional as F
from pytorch_fid import fid_score
from criteria.id_loss_calc import IDLoss
import argparse
import json
import shutil

def calc_fid_from_two_paths(real_images_path, generated_images_path, device):
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_path, generated_images_path],
        batch_size=64,
        device=device,
        dims=2048,
    )
    return fid_value

def calc_recon_metrics(real_dir, recon_dir, device, max_images=None, resize=True):
    transform_list = []
    if resize:
        transform_list.append(transforms.Resize((256, 256)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)


    id_model = IDLoss().to(device).eval()
    lpips_model = lpips.LPIPS(net='alex').to(device)

    lpips_scores, mse_scores, ms_ssim_scores, id_sim_scores = [], [], [], []
    per_image_results = []

    real_files = set(os.listdir(real_dir))
    recon_files = set(os.listdir(recon_dir))
    common_filenames = sorted(real_files & recon_files)

    print(f"Found {len(common_filenames)} matching image pairs")

    with torch.no_grad():
        for idx, fname in enumerate(tqdm(common_filenames)):
            if max_images is not None and idx >= max_images:
                break

            real_path = os.path.join(real_dir, fname)
            recon_path = os.path.join(recon_dir, fname)

            try:
                real_img_raw = Image.open(real_path).convert('RGB')
                recon_img_raw = Image.open(recon_path).convert('RGB')
            except Exception as e:
                print(f"Skipping {fname} due to error: {e}")
                continue

            real_img = transform(real_img_raw).unsqueeze(0).to(device)
            recon_img = transform(recon_img_raw).unsqueeze(0).to(device)

            lp = lpips_model(real_img, recon_img).item()
            lpips_scores.append(lp)

            mse = F.mse_loss(real_img, recon_img).item()
            mse_scores.append(mse)

            ms = ms_ssim(real_img, recon_img, data_range=1.0).item()
            ms_ssim_scores.append(ms)

            real_feat = id_model.extract_feats(real_img)
            recon_feat = id_model.extract_feats(recon_img)
            sim = F.cosine_similarity(real_feat, recon_feat).item()
            id_sim_scores.append(sim)

            per_image_results.append({
                "image_name": fname,
                "lpips": lp,
                "mse": mse,
                "ms_ssim": ms,
                "id_sim": sim
            })

    fid_value = calc_fid_from_two_paths(real_dir, recon_dir, device)

    results = {
        "mean": {
            "lpips": np.mean(lpips_scores),
            "mse": np.mean(mse_scores),
            "ms_ssim": np.mean(ms_ssim_scores),
            "id_sim": np.mean(id_sim_scores),
            "fid": fid_value,
        },
        "per_image": per_image_results
    }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch metric calculation for multiple image folder pairs.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the base directory containing image subfolders')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save each pair\'s JSON results')
    parser.add_argument('--max_images', type=int, default=None, help='Optional cap on number of image pairs to evaluate')
    parser.add_argument('--resize', action='store_true', help='Resize images to 256x256 before evaluation')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # folder_pairs = [
    #     ("real_x", "edited_x"),
    #     ("real_x", "edited_y"),
    #     ("real_y", "edited_x"),
    #     ("real_y", "edited_y"),

    #     ("real_x", "enc_edited_x"),
    #     ("real_x", "enc_edited_y"),
    #     ("real_y", "enc_edited_x"),
    #     ("real_y", "enc_edited_y"),
    # ]

    folder_pairs = [
    #     ("real_x", "recon_w_x"),
    #     ("real_y", "recon_w_y"),
    #     ("real_x", "recon_f_x"),
    #     ("real_y", "recon_f_y"),

    #     # ("real_x", "swap_w_y2x"),
    #     # ("real_x", "swap_f_y2x"),
    #     # ("real_x", "swap_w_x2y"),
    #     # ("real_x", "swap_f_x2y"),

    #     # ("real_y", "swap_w_y2x"),
    #     # ("real_y", "swap_f_y2x"),
    #     # ("real_y", "swap_w_x2y"),
    #     # ("real_y", "swap_f_x2y"),
    ("real_x", "recon_x"),
    ("real_x", "swap_y2x"),
    ("real_x", "swap_x2y"),
    
    ("real_y", "recon_y"),
    ("real_y", "swap_x2y"),
    ("real_y", "swap_y2x"),
    
    ]


    # folder_pairs = [
    #     ("real_x", "inv_x"),
    #     ("real_y", "inv_y"),
    #     # ("real_x", "recon_f_x"),
    #     # ("real_y", "recon_f_y"),

    #     # ("real_x", "swap_w_y2x"),
    #     # ("real_x", "swap_f_y2x"),
    #     # ("real_x", "swap_w_x2y"),
    #     # ("real_x", "swap_f_x2y"),

    #     # ("real_y", "swap_w_y2x"),
    #     # ("real_y", "swap_f_y2x"),
    #     # ("real_y", "swap_w_x2y"),
    #     # ("real_y", "swap_f_x2y"),
    # ]
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    for real_name, recon_name in folder_pairs:
        real_dir = os.path.join(args.model_path, real_name)
        recon_dir = os.path.join(args.model_path, recon_name)
        print(f"\nEvaluating: {real_name} vs {recon_name}")
        metrics = calc_recon_metrics(real_dir, recon_dir, device, max_images=args.max_images, resize = args.resize)

        save_filename = f"{real_name}_vs_{recon_name}.json"
        save_path = os.path.join(args.save_dir, save_filename)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved results to {save_path}")
        

