import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import shutil
import torch
import json
import os
import sys
import torch.nn.functional as F
sys.path.append(".")
sys.path.append("..")
from base_functions.model_funcs import load_pSp_cmlp_models
from base_functions.data_funcs import reproduce_latent_hdf5, HDF5LatentDataset
from base_functions.base_funcs import seed_experiments
import random
from torch.utils.data import Dataset, DataLoader
from models.c2s_mlp import SimpleLinearModel, DeepC2SModel, StrongerC2SModel, MappingNetwork_c2s

# from models.mlp2D import MappingNetwork_c2s

def train(model, train_dataloader, val_dataloader, optimizer, args, device):
    """
    Train the classifier and evaluate on the validation dataset at specified intervals.
    Saves model checkpoints and logs losses to files.

    Args:
        model: PyTorch model.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        optimizer: Optimizer.
        args: Argument namespace with batch_size, max_epochs, val_interval, save_interval, results_dir, and device.
        device: Device to use (CPU or GPU).
    """

    model.train()

    # Remove and recreate directories for logs and checkpoints

    os.makedirs(f"{args.results_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)

    for epoch in range(args.max_epochs):
        running_loss = 0.0  # Track real loss

        for latent_t_c, latent_t_s in train_dataloader:
            latent_t_c, latent_t_s = latent_t_c.to(device), latent_t_s.to(device)

            # Compute MI loss and separate real & fake losses
            output_t_s = model(latent_t_c)

            mi_loss = F.mse_loss(output_t_s, latent_t_s)  # Fix: Compare with latent_t_s

            # **Backpropagation**
            optimizer.zero_grad()
            mi_loss.backward()
            optimizer.step()

            # **Track losses for epoch summary**
            running_loss += mi_loss.item()


        # **Compute the average loss for the epoch**
        avg_loss = running_loss / len(train_dataloader)

        # Validation step
        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.max_epochs:
            model.eval()  # Ensure evaluation mode
            val_loss = validate(model, val_dataloader, device)
            model.train()  # Restore training mode      
            log_msg = f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}\n"
        else:
            log_msg = f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}\n"
        
        write_log_to_txt(log_msg, args.results_dir, "train_val_loss.txt")


        # Save model checkpoint every `save_interval` epochs
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.max_epochs:
            checkpoint_path = f"{args.results_dir}/checkpoints/model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

# def perform_pca(latent, U_pca):

#         # Flatten (20000, 18, 512) → (20000, 9216)
#     latent_flat = latent.view(latent.shape[0], -1)

#     latent_proj = latent_flat @ U_pca.T  # Shape: (batch_size, k)      
#     # W_recon_flat = np.dot(W_pca, U)  

#     # # Convert back to PyTorch tensor and reshape
#     # W_recon = torch.tensor(W_recon_flat).view(latent.shape)
#     return latent_proj

def validate(model, val_dataloader, device):
    """
    Evaluate the model on the validation dataset.
    """
    model.eval()
    running_loss = 0.0

    num_batches = len(val_dataloader)  # Store once for efficiency

    with torch.no_grad():
        for latent_t_c, latent_t_s in val_dataloader:
            latent_t_c, latent_t_s = latent_t_c.to(device), latent_t_s.to(device)
            # Compute MI loss and separate real & fake losses
            output_t_s = model(latent_t_c)

            mi_loss = F.mse_loss(output_t_s, latent_t_s)  # Fix: Compare with latent_t_s

            # **Track losses for epoch summary**
            running_loss += mi_loss.item()

    # **Safeguard Against Division by Zero**
    if num_batches > 0:
        avg_loss = running_loss / num_batches
    else:
        avg_loss = 0.0  # Edge case handling

    model.train()
    return avg_loss


def write_log_to_txt(log_msg, results_dir, filename):
    with open(f"{results_dir}/logs/{filename}", 'a') as f:
        f.write(log_msg)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./results/cmlp_ffhq_glasses/iteration_130000.pt')
    parser.add_argument('--results_dir', type=str, default='./results/adverserial_mutual')  # Path to save logs & checkpoints
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=1)  # Validate every 10 epochs
    parser.add_argument('--val_interval', type=int, default=2)  # Validate every 10 epochs
    parser.add_argument('--save_interval', type=int, default=10)  # Save checkpoint every 10 epochs
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_c2s_layers', type=int, default=12)
    parser.add_argument('--network_type', type=str, default='simple')
    parser.add_argument('--style_dim', type=int, default=18)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--latent_hdf5_path', type=str, default='./results/alternative_training/classifiers')  # Path to save logs & checkpoints
    parser.add_argument('--reproduce_latent', action='store_true', help="Enable latent reproduction")
    parser.add_argument('--no_reproduce_latent', action='store_false', dest="reproduce_latent", help="Disable latent reproduction")
    parser.set_defaults(reproduce_latent=True)  # Set default to True

    args = parser.parse_args()

    if os.path.exists(args.results_dir):
        shutil.rmtree(args.results_dir)  # Remove the existing directory and all its contents    
    os.makedirs(args.results_dir, exist_ok=True)

    seed_value = random.randint(0, 2**32 - 1)  # Generate a random seed
    print("Random Seed:", seed_value)
    args.seed = seed_value
    
    # Save args to JSON
    args_json_path = os.path.join(args.results_dir, 'args.json')
    with open(args_json_path, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
    print(f"Arguments saved to {args_json_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #seed_experiments(seed=args.seed) 
    pSp_net, cs_mlp_net, opts = load_pSp_cmlp_models(args.model_path, device=device)

    train_hdf5_path = f"{args.latent_hdf5_path}/train_latents.h5"
    val_hdf5_path = f"{args.latent_hdf5_path}/val_latents.h5"

    if args.reproduce_latent: 
        reproduce_latent_hdf5(cs_mlp_net, pSp_net, train_hdf5_path, val_hdf5_path, opts, seed=args.seed, device=device)

    # latent_keys = ["w_pSp_bg", "w_pSp_t", "latent_bg_c", "latent_bg_s", "latent_t_c", "latent_t_s"]
    latent_keys = ["latent_t_c", "latent_t_s"]
    
    # Initialize dataset
    train_dataset = HDF5LatentDataset(train_hdf5_path, latent_keys)
    train_latent_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataset = HDF5LatentDataset(val_hdf5_path, latent_keys)
    val_latent_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    #model = MappingNetwork_c2s(features=args.pca_latent_dim, n_layers=args.n_c2s_layers).to(device)
    if args.network_type == 'simple':
        model = SimpleLinearModel().to(device)
    elif args.network_type == 'deep':
        model = DeepC2SModel().to(device)
    elif args.network_type == 'strong': 
        model = StrongerC2SModel().to(device)
    elif args.network_type == 'c2smlp': 
        model = MappingNetwork_c2s(n_layers=args.n_c2s_layers).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model with periodic validation
    train(model, train_latent_dataloader, val_latent_dataloader, optimizer, args, device)


if __name__ == "__main__":
    main()
