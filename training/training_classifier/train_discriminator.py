import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import shutil
import torch
import json
import os
import sys
sys.path.append(".")
sys.path.append("..")
from models.discriminator import CustomLatentClassifier
from base_functions.model_funcs import load_pSp_cmlp_models
from base_functions.data_funcs import reproduce_latent_hdf5, get_latent_dataloader
from base_functions.base_funcs import seed_experiments
import random

def train(model, train_dataloader, val_dataloader, criterion, optimizer, args, device):
    model.train()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"  # Convert `torch.device` to string
    
    os.makedirs(f"{args.results_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)

    scaler = torch.amp.GradScaler()  # Enable AMP
    gradient_accumulation_steps = 8  # Accumulate gradients over multiple steps

    for epoch in range(args.max_epochs):
        total_loss, correct, total = 0, 0, 0
        
        for step, (latent_c, labels) in enumerate(train_dataloader):
            latent_c = latent_c.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device_type):  # Enable Mixed Precision
                outputs = model(latent_c).squeeze()
                loss = criterion(outputs, labels)
            
            # Gradient accumulation for small batch sizes
            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Compute accuracy
            predicted = (outputs > 0).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

        train_loss = total_loss / total
        train_acc = correct / total

        # Log training loss
        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.max_epochs:
            val_loss, val_acc = validate(model, val_dataloader, criterion, device)
            log_msg = (f"Epoch [{epoch+1}/{args.max_epochs}] | "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
        else:
            log_msg = (f"Epoch [{epoch+1}/{args.max_epochs}] | "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n")

        write_log_to_txt(log_msg, args.results_dir, "train_val_loss.txt")

        # Save model checkpoint every `save_interval` epochs
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.max_epochs:
            checkpoint_path = f"{args.results_dir}/checkpoints/model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")


def validate(model, val_dataloader, criterion, device):
    """
    Evaluate the model on the validation dataset.
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for latent_c, labels in val_dataloader:
            latent_c = latent_c.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type):  # Enable Mixed Precision in validation
                outputs = model(latent_c).squeeze()
                loss = criterion(outputs, labels)
            
            # Compute accuracy
            predicted = (outputs > 0).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

    val_loss = total_loss / total
    val_acc = correct / total
    
    model.train()
    return val_loss, val_acc


def write_log_to_txt(log_msg, results_dir, filename):
    with open(f"{results_dir}/logs/{filename}", 'a') as f:
        f.write(log_msg)

def load_config(config_path, experiment_name):
    """Load JSON config file and return the selected experiment's config."""
    with open(config_path, "r") as file:
        full_config = json.load(file)

    if experiment_name not in full_config:
        raise ValueError(f"Experiment '{experiment_name}' not found in {config_path}")

    return full_config[experiment_name]

def save_hyparams(args):
    # Save hyperparameters to JSON
    args_json_path = os.path.join(args.results_dir, 'args.json')
    with open(args_json_path, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
    print(f"Arguments saved to {args_json_path}")

def main():
    # Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to JSON config file")
    parser.add_argument('--experiment', type=str, required=True, help="Name of the experiment to run")

    args, remaining_args = parser.parse_known_args()

    # Load selected experiment config
    config = load_config(args.config, args.experiment)

    # arguments with defaults from config
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=config.get('model_path', './results/cmlp_ffhq_glasses/iteration_130000.pt'))
    parser.add_argument('--latent_hdf5_path', type=str, default=config.get('latent_hdf5_path', './results/baseline/cmlp130k_latent'))
    parser.add_argument('--results_dir', type=str, default=config.get('results_dir', './results/discriminators'))
    parser.add_argument('--max_epochs', type=int, default=config.get('max_epochs', 200))
    parser.add_argument('--log_interval', type=int, default=config.get('log_interval', 1))
    parser.add_argument('--val_interval', type=int, default=config.get('val_interval', 2))
    parser.add_argument('--save_interval', type=int, default=config.get('save_interval', 20))
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 4))
    parser.add_argument('--num_layers', type=int, default=config.get('num_layers', 2))
    parser.add_argument('--lr', type=float, default=config.get('lr', 0.0001))
    parser.add_argument('--reproduce_latent', action='store_true', help="Enable latent reproduction")
    parser.add_argument('--no_reproduce_latent', action='store_false', dest="reproduce_latent", help="Disable latent reproduction")
    parser.set_defaults(reproduce_latent=not config.get('no_reproduce_latent', True))

    # Parse final arguments (command-line overrides config file)
    args = parser.parse_args(remaining_args)
    seed_value = random.randint(0, 2**10 - 1)  # Generate a random seed
    print("Random Seed:", seed_value)
    args.seed = seed_value
    # Print loaded settings for debugging
    print("training arguments:", vars(args))

    if os.path.exists(args.results_dir):
        shutil.rmtree(args.results_dir)  # Remove the existing directory and all its contents    
    os.makedirs(args.results_dir, exist_ok=True)
    save_hyparams(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_hdf5_path = f"{args.latent_hdf5_path}/train_latents.h5"
    val_hdf5_path = f"{args.latent_hdf5_path}/val_latents.h5"

    if args.reproduce_latent: 
        #seed_experiments(seed=args.seed) 
        pSp_net, cs_mlp_net, opts = load_pSp_cmlp_models(args.model_path, device=device)
        reproduce_latent_hdf5(cs_mlp_net, pSp_net, train_hdf5_path, val_hdf5_path, opts, seed=args.seed, device=device)

    train_latent_dataloader, val_latent_dataloader = get_latent_dataloader(train_hdf5_path, val_hdf5_path, args, label_0="latent_bg_c", label_1="latent_t_c")

    # Define model, loss, and optimizer
    model = CustomLatentClassifier(num_layers=args.num_layers).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model with periodic validation
    train(model, train_latent_dataloader, val_latent_dataloader, criterion, optimizer, args, device)


if __name__ == "__main__":
    main()
