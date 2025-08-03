import os
import torch
from IPython.display import display, HTML
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import random
def set_current_dir(CODE_DIR):
  os.chdir(f'{CODE_DIR}')
  
  notebook_path = os.getcwd()
  print('Current working directory is:', '\n', notebook_path, '\n') 

# Map GPU names to CUDA architectures
cuda_arch_map = {
    "A100": "8.0",
    "H100": "9.0",
    "V100": "7.0",
    "T4": "7.5",
    "RTX 3090": "8.6",
    "RTX 4090": "8.9"
}


def set_cuda_arch_list():
    if not torch.cuda.is_available():
        print("CUDA is not available. TORCH_CUDA_ARCH_LIST will not be set.")
        return

    # Get the GPU name
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {gpu_name}")

    # Match GPU name to CUDA architecture
    for key, arch in cuda_arch_map.items():
        if key in gpu_name:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            print(f"Setting TORCH_CUDA_ARCH_LIST to {arch} for {key}")
            return
    # Verify the environment variable
    print(f"TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST', 'Not set')}")


def show_images(*images, titles=None, img_size=(5, 5)):
    """
    Display multiple images in a single row.
    
    Args:
        images: Variable number of images (PIL.Image, NumPy arrays, or tensors).
        titles: Optional list of titles corresponding to each image.
        img_size: Tuple (width per image, height), default is (5,5).
    """
    num_images = len(images)
    
    # Create subplots dynamically based on number of images
    fig, axes = plt.subplots(1, num_images, figsize=(img_size[0] * num_images, img_size[1]))

    # Ensure axes is always iterable (for a single image case)
    if num_images == 1:
        axes = [axes]

    for i, (ax, img) in enumerate(zip(axes, images)):
        if not isinstance(img, np.ndarray):
            img = np.array(img)  # Convert PIL.Image to NumPy array if needed

        ax.imshow(img)
        ax.axis("off")  # Remove axes
        
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12)

    plt.show()



def display_images(*images, img_size=(256, 256)):
    """
    Displays multiple images in a single row using HTML and Base64 encoding.
    
    Args:
        images: Variable number of PIL.Image or tensors to be displayed.
        img_size: Tuple (width, height) for resizing images (default: 256x256).
    """
    img_tags = []

    for img in images:
        if not isinstance(img, Image.Image):
            img = tensor2im(img)  # Convert tensor to PIL.Image if needed

        img = img.resize(img_size)  # Resize image
        buffer = BytesIO()
        img.save(buffer, format="PNG")  # Convert to PNG
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")  # Encode to Base64
        img_tags.append(f"<img src='data:image/png;base64,{img_str}' style='margin:5px;'>")

    # Combine images into a single row
    display(HTML(f"<div style='display:flex; align-items:center;'>{''.join(img_tags)}</div>"))

def tensor2im(var):
    """
    Converts a PyTorch tensor to a PIL Image.
    
    Args:
        var: PyTorch tensor (CHW format, normalized between -1 and 1).
    
    Returns:
        PIL.Image object.
    """
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2) * 255  # Normalize to 0-255
    var = np.clip(var, 0, 255).astype(np.uint8)  # Clip values
    return Image.fromarray(var)

def load_folder_images(image_folder, num_images=4, seed=None):
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Get all image paths from the folder
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")]

    # Randomly select 'num_images' from the image_paths
    selected_paths = random.sample(image_paths, min(num_images, len(image_paths)))

    images = []

    for image_path in selected_paths:
        img = Image.open(image_path).convert("RGB")
        images.append(img)

    return images

def transform_images_to_batch(images_list, transforms):
    
    transformed_images = [transforms(image) for image in images_list] 
    # 'transformed_images' is a transformed version of images_list, each element is convert from Image  to 
    # a torch tensor using torchvision.transforms 

    batched_images = torch.stack(transformed_images, dim=0)
    # batched_images type of image tensors, e.g., 13x3x255x255

    return batched_images


def seed_experiments(seed):
    # Set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you use multi-GPU.

    # Ensures deterministic behavior for some PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def PCA_reconstruction(w_pSp, U_load_path, k):
    
    U = np.load(U_load_path)  # Load basis matrix
    
    # Flatten (20000, 18, 512) → (20000, 9216)
    w_pSp_flat = w_pSp.view(w_pSp.shape[0], -1).cpu().numpy()
      
    # Compute W_pca (Projection into lower-dimensional space)
    W_pca = np.dot(w_pSp_flat, U.T)  # Shape: (20000, k)
    W_recon_flat = np.dot(W_pca, U)  
    
    # Convert back to PyTorch tensor and reshape
    W_recon = torch.tensor(W_recon_flat).view(w_pSp.shape)

    # Compute reconstruction error
    error = np.linalg.norm(w_pSp_flat - W_recon_flat) / np.linalg.norm(w_pSp_flat)
    print(f"Reconstruction error for k={k}: {error:.6f}")

    return W_recon 

def PCA_projection(w_pSp, pSp_net, k, index = [0, 1, 2, 3]):
    
    W_latent = PCA_reconstruction(w_pSp, U_load_path = f"./PCA/U_pca_k{k}.npy", k=k)
    
    with torch.no_grad():
        recon_pSp_appr = pSp_net.forward(W_latent[index].to(device), input_code=True, randomize_noise=False, recon_modle=True)
    return recon_pSp_appr



def aggregate_loss_dict(agg_loss_dict):
	mean_vals = {}
	for output in agg_loss_dict:
		for key in output:
			mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
	for key in mean_vals:
		if len(mean_vals[key]) > 0:
			mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
		else:
			print('{} has no value'.format(key))
			mean_vals[key] = 0
	return mean_vals