import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch
import os

def visualize_middle_h_edit_results(results, nrow=4):
    """
    Display a grid of images: 4 rows (batch samples), columns = different scales.
    Each cell = one generated image.
    """
    scales = sorted(results.keys())  # Ensure order
    batch_size = results[scales[0]].shape[0]
    assert batch_size >= nrow, f"Batch size {batch_size} < nrow {nrow}"

    # Normalize images from [-1, 1] to [0, 1] for display
    def norm_img(img):
        return (img.clamp(-1, 1) + 1) / 2

    # Build image grid row by row
    all_rows = []
    for i in range(nrow):  # for each sample in batch
        row_imgs = [norm_img(results[scale][i]) for scale in scales]
        row = torch.stack(row_imgs, dim=0)  # shape: [num_scales, 3, H, W]
        all_rows.append(row)

    # Stack all rows vertically
    grid = torch.cat(all_rows, dim=0)  # shape: [nrow * num_scales, 3, H, W]

    # Create grid image
    img_grid = vutils.make_grid(grid, nrow=len(scales), padding=2)

    # Plot
    plt.figure(figsize=(len(scales) * 3, nrow * 3))
    plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title("Rows = samples, Columns = scale values")
    plt.show()
    

def visualize_time_edit_results(all_results, edit_times, nrow=4):
    """
    Show a grid: rows = batch samples, columns = different t_edit points.
    all_results: list of result dicts (each with 1 scale)
    edit_times: list of t_edit values corresponding to each result
    """
    assert all(len(r) == 1 for r in all_results), "Each result dict should have only one scale"
    
    # Get the single scale value (e.g., 200)
    scale_key = list(all_results[0].keys())[0]
    
    batch_size = all_results[0][scale_key].shape[0]
    assert batch_size >= nrow, f"Batch size {batch_size} < nrow {nrow}"

    # Normalize for display
    def norm_img(img):
        return (img.clamp(-1, 1) + 1) / 2

    # Build rows of images
    all_rows = []
    for i in range(nrow):  # each row = one sample
        row_imgs = [norm_img(results[scale_key][i]) for results in all_results]
        row = torch.stack(row_imgs, dim=0)
        all_rows.append(row)

    # Stack all rows into one grid
    grid = torch.cat(all_rows, dim=0)
    img_grid = vutils.make_grid(grid, nrow=len(edit_times), padding=2)

    # Plot
    plt.figure(figsize=(len(edit_times) * 3, nrow * 3))
    plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title("Columns = different t_edit values")
    plt.show()




def visualize_batch_grid(image_batches, titles=None, row_indices=None, save_path=None):
    """
    Show or save a grid of images with aligned column titles.
    Each row = one sample across all versions (original, x_T, rec...).
    
    Args:
        image_batches (List[Tensor]): List of [B, C, H, W] tensors
        titles (List[str]): Column titles (same length as image_batches)
        row_indices (List[int]): Indices from batch to visualize as rows (e.g., [0, 2, 3])
        save_path (str): Optional path to save the image
    """
    def norm(img):
        return (img.clamp(-1, 1) + 1) / 2

    num_versions = len(image_batches)
    batch_size = image_batches[0].shape[0]

    # Default: show top-4 or less if batch is small
    if row_indices is None:
        max_show = min(4, batch_size)
        row_indices = list(range(max_show))

    n_rows = len(row_indices)

    fig, axes = plt.subplots(n_rows, num_versions, figsize=(3 * num_versions, 3 * n_rows))

    if n_rows == 1:
        axes = axes[None, :]  # ensure 2D shape

    for i, row_idx in enumerate(row_indices):
        for j in range(num_versions):
            # img = norm(image_batches[j][row_idx]).permute(1, 2, 0).cpu().numpy()
            img = norm(image_batches[j][row_idx]).detach().permute(1, 2, 0).cpu().numpy()
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis('off')
            if i == 0 and titles:
                ax.set_title(titles[j], fontsize=14)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


import matplotlib.pyplot as plt
import torchvision.utils as vutils

def visualize_denoising_timeline(xt_list, save_path=None, title="Denoising Timeline", step_interval=5):
    """
    Visualize the denoising process from x_T to x_0.
    
    Args:
        xt_list (List[Tensor]): List of [C, H, W] or [1, C, H, W] tensors
        save_path (str): Optional path to save image
        title (str): Title above the timeline
        step_interval (int): Plot every n-th timestep (e.g., every 5 steps)
    """
    # Take a single sample from each step
    xt_list = [x[0] if x.dim() == 4 else x for x in xt_list]

    # Select every n-th frame to reduce width
    xt_sampled = xt_list[::step_interval]
    steps = list(range(0, len(xt_list), step_interval))

    # Stack horizontally
    grid = vutils.make_grid(xt_sampled, nrow=len(xt_sampled), padding=2, normalize=True, range=(-1, 1))

    # Plot
    plt.figure(figsize=(len(xt_sampled) * 1.5, 3))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title(title, fontsize=16)
    
    # Optional save
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[✓] Saved timeline to: {save_path}")
    else:
        plt.show()

# def visualize_batch_grid(image_batches, titles=None, row_indices=None, save_path=None):
#     def norm(img):
#         return (img.clamp(-1, 1) + 1) / 2

#     num_versions = len(image_batches)
#     batch_size = image_batches[0].shape[0]

#     # Default: show top-4 or less if batch is small
#     if row_indices is None:
#         max_show = min(4, batch_size)
#         row_indices = list(range(max_show))

#     n_rows = len(row_indices)

#     fig, axes = plt.subplots(n_rows, num_versions, figsize=(3 * num_versions, 3 * n_rows))

#     # Normalize axes to always be 2D
#     if n_rows == 1 and num_versions == 1:
#         axes = np.array([[axes]])
#     elif n_rows == 1:
#         axes = axes[None, :]
#     elif num_versions == 1:
#         axes = axes[:, None]

#     for i, row_idx in enumerate(row_indices):
#         for j in range(num_versions):
#             img = norm(image_batches[j][row_idx]).permute(1, 2, 0).cpu().numpy()
#             ax = axes[i, j]
#             ax.imshow(img)
#             ax.axis('off')
#             if i == 0 and titles:
#                 ax.set_title(titles[j], fontsize=14)

#     plt.tight_layout()

#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, bbox_inches='tight')
#         print(f"Saved to {save_path}")
#         plt.close(fig)
#     else:
#         plt.show()


