from models.guided_diffusion.script_util import guided_Diffusion
from models.improved_ddpm.script_util import i_DDPM
from models.guided_diffusion.unet import UNet_h_encoder, UNet_h_decoder  # adjust import to your file structure
from configs.paths_config import MODEL_PATHS
import argparse
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as tvu
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import time
import yaml


# Helper function
class dotdict(dict):
    """Helper class to access dictionary attributes with dot notation."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Your original function
def diffusion_args_and_config(device=None):
    # parser = argparse.ArgumentParser(description=globals().get('__doc__', ''))
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_inv_step', type=int, default=40, help='# of steps during generative pross for inversion')
    parser.add_argument('--t_0', type=int, default=999, help='Return step in [0, 1000)')
    parser.add_argument('--sample_type', type=str, default='ddim', help='ddpm for Markovian sampling, ddim for non-Markovian sampling')
    parser.add_argument('--config', type=str, default='custom.yml', help='Path to the config file')

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    new_config.device = device

    return args, new_config

class utils:
    def extract(a, t, x_shape):
        """Extract coefficients from a based on t and reshape to make it
        broadcastable with x_shape."""
        bs, = t.shape
        assert x_shape[0] == bs, f"{x_shape[0]}, {t.shape}"
        # out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())

        # If a might be a list or numpy array, do safe conversion
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float, device=t.device)
        else:
            a = a.to(dtype=torch.float, device=t.device)

        out = torch.gather(a, 0, t.long())

        assert out.shape == (bs,)
        out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
        return out

    def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
        betas = np.linspace(beta_start, beta_end,
                            num_diffusion_timesteps, dtype=np.float64)
        assert betas.shape == (num_diffusion_timesteps,)
        return betas
    
class reconstruction:
    def __init__(self, args, config, device):
        self.device = device
        self.args = args
        self.config = config

        model_var_type = config.model.var_type
        betas = utils.get_beta_schedule(
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        
        if model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        self.learn_sigma = True

    def generate_time_steps(self, n_inv_step, t_0):
        seq_inv = np.linspace(0, 1, n_inv_step) * t_0
        seq_inv = [int(s+1e-6) for s in list(seq_inv)] # [s₀, s₁, s₂, ..., sₙ₋₁]
        seq_inv_next = [-1] + list(seq_inv[:-1]) # [ -1, s₀, s₁, ..., sₙ₋₂ ]

        return seq_inv, seq_inv_next   

    def denoising_step_enc_dec(self, xt, t, t_next, *,
                    models,
                    et,
                    et_modified,
                    logvars,
                    b,
                    sampling_type='ddim',
                    eta=0.0,
                    learn_sigma=False,
                    index=None,
                    t_edit=0,
                    hs_coeff=(1.0),
                    dt_lambda=1,
                    image_space_noise=0,
                    dt_end = 999,
                    warigari=False,
                    ):

        # Compute noise and variance

 
        # print(middle_h.shape)
        if learn_sigma:
            et, logvar_learned = torch.split(et, et.shape[1] // 2, dim=1)
            if index is not None:
                et_modified, _ = torch.split(et_modified, et_modified.shape[1] // 2, dim=1)
            logvar = logvar_learned
        else:
            logvar = utils.extract(logvars, t, xt.shape)

        if type(image_space_noise) != int:
            if t[0] >= t_edit:
                index = 0
                if type(image_space_noise) == torch.nn.parameter.Parameter:
                    et_modified = et + image_space_noise * hs_coeff[1]
                else:
                    # print(type(image_space_noise))
                    temb = models.module.get_temb(t)
                    et_modified = et + image_space_noise(et, temb) * 0.01

        # Compute the next x
        bt = utils.extract(b, t, xt.shape)
        at = utils.extract((1.0 - b).cumprod(dim=0), t, xt.shape)
        if t_next.sum() == -t_next.shape[0]:
            at_next = torch.ones_like(at)
        else:
            at_next = utils.extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)

        xt_next = torch.zeros_like(xt)
        # Different sampling strategies to update the latent code. the inverse process of mapping an image from the data space to the latent space.
        if sampling_type == 'ddpm':
            weight = bt / torch.sqrt(1 - at)

            mean = 1 / torch.sqrt(1.0 - bt) * (xt - weight * et)
            noise = torch.randn_like(xt)
            mask = 1 - (t == 0).float()
            mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
            xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
            xt_next = xt_next.float()

        elif sampling_type == 'ddim':
            if index is not None:
                x0_t = (xt - et_modified * (1 - at).sqrt()) / at.sqrt()
            else:
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            # Deterministic.
            if eta == 0:
                xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
            # Add noise. When eta is 1 and time step is 1000, it is equal to ddpm.
            else:
                c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)

        if dt_lambda != 1 and t[0] >= dt_end:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et * dt_lambda

        # Asyrp & DiffStyle
        if not warigari or index is None:
            return xt_next, x0_t

        # Warigari by young-hyun, Not in the paper
        else:
            # will be updated
            # Returns the updated latent encoding xt_next, the estimated denoised image x0_t, the editing parameter delta_h, and the middle hidden feature middle_h.
            return xt_next, x0_t
        

    def denoising_step(self, xt, t, t_next, *,
                    models,
                    logvars,
                    b,
                    sampling_type='ddim',
                    eta=0.0,
                    learn_sigma=False,
                    index=None,
                    t_edit=0,
                    hs_coeff=(1.0),
                    delta_h=None,
                    use_mask=False,
                    dt_lambda=1,
                    ignore_timestep=False,
                    image_space_noise=0,
                    dt_end = 999,
                    warigari=False,
                    ):

        # Compute noise and variance
        model = models
        # The model is called with the current latent encoding xt and the current time step t .......... Here middle_h is the output of layer 8 (middle hidden feature)
        et, et_modified, delta_h, middle_h = model(xt, t, index=index, t_edit=t_edit, hs_coeff=hs_coeff, delta_h=delta_h, ignore_timestep=ignore_timestep, use_mask=use_mask)
        # print(middle_h.shape)
        if learn_sigma:
            et, logvar_learned = torch.split(et, et.shape[1] // 2, dim=1)
            if index is not None:
                et_modified, _ = torch.split(et_modified, et_modified.shape[1] // 2, dim=1)
            logvar = logvar_learned
        else:
            logvar = utils.extract(logvars, t, xt.shape)

        if type(image_space_noise) != int:
            if t[0] >= t_edit:
                index = 0
                if type(image_space_noise) == torch.nn.parameter.Parameter:
                    et_modified = et + image_space_noise * hs_coeff[1]
                else:
                    # print(type(image_space_noise))
                    temb = models.module.get_temb(t)
                    et_modified = et + image_space_noise(et, temb) * 0.01

        # Compute the next x
        bt = utils.extract(b, t, xt.shape)
        at = utils.extract((1.0 - b).cumprod(dim=0), t, xt.shape)
        if t_next.sum() == -t_next.shape[0]:
            at_next = torch.ones_like(at)
        else:
            at_next = utils.extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)

        xt_next = torch.zeros_like(xt)
        # Different sampling strategies to update the latent code. the inverse process of mapping an image from the data space to the latent space.
        if sampling_type == 'ddpm':
            weight = bt / torch.sqrt(1 - at)

            mean = 1 / torch.sqrt(1.0 - bt) * (xt - weight * et)
            noise = torch.randn_like(xt)
            mask = 1 - (t == 0).float()
            mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
            xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
            xt_next = xt_next.float()

        elif sampling_type == 'ddim':
            if index is not None:
                x0_t = (xt - et_modified * (1 - at).sqrt()) / at.sqrt()
            else:
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            # Deterministic.
            if eta == 0:
                xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
            # Add noise. When eta is 1 and time step is 1000, it is equal to ddpm.
            else:
                c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)

        if dt_lambda != 1 and t[0] >= dt_end:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et * dt_lambda

        # Asyrp & DiffStyle
        if not warigari or index is None:
            return xt_next, x0_t, delta_h, middle_h

        # Warigari by young-hyun, Not in the paper
        else:
            # will be updated
            # Returns the updated latent encoding xt_next, the estimated denoised image x0_t, the editing parameter delta_h, and the middle hidden feature middle_h.
            return xt_next, x0_t, delta_h, middle_h


    def inject_noise_batch(self, model, batch_image, seq_inv, seq_inv_next):
        x = batch_image  # Ensure tensor is on correct device

        # if save_images:
        #     for step in range(x.shape[0]):
        #         tvu.save_image((x[step] + 1) * 0.5, os.path.join(
        #             self.args.destination_folder, f'{step}_orig.png'))

        with torch.no_grad():
            for i, j in zip(seq_inv_next[1:], seq_inv[1:]):
                t = torch.full((x.size(0),), i, device=x.device)
                t_prev = torch.full((x.size(0),), j, device=x.device)

                x, _, _, _ = self.denoising_step(
                    x, t=t, t_next=t_prev, models=model,
                    logvars=self.logvar, sampling_type='ddim',
                    b=self.betas, eta=0, learn_sigma=self.learn_sigma
                )

        # if save_images:
        #     for step in range(x.shape[0]):
        #         tvu.save_image((x[step] + 1) * 0.5, os.path.join(
        #             self.args.destination_folder, f'{step}_lat_t{self.args.n_inv_step}.png'))

        return x  # shape: [B, C, H, W]


    def generate_from_xT_batch(self, model, encoder, decoder, batch_xT, seq_inv, seq_inv_next, save_images=False):
        x = batch_xT  # Ensure tensor is on correct device

        with torch.no_grad():
            for i, j in zip(reversed(seq_inv), reversed(seq_inv_next)):
                t = torch.full((x.size(0),), i, device=x.device)
                t_next = torch.full((x.size(0),), j, device=x.device)
                
                middle_h, hs, emb = encoder(x, t)
                et, et_modified = decoder(middle_h, hs, emb, x)

                x, _, = self.denoising_step_enc_dec(
                    xt=x, t=t, t_next=t_next, models=model,
                    et=et, et_modified=et_modified,
                    logvars=self.logvar, sampling_type=self.args.sample_type,
                    b=self.betas, learn_sigma=self.learn_sigma
                )
        # if save_images:
        #     for step in range(x.shape[0]):
        #         tvu.save_image((x[step] + 1) * 0.5, os.path.join(
        #             self.args.destination_folder, f'{step}_rec_t{self.args.n_inv_step}.png'))

        return x  # shape: [B, C, H, W]


    def experiment_middle_h_edit(self, model, encoder, decoder, batch_xT, 
                                 seq_inv, seq_inv_next, t_edit=400, scales=[0.0, 0.2, 0.5], set_random_seed=None):
        results = {}

        for scale in scales:
            x = batch_xT.clone()  # Reset for each scale

            with torch.no_grad():
                for i, j in zip(reversed(seq_inv), reversed(seq_inv_next)):
                    t = torch.full((x.size(0),), i, device=x.device)
                    t_next = torch.full((x.size(0),), j, device=x.device)

                    # Standard encoder pass
                    middle_h, hs, emb = encoder(x, t)

                    # Inject edit at the exact step
                    if i == t_edit:
                        if set_random_seed is not None:
                            torch.manual_seed(set_random_seed)
                            torch.cuda.manual_seed_all(set_random_seed)
                        middle_h = middle_h + scale * torch.randn_like(middle_h)
                        print(f"[Edit] Applied scale={scale:.2f} noise at t={i}")

                    # Decode and denoise
                    et, et_modified = decoder(middle_h, hs, emb, x)
                    x, _ = self.denoising_step_enc_dec(
                        xt=x, t=t, t_next=t_next, models=model,
                        et=et, et_modified=et_modified,
                        logvars=self.logvar, sampling_type=self.args.sample_type,
                        b=self.betas, learn_sigma=self.learn_sigma
                    )

            results[scale] = x

        return results, t_edit

    def experiment_middle_h_edit_all_layers(self, model, encoder, decoder, batch_xT, 
                                 seq_inv, seq_inv_next, scales=[0.0, 0.2, 0.5], set_random_seed=None):
        results = {}

        for scale in scales:
            x = batch_xT.clone()  # Reset for each scale

            with torch.no_grad():
                for i, j in zip(reversed(seq_inv), reversed(seq_inv_next)):
                    t = torch.full((x.size(0),), i, device=x.device)
                    t_next = torch.full((x.size(0),), j, device=x.device)

                    # Standard encoder pass
                    middle_h, hs, emb = encoder(x, t)

                    # Inject edit at the exact step
                    t_edit=i
                    if set_random_seed is not None:
                        torch.manual_seed(set_random_seed)
                        torch.cuda.manual_seed_all(set_random_seed)
                    middle_h = middle_h + scale * torch.randn_like(middle_h)
                    print(f"[Edit] Applied scale={scale:.2f} noise at t={i}")

                    # Decode and denoise
                    et, et_modified = decoder(middle_h, hs, emb, x)
                    x, _ = self.denoising_step_enc_dec(
                        xt=x, t=t, t_next=t_next, models=model,
                        et=et, et_modified=et_modified,
                        logvars=self.logvar, sampling_type=self.args.sample_type,
                        b=self.betas, learn_sigma=self.learn_sigma
                    )

            results[scale] = x

        return results, t_edit
