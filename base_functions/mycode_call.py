from models.guided_diffusion.script_util import guided_Diffusion
from models.improved_ddpm.script_util import i_DDPM
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
from base_functions.custom_utils import dict2namespace



def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--n_inv_step', type=int, default=40, help='# of steps during generative pross for inversion')
    parser.add_argument('--t_0', type=int, default=999, help='Return step in [0, 1000)')
    parser.add_argument('--sample_type', type=str, default='ddim', help='ddpm for Markovian sampling, ddim for non-Markovian sampling')
    parser.add_argument('--config', type=str, default='custom.yml', help='Path to the config file')

    parser.add_argument('--destination_folder', type=str, default='/home/ids/ziliu-24/Asyrp_official_original/my_sample_img/result')
    parser.add_argument('--original_image_folder', type=str, default='/home/ids/ziliu-24/Asyrp_official_original/my_sample_img')
    # Reminder in the source code that it can only be set to 1 ..... 
    # Unfortunately, ima_lat_pairs_dic does not match with batch_size
    # I'm sorry but you have to get ima_lat_pairs_dic with batch_size == 1
    parser.add_argument('--batch_size', type=int, default='1')

    # Because the article is using different model loading and processing methods based on different datasets
    # So for model_path, you need to revise the data/dataset in "your_path/configs/custom.yml" and "your_path/configs/paths_config.py"

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    new_config.device = device

    return args, new_config

args, config = parse_args_and_config()
os.makedirs(args.destination_folder, exist_ok=True)

# Custom Dataset Classes
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Get all image paths
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Image preprocessing 
transform = transforms.Compose([
    transforms.Resize((256, 256)),                      
    transforms.ToTensor(),                             
    transforms.Normalize((0.5, 0.5, 0.5),              
                         (0.5, 0.5, 0.5))
])

dataset = CustomImageDataset(args.original_image_folder, transform)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Load a batch and print its dimensions.
# for batch in loader:
#     print("Loaded batch shape:", batch.shape)  # 形状应为 [batch_size, 3, 256, 256]
#     break

class utils:
    def extract(a, t, x_shape):
        """Extract coefficients from a based on t and reshape to make it
        broadcastable with x_shape."""
        bs, = t.shape
        assert x_shape[0] == bs, f"{x_shape[0]}, {t.shape}"
        out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
        assert out.shape == (bs,)
        out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
        return out

    def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
        betas = np.linspace(beta_start, beta_end,
                            num_diffusion_timesteps, dtype=np.float64)
        assert betas.shape == (num_diffusion_timesteps,)
        return betas
    
class reconstruction:
    def __init__(self, args, config):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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

    def load_pretrained_model(self):
        if self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            model = i_DDPM(self.config.data.dataset) #Get_h(self.config, model="i_DDPM", layer_num=self.args.get_h_num) #
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            self.learn_sigma = True
            print("Improved diffusion Model loaded.")
        elif self.config.data.dataset in ["MetFACE", "CelebA_HQ_P2"]:
            model = guided_Diffusion(self.config.data.dataset)
            init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            self.learn_sigma = True
        else:
            print('Not implemented dataset')
            raise ValueError
        
        model.load_state_dict(init_ckpt, strict=False)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        return model
    


    def predict_noise_and_x0(self, xt, t, encoder, decoder, model=None, index=None, t_edit=0, hs_coeff=(1.0,), delta_h=None, ignore_timestep=False, use_mask=False, image_space_noise=0):
        middle_h, hs, emb = encoder(xt, t)
        edited_h = middle_h if middle_h is None else middle_h
        et, et_modified = decoder(edited_h, hs.copy(), emb, xt)

        if type(image_space_noise) != int:
            if t[0] >= t_edit:
                index = 0
                if isinstance(image_space_noise, torch.nn.parameter.Parameter):
                    et_modified = et + image_space_noise * hs_coeff[1]
                else:
                    temb = model.module.get_temb(t)
                    et_modified = et + image_space_noise(et, temb) * 0.01

        return et, et_modified, middle_h

    def compute_xt_next(self, xt, t, t_next, et, et_modified, logvars, b, sampling_type='ddim', eta=0.0, learn_sigma=False, dt_lambda=1, dt_end=999, index=None):
        bt = utils.extract(b, t, xt.shape)
        at = utils.extract((1.0 - b).cumprod(dim=0), t, xt.shape)
        if t_next.sum() == -t_next.shape[0]:
            at_next = torch.ones_like(at)
        else:
            at_next = utils.extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)

        if learn_sigma:
            et, logvar_learned = torch.split(et, et.shape[1] // 2, dim=1)
            if index is not None:
                et_modified, _ = torch.split(et_modified, et_modified.shape[1] // 2, dim=1)
            logvar = logvar_learned
        else:
            logvar = utils.extract(logvars, t, xt.shape)

        if sampling_type == 'ddpm':
            weight = bt / torch.sqrt(1 - at)
            mean = 1 / torch.sqrt(1.0 - bt) * (xt - weight * et)
            noise = torch.randn_like(xt)
            mask = 1 - (t == 0).float()
            mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
            xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
            xt_next = xt_next.float()
        else:
            if index is not None:
                x0_t = (xt - et_modified * (1 - at).sqrt()) / at.sqrt()
            else:
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            if eta == 0:
                xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
            else:
                c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)

        if dt_lambda != 1 and t[0] >= dt_end:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et * dt_lambda

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

    def predict_noise_and_x0(self, xt, t, encoder, decoder, model=None, index=None, t_edit=0, hs_coeff=(1.0,), delta_h=None, ignore_timestep=False, use_mask=False, image_space_noise=0):
        middle_h, hs, emb = encoder(xt, t)
        edited_h = middle_h if middle_h is None else middle_h
        et, et_modified = decoder(edited_h, hs.copy(), emb, xt)

        if type(image_space_noise) != int:
            if t[0] >= t_edit:
                index = 0
                if isinstance(image_space_noise, torch.nn.parameter.Parameter):
                    et_modified = et + image_space_noise * hs_coeff[1]
                else:
                    temb = model.module.get_temb(t)
                    et_modified = et + image_space_noise(et, temb) * 0.01

        return et, et_modified, middle_h
    
    def noise_injection(self, model):
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s + 1e-6) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])
        n = 1

        x_lat_all_steps = []

        for step, img in enumerate(self.loader):
            x0 = img.to(self.device)
            x = x0.clone()
            with torch.no_grad():
                for i, j in zip(seq_inv_next[1:], seq_inv[1:]):
                    t = (torch.ones(n) * i).to(self.device)
                    t_prev = (torch.ones(n) * j).to(self.device)
                    x, _, _, _ = self.denoising_step(
                        x, t=t, t_next=t_prev, models=model,
                        logvars=self.logvar,
                        sampling_type='ddim',
                        b=self.betas,
                        eta=0,
                        learn_sigma=self.learn_sigma
                    )
                    x_lat_all_steps.append(x.detach().cpu())
            break  # only sample one image for now
        return x_lat_all_steps[-1]  # return x_T (final latent after all inversion steps)

    def generate_time_steps(args):
        seq_inv = np.linspace(0, 1, args.n_inv_step) * args.t_0
        seq_inv = [int(s+1e-6) for s in list(seq_inv)] # [s₀, s₁, s₂, ..., sₙ₋₁]
        seq_inv_next = [-1] + list(seq_inv[:-1]) # [ -1, s₀, s₁, ..., sₙ₋₂ ]

        return seq_inv, seq_inv_next        
    


    def precompute_pairs(self, model):
        # Generate a sequence of time steps in the inversion process
        seq_inv = np.linspace(0, 1, args.n_inv_step) * args.t_0
        seq_inv = [int(s+1e-6) for s in list(seq_inv)] # [s₀, s₁, s₂, ..., sₙ₋₁]
        seq_inv_next = [-1] + list(seq_inv[:-1]) # [ -1, s₀, s₁, ..., sₙ₋₂ ]
        n = 1

        middle_h_dict = {}
        middle_h_all_steps = []  # Used to save the middle_h for each step
        x_lat_dict = {}
        x_lat_all_steps = []  # Used to save the x_lat for each step

        # reverse diffusion process
        for step, img in enumerate(loader):
            img_lat_pairs = []
            x0 = img.to(self.device)
            tvu.save_image((x0 + 1) * 0.5, os.path.join(args.destination_folder, f'{step}_0_orig.png'))

            x = x0.clone()
            model.eval()
            time_s = time.time()
            with torch.no_grad():
                # The original image x0 is mapped to the latent space through an inverse diffusion process to obtain the corresponding latent representation (x is finally added to pure noise).
                with tqdm(total=len(seq_inv), desc=f"Inversion process {step}") as progress_bar:
                    for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                        t = (torch.ones(n) * i).to(self.device)
                        t_prev = (torch.ones(n) * j).to(self.device)
                        # Each call to denoising_step updates x with the current timestep t and the previous timestep t_prev, so that x is progressively “denoised” and approaches the potential representation.
                        # x is latent code
                        x, _, _, middle_h = self.denoising_step(x, t=t, t_next=t_prev, models=model,
                                            logvars=self.logvar,
                                            sampling_type='ddim',
                                            b=self.betas,
                                            eta=0,
                                            learn_sigma=self.learn_sigma,
                                            )
                        progress_bar.update(1)
                        # Here I preserve middle_h of each image in array
                        middle_h_all_steps.append(middle_h.detach().cpu())
                        x_lat_all_steps.append(x.detach().cpu())
                    middle_h_dict[step] = middle_h_all_steps
                    x_lat_dict[step] = x_lat_all_steps
                
                time_e = time.time()
                print(f'{time_e - time_s} seconds')
                x_lat = x.clone() 
                tvu.save_image((x_lat + 1) * 0.5, os.path.join(args.destination_folder, f'{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                # Forward generation of the reconstructed image based on x_lat is used to check whether the precomputed latent coding is able to reconstruct a result that is consistent with the original image.
                with tqdm(total=len(seq_inv), desc=f"Generative process {step}") as progress_bar:
                    time_s = time.time()
                    for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)
                        # The time_step is ticking.
                        x, x0t, _, _ = self.denoising_step(x, t=t, t_next=t_next, models=model,
                                            logvars=self.logvar,
                                            sampling_type=self.args.sample_type,
                                            b=self.betas,
                                            learn_sigma=self.learn_sigma)
                        progress_bar.update(1)
                    time_e = time.time()
                    print(f'{time_e - time_s} seconds')
                # Original image x0, reconstructed image x, latent encoding x_latent
                img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])

                tvu.save_image((x + 1) * 0.5, os.path.join(args.destination_folder,
                                                        f'{step}_1_rec_ninv{self.args.n_inv_step}.png'))
            
        # img_lat_pairs_dic = img_lat_pairs
        # return img_lat_pairs_dic, x_latent 
        return middle_h_dict, x_lat_dict

# Load pretrained model
reconstruction = reconstruction(args, config)
model = reconstruction.load_pretrained_model()
middle_h_dict, x_lat_dict = reconstruction.precompute_pairs(model)

# ---------check code---------
# bacause we test 5 images, so len(middle_h_dict) is 5. you can check by this code below.
# The paper is adding Delta_h to the middle_h of each time step of an image, so here I've saved the middle_h of the output for each time step
for key, value in middle_h_dict.items():
    print(f"{key}: {len(value)}")

print('Above is the middle_h of each image at each time step, and below is the lat_x of each image at each time step.')

for key, value in x_lat_dict.items():
    print(f"{key}: {len(value)}")
