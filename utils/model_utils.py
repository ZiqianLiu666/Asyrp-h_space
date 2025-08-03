# utils/model_utils.py
import torch
import numpy as np
from models.guided_diffusion.script_util import guided_Diffusion
from models.improved_ddpm.script_util import i_DDPM
from models.guided_diffusion.unet import UNet_h_encoder, UNet_h_decoder  # adjust import to your file structure
from criteria import id_loss
from criteria.lpips.lpips import LPIPS
# from base_functions.Asyrp_functions import reconstruction  # or your new diffusion_utils
from configs.paths_config import MODEL_PATHS


# MODEL_PATHS = {
# 	'AFHQ': "pretrained_models/afhqdog_4m.pt",
# 	'FFHQ': "pretrained_models/ffhq_baseline.pt",
#     'FFHQ': "pretrained_models/ffhq_baseline.pt",
# 	'ir_se50': 'pretrained_models/model_ir_se50.pth',
#     'IMAGENET': "pretrained_models/256x256_diffusion_uncond.pt",
# 	'shape_predictor': "pretrained_models/shape_predictor_68_face_landmarks.dat.bz2",
# 	'MetFACE' : "pretrained_models/metface_p2.pt",
#     'CelebA_HQ_P2' : "pretrained_models/celebahq_p2.pt",
#     'ir_se50': 'pretrained_models/model_ir_se50.pth',
#     'previous_ckpt_path': None
# }

def load_pretrained_diffu(config, opts, device):
    if opts.dataset_type == 'ffhq_glasses' or "ffhq_age":
        dataset_name = "FFHQ"
    elif opts.dataset_type == 'tumor':
        dataset_name = "tumor"
    elif opts.dataset_type == 'celeba_gender' or 'celeba_smile':
        dataset_name = "CelebA_HQ_P2"
        
    # 下面是测试图像
    elif opts.dataset_type == "infer_special_glasses" or "infer_special_age":
        dataset_name = "FFHQ"
    elif opts.dataset_type == "infer_special_gender" or "infer_special_smile":
        dataset_name = "CelebA_HQ_P2"
    elif opts.dataset_type == 'infer_special_tumor':
        dataset_name = "tumor"

    if dataset_name in ["FFHQ", "FFHQ_p2", "AFHQ", "IMAGENET", "tumor"]:
        model = i_DDPM(dataset_name)
        if opts.diffu_weights is not None:
            ckpt = torch.load(opts.diffu_weights)
        else:
            ckpt = torch.load(MODEL_PATHS[dataset_name])
        learn_sigma = True
        print(f"[INFO] Loaded improved diffusion model for {dataset_name}")
    elif dataset_name in ["MetFACE", "CelebA_HQ_P2"]:
        model = guided_Diffusion(dataset_name)
        if opts.diffu_weights is not None:
            ckpt = torch.load(opts.diffu_weights)
        else:
            ckpt = torch.load(MODEL_PATHS[dataset_name])
        learn_sigma = True
        print(f"[INFO] Loaded guided diffusion model for {dataset_name}")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    #model = torch.nn.DataParallel(model)

    encoder = UNet_h_encoder(model)
    decoder = UNet_h_decoder(model)

    return model, encoder, decoder, learn_sigma



def load_diffusion_model(opts, config, device):
    recon = diffusion_reconstruction(opts, config, device)
    
    # Fixed this line:
    model, encoder, decoder, learn_sigma = load_pretrained_diffu(config, opts, device)
    
    seq_inv, seq_inv_next = recon.generate_time_steps(opts.n_inv_step, opts.t_0)

    return model, encoder, decoder, {
        'recon': recon,
        'seq_inv': seq_inv,
        'seq_inv_next': seq_inv_next,
        'logvar': recon.logvar,
        'betas': recon.betas,
        'sample_type': opts.sample_type,
        'learn_sigma': learn_sigma  # ← use this from the pretrained loader
    }


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
    
class diffusion_reconstruction:
    def __init__(self, args, config, device):
        self.device = device
        self.args = args
        self.config = config
        print(self.config)

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

    def noise_injection(self, xt, t, t_next, *,
                    et,
                    logvars,
                    b,
                    sampling_type='ddim',
                    eta=0.0,
                    learn_sigma=False,
                    dt_lambda=1,
                    dt_end = 999,
                    ):

        # print(middle_h.shape)
        if learn_sigma:
            et, logvar_learned = torch.split(et, et.shape[1] // 2, dim=1)
            logvar = logvar_learned
        else:
            logvar = utils.extract(logvars, t, xt.shape)

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

        # Warigari by young-hyun, Not in the paper
        else:
            # will be updated
            # Returns the updated latent encoding xt_next, the estimated denoised image x0_t, the editing parameter delta_h, and the middle hidden feature middle_h.
            return xt_next, x0_t


    def reverse_denoising(self, xt, t, t_next, *,
                    et,
                    logvars,
                    b,
                    sampling_type='ddim',
                    eta=0.0,
                    learn_sigma=False,
                    index=None,
                    dt_lambda=1,
                    dt_end = 999,
                    ):

        if learn_sigma:
            et, logvar_learned = torch.split(et, et.shape[1] // 2, dim=1)
            logvar = logvar_learned
        else:
            logvar = utils.extract(logvars, t, xt.shape)

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
            # if index is not None:
            #     x0_t = (xt - et_modified * (1 - at).sqrt()) / at.sqrt()
            # else:
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

        return xt_next, x0_t


    def inject_noise_batch(self, encoder, decoder, batch_image, seq_inv, seq_inv_next):
        x = batch_image

        with torch.no_grad():
            for i, j in zip(seq_inv_next[1:], seq_inv[1:]):
                t = torch.full((x.size(0),), i, device=x.device)
                t_prev = torch.full((x.size(0),), j, device=x.device)

                # Step through encoder → decoder to get et
                middle_h, hs, emb = encoder(x, t)
                et = decoder(middle_h, hs, emb, x)  # Only returns et, not et_modified

                # Run forward noise addition
                x, _ = self.noise_injection(
                    xt=x, t=t, t_next=t_prev,
                    et=et,
                    logvars=self.logvar,
                    b=self.betas,
                    sampling_type='ddim',
                    eta=0,
                    learn_sigma=self.learn_sigma
                )

        return x  # Noisy version of input (e.g., at timestep T)

    def generation_batch_for_TEST(self, encoder, decoder, batch_xT, seq_inv, seq_inv_next, h_modified_list=None, index=None):
        x = batch_xT
        middle_h_list = []
        xt_list = []
        with torch.no_grad():
            for step, (i, j) in enumerate(zip(reversed(seq_inv), reversed(seq_inv_next))):
                t = torch.full((x.size(0),), i, device=x.device)
                t_next = torch.full((x.size(0),), j, device=x.device)

                # Run encoder → decoder
                middle_h, hs, emb = encoder(x, t)

                # Replace only if current timestep is in `index`
                if h_modified_list is not None and index is not None and step in index:
                    print(f"replace at timestep {i} → {j} ")
                    middle_h = h_modified_list[step]

                et = decoder(middle_h, hs, emb, x)  # Full decoder output
                
                # Denoising step using et_modified
                x, _ = self.reverse_denoising(
                    xt=x, t=t, t_next=t_next,
                    et=et,
                    logvars=self.logvar,
                    b=self.betas,
                    sampling_type='ddim',
                    learn_sigma=self.learn_sigma
                )
                xt_list.append(x)
                middle_h_list.append(middle_h)

        return x, xt_list, middle_h_list

    def reverse_generation_batch(self, encoder, decoder, batch_xT, seq_inv, seq_inv_next, h_modified_list=None, index=None):
        x = batch_xT
        with torch.no_grad():
            for step, (i, j) in enumerate(zip(reversed(seq_inv), reversed(seq_inv_next))):
                t = torch.full((x.size(0),), i, device=x.device)
                t_next = torch.full((x.size(0),), j, device=x.device)

                # Run encoder → decoder
                middle_h, hs, emb = encoder(x, t)

                # Replace only if current timestep is in `index`
                if h_modified_list is not None and index is not None and step in index:
                    print(f"replace at timestep {i} → {j} ")
                    middle_h = h_modified_list[step]

                et = decoder(middle_h, hs, emb, x)  # Full decoder output
                
                # Denoising step using et_modified
                x, _ = self.reverse_denoising(
                    xt=x, t=t, t_next=t_next,
                    et=et,
                    logvars=self.logvar,
                    b=self.betas,
                    sampling_type='ddim',
                    learn_sigma=self.learn_sigma
                )

        return x

def load_cs_model(model_path, opts, device, is_eval=False):
    # it's regular middle_h=(C+S)
    if opts.cs_net_type == "equalized_mlp":
        from models.cs_models.no_fused.cs_models_oneforC_oneforS import MappingNetwork_cs_Unet
        print("[INFO] Using CS model: Equalized MLP-based Mapping Network")
        model = MappingNetwork_cs_Unet(opts).to(device)
    elif opts.cs_net_type == "conv":
        from models.cs_models.no_fused.cs_models_oneforC_oneforS import ConvMappingNetwork_cs_Unet
        print("[INFO] Using CS model: Conv2D-based Mapping Network")
        model = ConvMappingNetwork_cs_Unet(opts).to(device)
        
    # below is fused network middle_h=fused(C, S)
    elif opts.cs_net_type == "equalized_mlp_fused":
        from models.cs_models.fused.cs_models_oneforC_oneforS_fused import MappingNetwork_cs_Unet_fused
        print("[INFO] Using CS model: Asyrp-style Directional Decomposer")
        model = MappingNetwork_cs_Unet_fused(opts).to(device)
    elif opts.cs_net_type == "conv_fused":
        from models.cs_models.fused.cs_models_oneforC_oneforS_fused import ConvMappingNetwork_cs_Unet_fused
        print("[INFO] Using CS model: Asyrp-style Directional Decomposer")
        model = ConvMappingNetwork_cs_Unet_fused(opts).to(device)

    else:
        raise ValueError(f"[ERROR] Unknown cs_net_type: {opts.cs_net_type}")

    if model_path is not None:
        print(f"[INFO] Loading CS model weights from: {model_path}")
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict_cs_enc'], strict=False)
    
    if is_eval:
        model.eval()

    return model
        
    
def load_cs_model_specific(model_path, opts, device, is_eval=False):
    if opts.cs_net_type == "specific_mlp":
        from models.cs_models.no_fused.cs_models_single_salient_specific_encoder import specific_mlp
        print("[INFO] Using CS model: Equalized MLP-based Mapping Network")
        model = specific_mlp(opts).to(device)
    elif opts.cs_net_type == "specific_conv":
        from models.cs_models.no_fused.cs_models_single_salient_specific_encoder import specific_conv
        print("[INFO] Using CS model: Conv2D-based Mapping Network")
        model = specific_conv(opts).to(device)
        
    # below is fused network middle_h=fused(C, S)
    elif opts.cs_net_type == "specific_mlp_fused":
        from models.cs_models.fused.cs_models_single_salient_specific_encoder_fused import specific_mlp_fused
        print("[INFO] Using CS model: Conv2D-based Mapping Network")
        model = specific_mlp_fused(opts).to(device)
    elif opts.cs_net_type == "specific_conv_fused":
        from models.cs_models.fused.cs_models_single_salient_specific_encoder_fused import specific_conv_fused
        print("[INFO] Using CS model: Conv2D-based Mapping Network")
        model = specific_conv_fused(opts).to(device)

    else:
        raise ValueError(f"[ERROR] Unknown cs_net_type: {opts.cs_net_type}")

    if model_path is not None:
        print(f"[INFO] Loading CS model weights from: {model_path}")
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict_cs_enc'], strict=False)
    
    if is_eval:
        model.eval()

    return model
        
        
        
def load_cs_model_one_encoder(model_path, opts, device, is_eval=False):
    if opts.cs_net_type == "one_encoder_mlp":
        from models.cs_models.no_fused.cs_models_single_salient_one_encoder import one_encoder_mlp
        print("[INFO] Using CS model: Equalized MLP-based Mapping Network")
        model = one_encoder_mlp(opts).to(device)
    elif opts.cs_net_type == "one_encoder_conv":
        from models.cs_models.no_fused.cs_models_single_salient_one_encoder import one_encoder_conv
        print("[INFO] Using CS model: Conv2D-based Mapping Network")
        model = one_encoder_conv(opts).to(device)
        
    # below is fused network middle_h=fused(C, S)
    elif opts.cs_net_type == "one_encoder_mlp_fused":
        from models.cs_models.fused.cs_models_single_salient_one_encoder_fused import one_encoder_mlp_fused
        print("[INFO] Using CS model: Conv2D-based Mapping Network")
        model = one_encoder_mlp_fused(opts).to(device)
    elif opts.cs_net_type == "one_encoder_conv_fused":
        from models.cs_models.fused.cs_models_single_salient_one_encoder_fused import one_encoder_conv_fused
        print("[INFO] Using CS model: Conv2D-based Mapping Network")
        model = one_encoder_conv_fused(opts).to(device)
        
    else:
        raise ValueError(f"[ERROR] Unknown cs_net_type: {opts.cs_net_type}")

    if model_path is not None:
        print(f"[INFO] Loading CS model weights from: {model_path}")
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict_cs_enc'], strict=False)

    if is_eval:
        model.eval()

    return model


    
def load_id_lpips_models(opts, device):

    lpips_loss = LPIPS(net_type='alex').to(device).eval()
    id_loss_fn = id_loss.IDLoss().to(device).eval()
    
    return lpips_loss, id_loss_fn

