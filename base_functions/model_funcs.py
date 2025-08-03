

def load_diffusion_model(opts, is_eval=True):
    if self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
        model = i_DDPM(self.config.data.dataset)
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
        raise ValueError("Dataset not implemented")

    model.load_state_dict(init_ckpt, strict=False)
    model = model.to(self.device)
    model = torch.nn.DataParallel(model)
    
    # Create encoder/decoder from the inner UNet (DataParallel module stores model in .module)
    encoder = UNet_h_encoder(model.module)
    decoder = UNet_h_decoder(model.module)
    if is_eval:
        model.eval()
        encoder.eval()
        decoder.eval()
    return model, encoder, decoder







