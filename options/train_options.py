import yaml
from argparse import ArgumentParser, Namespace
import os
from IPython import get_ipython

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except NameError:
        return False

def dict2namespace(d):
    """Convert dict to namespace (like argparse Namespace)"""
    ns = Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, dict2namespace(v))
        else:
            setattr(ns, k, v)
    return ns

class TrainOptions:

    def __init__(self, argv=None):  # <- Accept argv optionally (for Jupyter)
        self.parser = ArgumentParser()
        self.initialize()
        self.argv = argv

    def initialize(self):
        # Basic options
        self.parser.add_argument('--exp_scheme', default='baseline', type=str)
        self.parser.add_argument('--dataset_root', default='/home/ids/ziliu-24/diffu_asyrp_CA_revised', type=str)
        self.parser.add_argument('--dataset_type', default='ffhq_glasses', type=str)
        self.parser.add_argument('--results_dir', default='./results', type=str)
        self.parser.add_argument('--seed', default=99, type=int)

        # Diffusion-specific args
        self.parser.add_argument('--diffu_weights', type=str, default='pretrained_models/ffhq_baseline.pt')
        self.parser.add_argument('--n_inv_step', type=int, default=40)
        self.parser.add_argument('--t_0', type=int, default=999)
        self.parser.add_argument('--sample_type', type=str, default='ddim')
        self.parser.add_argument('--diffu_config_path', type=str, default='configs/ffhq.yml')
        
        # CS network
        self.parser.add_argument('--cs_model_weights', type=str, default=None)
        self.parser.add_argument('--cs_net_type', default='conv', type=str, choices=['equalized_mlp', 'conv', 'equalized_mlp_fused', 'conv_fused', \
            "one_encoder_mlp", "one_encoder_conv", "one_encoder_mlp_fused", "one_encoder_conv_fused", \
            "specific_mlp", "specific_conv", "specific_mlp_fused", "specific_conv_fused"])
        self.parser.add_argument('--latent_dim', default=512, type=int)
        self.parser.add_argument('--n_cs_layers', default=12, type=int)

        # Training
        self.parser.add_argument('--optim_name', default='adam', type=str)
        self.parser.add_argument('--learning_rate', default=0.01, type=float)
        self.parser.add_argument('--max_steps', default=500000, type=int)
        self.parser.add_argument('--batch_size', default=4, type=int)
        self.parser.add_argument('--image_interval', default=10000, type=int)
        self.parser.add_argument('--print_interval', default=10000, type=int)
        self.parser.add_argument('--log_interval', default=10000, type=int)
        self.parser.add_argument('--val_interval', default=10000, type=int)
        self.parser.add_argument('--save_interval', default=10000, type=int)

        self.parser.add_argument('--train_diffu_encoder', default=False, type=bool)
        self.parser.add_argument('--train_diffu_decoder', default=False, type=bool)
        self.parser.add_argument('--lr_mlp', default=1e-2, type=float)
        self.parser.add_argument('--lr_enc', default=1e-4, type=float)
        self.parser.add_argument('--lr_dec', default=1e-4, type=float)

        # Loss weights
        self.parser.add_argument('--id_lambda', default=1.0, type=float)
        self.parser.add_argument('--pix_lambda', default=1.0, type=float)
        self.parser.add_argument('--lpips_lambda', default=1.0, type=float)
        self.parser.add_argument('--lat_lambda', default=1.0, type=float)
        self.parser.add_argument('--sbg_lambda', default=1.0, type=float)
        self.parser.add_argument('--dao_lambda', default=1.0, type=float)
        self.parser.add_argument('--ortho_lambda', default=1.0, type=float)
        self.parser.add_argument('--swap_latent_lambda', default=1.0, type=float)
        self.parser.add_argument('--attr_lambda', default=1.0, type=float)
        self.parser.add_argument('--cls_lambda', default=1.0, type=float)
        
        self.parser.add_argument('--t_end_edit', default=500, type=int)
        self.parser.add_argument('--classifier_path', default='adam', type=str)

    def parse(self):
        if is_notebook():
            args = self.parser.parse_args([])  # Ignore sys.argv
        else:
            args = self.parser.parse_args()

        return args
