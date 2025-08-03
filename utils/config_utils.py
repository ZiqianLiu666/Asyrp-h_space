# utils/config_utils.py
import yaml
from argparse import Namespace

def dict2namespace(d):
    ns = Namespace()
    for k, v in d.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns

def load_diffusion_config(path='configs/ffhq.yml'):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return dict2namespace(config)
