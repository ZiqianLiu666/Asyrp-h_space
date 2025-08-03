from datasets.images_dataset import ImagesDataset_diffusion
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os

def get_transform(apply_flip=False):
    tf = [
        transforms.Resize((256, 256)),
    ]
    # if apply_flip:
    #     tf.append(transforms.RandomHorizontalFlip(0.5))

    tf += [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
    return transforms.Compose(tf)

def build_dataset_paths(dataset_root, dataset_type='ffhq_glasses'):
    
    if dataset_type == 'ffhq_glasses':
        dataset_args = {
            'train_bg_dir': os.path.join(dataset_root, 'ffhq_glasses/train_bg'),
            'train_t_dir': os.path.join(dataset_root, 'ffhq_glasses/train_t'),
            'test_bg_dir': os.path.join(dataset_root, 'ffhq_glasses/test_bg'),
            'test_t_dir': os.path.join(dataset_root, 'ffhq_glasses/test_t'),
        }
        return dataset_args
    
    elif dataset_type == 'celeba_smile':

        dataset_args = {
            'train_bg_dir': os.path.join(dataset_root, 'CelebA-HQ/Smiling/train_smile_no'),
            'train_t_dir': os.path.join(dataset_root, 'CelebA-HQ/Smiling/train_smile_yes'),
            'test_bg_dir': os.path.join(dataset_root, 'CelebA-HQ/Smiling/test_smile_no'),
            'test_t_dir': os.path.join(dataset_root, 'CelebA-HQ/Smiling/test_smile_yes'),
        }
        return dataset_args
    
    elif dataset_type == 'celeba_gender':

        dataset_args = {
            'train_bg_dir': os.path.join(dataset_root, 'CelebA-HQ/Gender/train_male'),
            'train_t_dir': os.path.join(dataset_root, 'CelebA-HQ/Gender/train_female'),
            'test_bg_dir': os.path.join(dataset_root, 'CelebA-HQ/Gender/test_male'),
            'test_t_dir': os.path.join(dataset_root, 'CelebA-HQ/Gender/test_female'),
        }
        return dataset_args
    
    elif dataset_type == 'ffhq_age':

        dataset_args = {
            'train_bg_dir': os.path.join(dataset_root, 'ffhq_cs_age/train_old'),
            'train_t_dir': os.path.join(dataset_root, 'ffhq_cs_age/train_young'),
            'test_bg_dir': os.path.join(dataset_root, 'ffhq_cs_age/test_old'),
            'test_t_dir': os.path.join(dataset_root, 'ffhq_cs_age/test_young'),
        }
        return dataset_args
    
    elif dataset_type == 'tumor':

        dataset_args = {
            'train_bg_dir': os.path.join(dataset_root, 'BraTS2023_GLI/train_healthy'),
            'train_t_dir': os.path.join(dataset_root, 'BraTS2023_GLI/train_tumor'),
            'test_bg_dir': os.path.join(dataset_root, 'BraTS2023_GLI/test_healthy'),
            'test_t_dir': os.path.join(dataset_root, 'BraTS2023_GLI/test_tumor'),
        }
        return dataset_args
    
    # 以下都是测试图像
    elif dataset_type == 'infer_special_glasses':

        dataset_args = {
            'test_bg_dir': os.path.join(dataset_root, 'special_images/ffhq-glasses/background'),
            'test_t_dir': os.path.join(dataset_root, 'special_images/ffhq-glasses/glasses'),
        }
        return dataset_args
    
    elif dataset_type == 'infer_special_age':

        dataset_args = {
            'test_bg_dir': os.path.join(dataset_root, 'special_images/ffhq-age/old'),
            'test_t_dir': os.path.join(dataset_root, 'special_images/ffhq-age/young'),
        }
        return dataset_args
    
    elif dataset_type == 'infer_special_smile':

        dataset_args = {
            'test_bg_dir': os.path.join(dataset_root, 'special_images/celeba-smile/serious'),
            'test_t_dir': os.path.join(dataset_root, 'special_images/celeba-smile/smile'),
        }
        return dataset_args
    
    elif dataset_type == 'infer_special_gender':

        dataset_args = {
            'test_bg_dir': os.path.join(dataset_root, 'special_images/celeba-gender/male'),
            'test_t_dir': os.path.join(dataset_root, 'special_images/celeba-gender/female'),
        }
        return dataset_args
    
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def build_dataloaders(opts, apply_flip=False):

    transform = get_transform(apply_flip=apply_flip)
    dataset_args = build_dataset_paths(opts.dataset_root, opts.dataset_type)
    if opts.dataset_type in ["infer_special_glasses", "infer_special_age", "infer_special_smile", "infer_special_gender"]:
        test_bg_dataset = ImagesDataset_diffusion(dataset_args['test_bg_dir'], transform=transform)
        test_t_dataset = ImagesDataset_diffusion(dataset_args['test_t_dir'], transform=transform)
        test_bg_loader = DataLoader(test_bg_dataset, batch_size=opts.batch_size, shuffle=False)
        test_t_loader = DataLoader(test_t_dataset, batch_size=opts.batch_size, shuffle=False)
        print(f"Number of test samples: {len(test_bg_dataset) + len(test_t_dataset)}")
        return test_bg_loader, test_t_loader
    else:
        train_bg_dataset = ImagesDataset_diffusion(dataset_args['train_bg_dir'], transform=transform)
        train_t_dataset = ImagesDataset_diffusion(dataset_args['train_t_dir'], transform=transform)
        test_bg_dataset = ImagesDataset_diffusion(dataset_args['test_bg_dir'], transform=transform)
        test_t_dataset = ImagesDataset_diffusion(dataset_args['test_t_dir'], transform=transform)

        train_bg_loader = DataLoader(train_bg_dataset, batch_size=opts.batch_size, shuffle=False)
        train_t_loader = DataLoader(train_t_dataset, batch_size=opts.batch_size, shuffle=False)
        test_bg_loader = DataLoader(test_bg_dataset, batch_size=opts.batch_size, shuffle=False)
        test_t_loader = DataLoader(test_t_dataset, batch_size=opts.batch_size, shuffle=False)

        print(f"Number of training samples: {len(train_bg_dataset) + len(train_t_dataset)}")
        print(f"Number of test samples: {len(test_bg_dataset) + len(test_t_dataset)}")
        
        return train_bg_loader, train_t_loader, test_bg_loader, test_t_loader
