DATASET_PATHS = {
	'FFHQ': '/hdd1/datasets/celeba_hq/',
	'AFHQ': '/hdd1/datasets/afhq/',
	'LSUN':  '/hdd1/datasets/lsun/',
    'IMAGENET': '/hdd1/datasets/imagenet/',
	'CUSTOM': '/hdd1/custom/',
	'CelebA_HQ_Dialog': '/hdd1/datasets/img_align_celeba/',
	'MetFACE': '/hdd1/datasets/metfaces/',

	'custom_train': "/home/ids/ziliu-24/Asyrp_official_original/CelebA_glasses-HQ/only_face_celeba_hq/train" ,
    'custom_test': "/home/ids/ziliu-24/Asyrp_official_original/CelebA_glasses-HQ/only_face_celeba_hq/test" ,
}


MODEL_PATHS = {
	'AFHQ': "pretrained_models/afhqdog_4m.pt",
	'FFHQ': "pretrained_models/ffhq_baseline.pt",
    'FFHQ_p2': "pretrained_models/ffhq_p2.pt",
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
    'IMAGENET': "pretrained_models/256x256_diffusion_uncond.pt",
	'shape_predictor': "pretrained_models/shape_predictor_68_face_landmarks.dat.bz2",
	'MetFACE' : "pretrained_models/metface_p2.pt",
    'CelebA_HQ_P2' : "pretrained_models/celebahq_p2.pt",
    'ir_se50': 'pretrained_models/model_ir_se50.pth',
    'previous_ckpt_path': None
}


