# import os, sys 
# # 保证项目根目录在模块搜索路径中
# proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, proj_root)

# # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# #!/usr/bin/env python3
# import os
# import csv
# from copy import deepcopy
# import torch
# import torch.nn as nn
# import timm
# import torchvision.transforms as transforms
# from PIL import Image
# from options.train_options import TrainOptions
# from inference import inference as InferenceClass  # 你的原 inference 类

# def find_all_checkpoints(root_dir):
#     pts = []
#     for cur_dir, _, files in os.walk(root_dir):
#         for f in files:
#             if f.endswith('.pt') or f.endswith('.pth'):
#                 pts.append(os.path.join(cur_dir, f))
#     return sorted(pts)

# def evaluate_with_classifier(model, device, img_paths, gt_label, transform):
#     correct, total = 0, 0
#     model.eval()
#     with torch.no_grad():
#         for im_path in img_paths:
#             img = Image.open(im_path).convert('RGB')
#             x = transform(img).unsqueeze(0).to(device)
#             logits = model(x)
#             pred = logits.argmax(dim=1).item()
#             if pred == gt_label:
#                 correct += 1
#             total += 1
#     return correct, total

# if __name__ == '__main__':
#     # 解析命令行参数，仅用于扩散模型和数据集设置
#     parser = TrainOptions()
#     opts = parser.parser.parse_args()

#     # 1) 准备输出 CSV
#     results_root = opts.results_dir
#     os.makedirs(results_root, exist_ok=True)
#     csv_path = os.path.join(results_root, 'accuracy_report.csv')
#     with open(csv_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['iteration', 'recon_x_acc', 'recon_y_acc', 'swap_x2y_acc', 'swap_y2x_acc'])

#     # 2) 加载 SwinV2 Large 模型与权重
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     cls_model = timm.create_model(
#         'swinv2_large_window12to16_192to256',
#         pretrained=False,
#         img_size=256,
#         num_classes=2
#     )
#     # 请将下面路径改为你训练脚本里保存的 best_age_swinv2_large_256.pth 的实际位置
#     cls_weights_path = opts.classifier_path
#     cls_model.load_state_dict(torch.load(cls_weights_path, map_location=device))
#     cls_model.to(device)
#     cls_model.eval()

#     # 3) 图像预处理：与训练时的 val_transform 保持一致
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(256),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
#     ])

#     # 4) 遍历所有 CS checkpoint，依次推理并评估
#     cs_root = opts.cs_model_weights
#     all_pts = find_all_checkpoints(cs_root)

#     for pt_path in all_pts:
#         # 构造本次 inference 的输出目录
#         rel = os.path.relpath(os.path.dirname(pt_path), cs_root)
#         ckpt_name = os.path.splitext(os.path.basename(pt_path))[0]
#         save_dir = os.path.join(opts.results_dir, rel, ckpt_name)
#         os.makedirs(save_dir, exist_ok=True)

#         # 运行 Diffusion + CS 推理
#         this_opts = deepcopy(opts)
#         this_opts.cs_model_weights = pt_path
#         this_opts.results_dir      = save_dir
#         print(f"\n>> Inference for checkpoint: {pt_path}")
#         runner = InferenceClass(this_opts)
#         runner.inference()

#         # 评估本次输出
#         mapping = {
#             'recon_x': 0,
#             'recon_y': 1,
#             'swap_x2y': 1,
#             'swap_y2x': 0,
#         }
#         row = [int(ckpt_name.split('_')[1])]
#         for sub, gt in mapping.items():
#             folder = os.path.join(save_dir, sub)
#             if not os.path.isdir(folder):
#                 row.append(None)
#                 continue
#             imgs = [os.path.join(folder, f) for f in os.listdir(folder)
#                     if f.lower().endswith(('.png','.jpg','.jpeg'))]
#             c, t = evaluate_with_classifier(cls_model, device, imgs, gt, transform)
#             acc = c / t if t > 0 else None
#             row.append(acc)

#         # 将结果追加到 CSV
#         with open(csv_path, 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(row)

#         print(f"-> Iteration {row[0]} accuracy: "
#               f"recon_x={row[1]:.4f}, recon_y={row[2]:.4f}, "
#               f"swap_x2y={row[3]:.4f}, swap_y2x={row[4]:.4f}")

#     print(f"\n>> All done. Accuracy report at: {csv_path}")


#### 以下是DenseNet121 （glasses）的分类器
import os, sys 
# 保证项目根目录在模块搜索路径中
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, proj_root)

import os
import csv
from copy import deepcopy
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from PIL import Image
from options.train_options import TrainOptions
from inference import inference as InferenceClass  # 你的原 inference 类

def find_all_checkpoints(root_dir):
    pts = []
    for cur_dir, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.pt') or f.endswith('.pth'):
                pts.append(os.path.join(cur_dir, f))
    return sorted(pts)

def evaluate_with_classifier(model, device, img_paths, gt_label, transform):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for im_path in img_paths:
            img = Image.open(im_path).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).item()
            if pred == gt_label:
                correct += 1
            total += 1
    return correct, total

if __name__ == '__main__':
    # 解析命令行参数，仅用于扩散模型和数据集设置
    parser = TrainOptions()
    opts = parser.parser.parse_args()

    # 1) 准备输出 CSV
    results_root = opts.results_dir
    os.makedirs(results_root, exist_ok=True)
    csv_path = os.path.join(results_root, 'accuracy_report.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'recon_x_acc', 'recon_y_acc', 'swap_x2y_acc', 'swap_y2x_acc'])

    # 2) 加载 DenseNet121 模型与权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # MODIFIED: Changed the model from swinv2_large to densenet121
    cls_model = timm.create_model(
        'densenet121',
        pretrained=False,     # Loading your own custom weights
        num_classes=2         # Assuming the task is still binary classification
    )
    
    # 请将下面路径改为你为 Densenet121 训练并保存的权重 .pth 文件的实际位置
    cls_weights_path = opts.classifier_path
    cls_model.load_state_dict(torch.load(cls_weights_path, map_location=device))
    cls_model.to(device)
    cls_model.eval()

    # 3) 图像预处理：与训练时的 val_transform 保持一致
    # IMPORTANT: Ensure these transforms, especially Normalize, match your DenseNet121 training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # 4) 遍历所有 CS checkpoint，依次推理并评估
    cs_root = opts.cs_model_weights
    all_pts = find_all_checkpoints(cs_root)

    for pt_path in all_pts:
        # 构造本次 inference 的输出目录
        rel = os.path.relpath(os.path.dirname(pt_path), cs_root)
        ckpt_name = os.path.splitext(os.path.basename(pt_path))[0]
        save_dir = os.path.join(opts.results_dir, rel, ckpt_name)
        os.makedirs(save_dir, exist_ok=True)

        # 运行 Diffusion + CS 推理
        this_opts = deepcopy(opts)
        this_opts.cs_model_weights = pt_path
        this_opts.results_dir      = save_dir
        print(f"\n>> Inference for checkpoint: {pt_path}")
        runner = InferenceClass(this_opts)
        runner.inference()

        # 评估本次输出
        mapping = {
            'recon_x': 0,
            'recon_y': 1,
            'swap_x2y': 1,
            'swap_y2x': 0,
        }
        row = [int(ckpt_name.split('_')[1])]
        for sub, gt in mapping.items():
            folder = os.path.join(save_dir, sub)
            if not os.path.isdir(folder):
                row.append(None)
                continue
            imgs = [os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(('.png','.jpg','.jpeg'))]
            c, t = evaluate_with_classifier(cls_model, device, imgs, gt, transform)
            acc = c / t if t > 0 else None
            row.append(acc)

        # 将结果追加到 CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"-> Iteration {row[0]} accuracy: "
              f"recon_x={row[1]:.4f}, recon_y={row[2]:.4f}, "
              f"swap_x2y={row[3]:.4f}, swap_y2x={row[4]:.4f}")

    print(f"\n>> All done. Accuracy report at: {csv_path}")