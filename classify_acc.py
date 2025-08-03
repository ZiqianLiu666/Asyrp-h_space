# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os
# import sys
# import pandas as pd

# def compute_metrics(df: pd.DataFrame):
#     # real_acc: 在所有样本上，Real==GT 的比例
#     total = len(df)
#     real_correct = (df['real'] == df['GT']).sum()
#     real_acc = real_correct / total if total > 0 else 0.0

#     # 只保留 Real 正确的样本，用于后续 Recon、Swap 的评估
#     df_valid = df[df['real'] == df['GT']]
#     valid_n = len(df_valid)

#     # recon_acc: Recon==GT
#     if valid_n > 0:
#         recon_correct = (df_valid['recon'] == df_valid['GT']).sum()
#         recon_acc = recon_correct / valid_n
#         # swap 对于 GT=1 要预测 0，对 GT=0 要预测 1，相当于 Swap == (1-GT)
#         swap_correct = (df_valid['swap'] == (1 - df_valid['GT'])).sum()
#         swap_acc = swap_correct / valid_n
#     else:
#         recon_acc = 0.0
#         swap_acc = 0.0

#     return real_acc, recon_acc, swap_acc

# def main(root_dir: str, out_csv: str):
#     rows = []
#     for dirpath, _, filenames in os.walk(root_dir):
#         if 'pred_label.csv' in filenames:
#             csv_path = os.path.join(dirpath, 'pred_label.csv')
#             try:
#                 df = pd.read_csv(csv_path)
#                 print(f"DEBUG: columns in {csv_path!r}:", df.columns.tolist())
#             except Exception as e:
#                 print(f"Warning: 无法读取 {csv_path}，跳过。原因: {e}", file=sys.stderr)
#                 continue

#             # 确保必要列存在
#             for col in ['GT', 'real', 'recon', 'swap']:
#                 if col not in df.columns:
#                     print(f"Warning: {csv_path} 中缺少列 {col}，跳过。", file=sys.stderr)
#                     continue

#             real_acc, recon_acc, swap_acc = compute_metrics(df)

#             # dataset name 使用相对路径
#             rel_path = os.path.relpath(dirpath, root_dir)
#             rows.append({
#                 'dataset': rel_path,
#                 'real_acc': real_acc,
#                 'recon_acc': recon_acc,
#                 'swap_acc': swap_acc
#             })

#     # 汇总并写出
#     result_df = pd.DataFrame(rows, columns=['dataset', 'real_acc', 'recon_acc', 'swap_acc'])
#     result_df.to_csv(out_csv, index=False, float_format='%.4f')
#     print(f"All done! 汇总写入 {out_csv}")

# if __name__ == '__main__':
#     if len(sys.argv) != 3:
#         print("用法: python compute_acc.py <root_dir> <out_csv>", file=sys.stderr)
#         sys.exit(1)
#     root_dir = sys.argv[1]
#     out_csv = sys.argv[2]
#     main(root_dir, out_csv)
# python classify_acc.py \
#     home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/pred \
#         /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/pred/summary_acc.csv

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd


def compute_metrics(df: pd.DataFrame):
    # 初始化指标字典
    metrics = {
        'real_acc_x': 0.0,
        'real_acc_y': 0.0,
        'recon_acc_x': 0.0,
        'recon_acc_y': 0.0,
        'swap_acc_x': 0.0,
        'swap_acc_y': 0.0,
    }
    
    # 分别按 GT 值计算 real_acc_x 和 real_acc_y
    for cls, suffix in [(0, 'x'), (1, 'y')]:
        # 该类样本总数
        total_cls = (df['GT'] == cls).sum()
        if total_cls > 0:
            # real 正确的数量
            real_correct_cls = ((df['GT'] == cls) & (df['real'] == df['GT'])).sum()
            metrics[f'real_acc_{suffix}'] = real_correct_cls / total_cls
        else:
            metrics[f'real_acc_{suffix}'] = 0.0

    # 只保留 real 正确的样本，用于后续 Recon、Swap 的评估
    df_valid = df[df['real'] == df['GT']]

    for cls, suffix in [(0, 'x'), (1, 'y')]:
        # 该类在有效集中的数量
        valid_cls = (df_valid['GT'] == cls).sum()
        if valid_cls > 0:
            # recon 正确
            recon_correct = ((df_valid['GT'] == cls) & (df_valid['recon'] == df_valid['GT'])).sum()
            metrics[f'recon_acc_{suffix}'] = recon_correct / valid_cls
            # swap 对于 GT=1 要预测 0，对 GT=0 要预测 1
            swap_correct = ((df_valid['GT'] == cls) & (df_valid['swap'] == (1 - df_valid['GT']))).sum()
            metrics[f'swap_acc_{suffix}'] = swap_correct / valid_cls
        else:
            metrics[f'recon_acc_{suffix}'] = 0.0
            metrics[f'swap_acc_{suffix}'] = 0.0

    return metrics


def main(root_dir: str, out_csv: str):
    rows = []
    for dirpath, _, filenames in os.walk(root_dir):
        if 'pred_label.csv' in filenames:
            csv_path = os.path.join(dirpath, 'pred_label.csv')
            try:
                df = pd.read_csv(csv_path)
                print(f"DEBUG: columns in {csv_path!r}: {df.columns.tolist()}")
            except Exception as e:
                print(f"Warning: 无法读取 {csv_path}，跳过。原因: {e}", file=sys.stderr)
                continue

            # 确保必要列存在
            required_cols = {'GT', 'real', 'recon', 'swap'}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                print(f"Warning: {csv_path} 中缺少列 {missing}，跳过。", file=sys.stderr)
                continue

            metrics = compute_metrics(df)

            # dataset name 使用相对路径
            rel_path = os.path.relpath(dirpath, root_dir)
            row = {'dataset': rel_path}
            row.update(metrics)
            rows.append(row)

    # 汇总并写出
    cols = ['dataset',
            'real_acc_x', 'real_acc_y',
            'recon_acc_x', 'recon_acc_y',
            'swap_acc_x',  'swap_acc_y']
    result_df = pd.DataFrame(rows, columns=cols)
    result_df.to_csv(out_csv, index=False, float_format='%.4f')
    print(f"All done! 汇总写入 {out_csv}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python compute_acc_split.py <root_dir> <out_csv>", file=sys.stderr)
        sys.exit(1)
    root_dir = sys.argv[1]
    out_csv = sys.argv[2]
    main(root_dir, out_csv)
# python classify_acc.py \
#     home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/pred \
#         /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/pred/summary_acc.csv

# python classify_acc.py \
#     /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/pred/ffhq_glasses/specific_encoder/inv=100/fused/t_edit999/conv/layer_12/lr0.00001 \
#         /home/ids/ziliu-24/diffu_asyrp_CA_optim_eachtimestep_revised/pred/ffhq_glasses/specific_encoder/inv=100/fused/t_edit999/conv/layer_12/lr0.00001/summary_acc.csv