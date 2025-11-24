import os
import numpy as np
import torch
import pandas as pd


# --------------------------
# 配置参数（修改为你的路径）
# --------------------------
CONFIG = {
    "npy_root_dir": "/data3/hanning/dust3r/cross_attn_npy/",  # 注意力图根目录
    "save_path": "./layer_match_counts.csv",  # 结果保存路径
    "patch_size": 14,  # 14x14=196个Patch
    "attn_filename": "img1_to_img2_attn.npy"  # 每层注意力图文件名
}


def debug_print(message):
    from datetime import datetime
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")


def merge_attn_map(attn_maps):
    """合并多头注意力（仅做均值）"""
    attn_maps = torch.stack(attn_maps, dim=1)
    return torch.mean(attn_maps, dim=(1, 2))  # 对“头”维度求均值


def process_single_layer(layer_dir):
    """处理单个层，返回处理后的注意力矩阵（用于统计或融合）"""
    attn_file = os.path.join(layer_dir, CONFIG["attn_filename"])
    if not os.path.exists(attn_file):
        raise FileNotFoundError(f"文件不存在：{attn_file}")
    
    # 加载并合并多头注意力
    attn_np = np.load(attn_file)
    attn_tensor = torch.from_numpy(attn_np)
    attn_merged = merge_attn_map([attn_tensor])  # 合并多头
    attn_merged_np = attn_merged.cpu().numpy()
    
    # 调整形状为 [Target_i, Target_j, Source_i, Source_j]
    expected_shape = (1, CONFIG["patch_size"], CONFIG["patch_size"], CONFIG["patch_size"], CONFIG["patch_size"])
    if attn_merged_np.size != np.prod(expected_shape):
        raise ValueError(f"形状不匹配：{attn_merged_np.size} vs {np.prod(expected_shape)}")
    attn_final = attn_merged_np.reshape(expected_shape)[0]
    attn_final = np.flipud(attn_final)  # 垂直翻转
    return attn_final


def count_matches(attn_matrix):
    """统计注意力矩阵中匹配成功的Patch数量（复用逻辑，支持单层和融合层）"""
    match_count = 0
    patch_size = CONFIG["patch_size"]
    for target_i in range(patch_size):
        for target_j in range(patch_size):
            # 找到Source上注意力最大的Patch位置
            source_attn = attn_matrix[target_i, target_j, :, :]
            max_source_i, max_source_j = np.unravel_index(np.argmax(source_attn), source_attn.shape)
            # 判断是否匹配（Target的(i,j)与Source的(i,j)在±1范围内）
            if abs(max_source_i - target_i) <= 1 and abs(max_source_j - target_j) <= 1:
                match_count += 1
    return match_count


def main():
    debug_print("开始统计每层及融合层的匹配成功Patch数量...")
    layer_counts = []  # 存储 (层号, 匹配数)
    valid_layer_matrices = []  # 存储有效层的注意力矩阵（用于融合）
    
    # 1. 统计单层匹配数，并收集有效层矩阵
    for layer_num in range(12):
        layer_dir = os.path.join(CONFIG["npy_root_dir"], f"layer_{layer_num}")
        if not os.path.isdir(layer_dir):
            debug_print(f"警告：层目录不存在，跳过 -> layer_{layer_num}")
            continue
        
        try:
            # 处理单层并统计匹配数
            attn_matrix = process_single_layer(layer_dir)
            match_count = count_matches(attn_matrix)
            layer_counts.append({"layer": layer_num, "match_count": match_count})
            debug_print(f"layer_{layer_num} 匹配成功：{match_count}个Patch")
            
            # 收集矩阵用于后续融合
            valid_layer_matrices.append(attn_matrix)
        
        except Exception as e:
            debug_print(f"处理layer_{layer_num}失败：{str(e)}，跳过")
            continue
    
    # 2. 计算融合层（所有有效层的均值）的匹配数
    if valid_layer_matrices:
        # 对有效层矩阵取均值（融合）
        fused_matrix = np.mean(np.stack(valid_layer_matrices, axis=0), axis=0)
        debug_print(f"融合完成，有效层数：{len(valid_layer_matrices)}，融合矩阵形状：{fused_matrix.shape}")
        
        # 统计融合层的匹配数（层号标记为12，区分于0-11的单层）
        fused_match_count = count_matches(fused_matrix)
        layer_counts.append({"layer": 12, "match_count": fused_match_count})
        debug_print(f"融合层（12层均值）匹配成功：{fused_match_count}个Patch")
    else:
        debug_print("无有效层可融合，跳过融合层统计")
    
    # 3. 保存结果为CSV
    if layer_counts:
        # 按层号排序（0-11 + 12）
        df = pd.DataFrame(layer_counts).sort_values(by="layer")
        df.to_csv(CONFIG["save_path"], index=False)
        debug_print(f"结果已保存至：{CONFIG['save_path']}")
    else:
        debug_print("无有效数据，未生成结果文件")


if __name__ == "__main__":
    main()