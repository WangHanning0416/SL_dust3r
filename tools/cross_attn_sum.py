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


def count_layer_matches(layer_dir):
    """统计单个层中匹配成功的Patch数量"""
    # 加载并处理注意力图
    attn_file = os.path.join(layer_dir, CONFIG["attn_filename"])
    if not os.path.exists(attn_file):
        raise FileNotFoundError(f"文件不存在：{attn_file}")
    
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
    
    # 统计匹配成功的Patch数量
    match_count = 0
    for target_i in range(CONFIG["patch_size"]):
        for target_j in range(CONFIG["patch_size"]):
            # 找到Source上注意力最大的Patch位置
            source_attn = attn_final[target_i, target_j, :, :]
            max_source_i, max_source_j = np.unravel_index(np.argmax(source_attn), source_attn.shape)
            # 判断是否匹配（Target的(i,j)对应Source的(i,j)）
            if abs(max_source_i - target_i) <= 1 and abs(max_source_j - target_j) <= 1:
                match_count += 1
    return match_count


def main():
    debug_print("开始统计每层匹配成功的Patch数量...")
    layer_counts = []  # 存储 (层号, 匹配数)
    
    for layer_num in range(12):
        layer_dir = os.path.join(CONFIG["npy_root_dir"], f"layer_{layer_num}")
        if not os.path.isdir(layer_dir):
            debug_print(f"警告：层目录不存在，跳过 -> layer_{layer_num}")
            continue
        
        try:
            match_count = count_layer_matches(layer_dir)
            layer_counts.append({"layer": layer_num, "match_count": match_count})
            debug_print(f"layer_{layer_num} 匹配成功：{match_count}个Patch")
        except Exception as e:
            debug_print(f"处理layer_{layer_num}失败：{str(e)}，跳过")
            continue
    
    # 保存结果为CSV
    if layer_counts:
        df = pd.DataFrame(layer_counts)
        df.to_csv(CONFIG["save_path"], index=False)
        debug_print(f"结果已保存至：{CONFIG['save_path']}")
    else:
        debug_print("无有效数据，未生成结果文件")


if __name__ == "__main__":
    main()