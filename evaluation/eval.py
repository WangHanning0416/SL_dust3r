import os
import numpy as np
import argparse
from os.path import join
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="点云对齐：平移（中位数中心）+ 尺度（平均值）适配GT")
    parser.add_argument("--root_dir", type=str, default="/data3/hanning/dust3r/result", help="结果根目录")
    parser.add_argument("--summary_path", type=str, default="/data3/hanning/dust3r/evaluation/scene_summary.txt", help="汇总结果路径")
    return parser.parse_args()


def calculate_loss(gt_pts, pred_pts):
    loss = np.linalg.norm(pred_pts - gt_pts, axis=1)
    return np.mean(loss)

def calcu_single_frame_loss(gt_pts3d, pred_pts3d, valid_mask):
    gt_pts = gt_pts3d[valid_mask].reshape(-1, 3)
    pred_pts = pred_pts3d[valid_mask].reshape(-1, 3)

    if len(gt_pts) < 100:
        return None

    gt_center = np.nanmedian(gt_pts, axis=0, keepdims=True) 
    gt_pts -= gt_center  
    pred_center = np.nanmedian(pred_pts, axis=0, keepdims=True)
    pred_pts -= pred_center 

    gt_scale = np.nanmedian(np.linalg.norm(gt_pts, axis=1)) 
    pred_scale = np.nanmedian(np.linalg.norm(pred_pts, axis=1)) 

    if gt_scale <= 1e-6:
        return None

    pred_pts *= gt_scale / pred_scale

    frame_loss = calculate_loss(gt_pts, pred_pts)
    return frame_loss


def main():
    args = parse_args()
    np.random.seed(42)
    
    scenes = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]

    summary_lines = [
        "场景 | GT尺度（平均值）归一化-中位数误差\n",
        "-" * 50 + "\n"
    ]
    
    for scene_name in scenes:
        npy_dir = join(args.root_dir, scene_name, "npy")
        # 数据加载容错
        try:
            pred_pcd = np.load(join(npy_dir, "predicted_pts3d.npy")).astype(np.float32)
            gt_pcd = np.load(join(npy_dir, "gt_pts3d.npy")).astype(np.float32)
            valid_pcd = np.load(join(npy_dir, "gt_valid_mask.npy")).astype(bool)
        except Exception:
            summary_lines.append(f"{scene_name} | 数据失败\n")
            continue

        loss = []

        for frame_idx in tqdm(range(len(pred_pcd)), desc=f"处理 {scene_name}"):
            frame_loss = calcu_single_frame_loss(
                gt_pcd[frame_idx], pred_pcd[frame_idx], valid_pcd[frame_idx]
            )
            if frame_loss is not None:
                loss.append(frame_loss)
        
        if not loss:
            summary_lines.append(f"{scene_name} | 无有效帧\n")
            continue
        
        avg_loss = np.round(np.mean(loss), 3)
        avg_loss = float(f"{avg_loss:.3f}")
        summary_lines.append(f"{scene_name} | {avg_loss}\n")

    with open(args.summary_path, "w", encoding="utf-8") as f:
        f.writelines(summary_lines)
    print(f"结果已保存至: {args.summary_path}")


if __name__ == "__main__":
    main()