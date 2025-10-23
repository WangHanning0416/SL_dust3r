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


def evaluate_scene_data(scene_name: str, 
                        pred_pts_list: list[np.ndarray], 
                        gt_pts_list: list[np.ndarray], 
                        valid_mask_list: list[np.ndarray]) -> tuple[str, float or str]:
    """
    评估整个场景中所有图像对的平均损失。
    
    Args:
        scene_name: 场景名称.
        pred_pts_list: 场景中所有图像对的预测点云列表 (List[(H*W, 3)]).
        gt_pts_list: 场景中所有图像对的 GT 点云列表 (List[(H*W, 3)]).
        valid_mask_list: 场景中所有图像对的 GT 有效掩码列表 (List[(H*W,)]).

    Returns:
        (场景名称, 平均损失或错误信息).
    """
    all_losses = []
    
    for i in tqdm(range(len(pred_pts_list)), desc=f"Evaluating {scene_name}"):
        # 注意：这里直接使用 list 中的元素，它们已经是展平后的 (H*W, 3) 数组
        frame_loss = calcu_single_frame_loss(
            gt_pts_list[i], pred_pts_list[i], valid_mask_list[i]
        )
        if frame_loss is not None:
            all_losses.append(frame_loss)

    if not all_losses:
        return scene_name, "无有效帧"

    avg_loss = np.round(np.mean(all_losses), 3)
    return scene_name, float(f"{avg_loss:.3f}")