import torch
import numpy as np
import random
import os
import cv2
import open3d as o3d
import PIL.Image
from dust3r.model import load_model
from dust3r.utils.image import imread_cv2
from dust3r.datasets.utils.cropping import crop_image_depthmap, rescale_image_depthmap, camera_matrix_of_crop, bbox_from_intrinsics_in_out
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from dust3r.datasets.utils.transforms import ImgNorm  # 导入图像归一化处理

# 核心配置
CONFIG = {
    "model_weight_path": "/data3/hanning/dust3r/checkpoints/dust3r_SL_224_kinectic_decoder/checkpoint-best.pth",
    "resolution": 224,  # 图像分辨率
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "conf_threshold": 0.3,
    "base_result_dir": "/data3/hanning/dust3r/result/",  # 基础结果目录
}

def init_scene_dir(scene_name):
    scene_dir = os.path.join(CONFIG["base_result_dir"], scene_name)
    npy_dir = os.path.join(scene_dir, "npy")
    ply_dir = os.path.join(scene_dir, "ply")
    
    # 创建目录（如果不存在）
    for dir_path in [scene_dir, npy_dir, ply_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    print(f"场景目录已准备：{os.path.abspath(scene_dir)}")
    return scene_dir, npy_dir, ply_dir

def init_model():
    """初始化原始模型"""
    model = load_model(
        model_path=CONFIG["model_weight_path"],
        device=CONFIG["device"],
        verbose=False
    )
    model.eval()
    print(f"原始模型加载完成，运行设备：{CONFIG['device']}")
    return model

def transpose_to_landscape(view):
    """将视图转换为横屏格式，与数据集处理保持一致"""
    height, width = view['true_shape']

    if width < height:
        # 交换图像维度
        view['img'] = view['img'].swapaxes(1, 2)
        
        # 交换深度图维度
        view['depthmap'] = view['depthmap'].swapaxes(0, 1)
        
        # 交换点云维度
        if 'pts3d' in view:
            view['pts3d'] = view['pts3d'].swapaxes(0, 1)
        
        # 交换有效掩码维度
        if 'valid_mask' in view:
            view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)

        # 调整内参（交换x和y）
        view['camera_intrinsics'] = view['camera_intrinsics'][[1, 0, 2]]

def _crop_resize_if_necessary(image, depthmap, intrinsics, resolution, rng=None):
    """与数据集类中相同的裁剪和缩放处理"""
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)

    # 基于主点裁剪
    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (l, t, r, b)
    image, depthmap, intrinsics = crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    # 处理分辨率（确保横屏）
    W, H = image.size
    assert resolution[0] >= resolution[1]
    if H > 1.1 * W:
        resolution = resolution[::-1]
    elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1] and rng is not None:
        if rng.integers(2):
            resolution = resolution[::-1]

    # 高质量下采样
    target_resolution = np.array(resolution)
    image, depthmap, intrinsics = rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

    # 最终裁剪
    intrinsics2 = camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
    
    return image, depthmap, intrinsics2

def process_view(img_path, depth_path, intrinsics, resolution, rng, idx, view_idx):
    """处理单个视图，完全遵循数据集的处理流程"""
    # 读取图像和深度图
    img = imread_cv2(img_path)
    depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
    
    depthmap = depthmap.astype(np.float32)
    max_val = float(np.nanmax(depthmap)) if depthmap.size > 0 else 0.0
    depthmap = depthmap / 6553.5

    # 非有限值置为0
    depthmap[~np.isfinite(depthmap)] = 0.0

    # 裁剪和缩放
    img_pil, depthmap, intrinsics = _crop_resize_if_necessary(
        img, depthmap, intrinsics, resolution, rng=rng)

    # 构建基础视图字典
    view = {
        'img': img_pil,
        'depthmap': depthmap.astype(np.float32),
        'camera_intrinsics': intrinsics.astype(np.float32),
        # 对于没有位姿的情况，使用单位矩阵
        'camera_pose': np.eye(4, dtype=np.float32),
        'dataset': 'custom',
        'label': os.path.basename(img_path),
        'instance': f'{idx}_{view_idx}',
    }

    # 记录原始形状
    width, height = img_pil.size
    view['true_shape'] = np.int32((height, width))

    # 图像归一化（与数据集保持一致）
    view['img'] = ImgNorm(view['img'])

    # 计算3D点云和有效掩码
    pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(** view)
    view['pts3d'] = pts3d
    view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

    # 转换为横屏格式
    transpose_to_landscape(view)

    return view

def filter_point_cloud(pts3d, view, model_output):
    view["img"] = view['img'][0]
    H, W = view["img"].shape[1:]  # 注意此时img形状是 (C, H, W)
    pts_flat = pts3d.reshape(-1, 3)
    # 颜色处理 - 从标准化值转换回0-1范围
    colors_flat = view["img"].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    colors_flat = (colors_flat * 0.5 + 0.5).clip(0, 1)  # 反归一化
    
    valid_mask = np.isfinite(pts_flat).all(axis=1)
    valid_mask &= (pts_flat[:, 2] > 0)
    
    if "conf" in model_output:
        conf_flat = model_output["conf"].squeeze(0).cpu().numpy().reshape(-1)
        valid_mask &= (conf_flat > CONFIG["conf_threshold"])
        
    return pts_flat[valid_mask], colors_flat[valid_mask]

def save_pts_to_ply(pts, colors, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=False)
    return True

def process_image_pair(model, image1_path, image2_path, depth1_path, depth2_path, 
                       pair_idx, ply_dir, intrinsics, resolution, rng):
    """处理单个图像对并返回评估所需数据（不单独保存npy文件）"""
    try:
        # 处理两个视图
        view1 = process_view(image1_path, depth1_path, intrinsics, resolution, rng, pair_idx, 0)
        view2 = process_view(image2_path, depth2_path, intrinsics, resolution, rng, pair_idx, 1)
        
        # 设备和维度处理
        view1['img'] = view1['img'].unsqueeze(0).to(CONFIG["device"])
        view2['img'] = view2['img'].unsqueeze(0).to(CONFIG["device"])
        view1['true_shape'] = torch.from_numpy(view1['true_shape']).unsqueeze(0).to(CONFIG["device"])
        view2['true_shape'] = torch.from_numpy(view2['true_shape']).unsqueeze(0).to(CONFIG["device"])
        
        # 执行推理
        with torch.no_grad():
            res1, res2 = model(view1, view2)
        
        # 处理预测点云
        pred_pts, pred_colors = filter_point_cloud(
            res1["pts3d"].squeeze(0).cpu().numpy(), 
            view1, 
            res1
        )
        
        # 处理GT点云
        gt_pts = view1['pts3d']  # 由GT深度图计算的点云
        gt_valid_mask = view1['valid_mask']
        gt_pts_flat = gt_pts.reshape(-1, 3)
        gt_valid_flat = gt_valid_mask.reshape(-1)
        
        pair_prefix = f"pair_{pair_idx:03d}"
        
        print(f"已处理图像对 {pair_idx+1}，点云数量: {pred_pts.shape[0]}")
        return {
            'pair_idx': pair_idx,
            'image1_path': image1_path,
            'image2_path': image2_path,
            'pred_pts': pred_pts,
            'gt_pts': gt_pts_flat,
            'gt_valid_mask': gt_valid_flat
        }
        
    except Exception as e:
        print(f"处理图像对 {pair_idx} 时出错: {str(e)}")
        return None

def process_scene(model, scene_name, image_pairs, intrinsics):
    scene_dir, npy_dir, ply_dir = init_scene_dir(scene_name)

    # 初始化随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    resolution = (CONFIG["resolution"], CONFIG["resolution"])
    rng = np.random.default_rng(seed=42)
    
    scene_results = []
    # 用于合并所有单帧数据
    all_pred_pts = []
    all_gt_pts = []
    all_gt_valid_masks = []
    
    for pair_idx in range(len(image_pairs)):
        img1, img2, depth1, depth2 = image_pairs[pair_idx]
        result = process_image_pair(
            model, img1, img2, depth1, depth2, 
            pair_idx, ply_dir, intrinsics, resolution, rng  # 不再传递npy_dir
        )
        
        if result is not None:
            scene_results.append(result)
            # 收集数据用于合并
            all_pred_pts.append(result['pred_pts'])
            all_gt_pts.append(result['gt_pts'])
            all_gt_valid_masks.append(result['gt_valid_mask'])
    
    pred_path = os.path.join(npy_dir, "predicted_pts3d.npy")
    gt_path = os.path.join(npy_dir, "gt_pts3d.npy")
    valid_path = os.path.join(npy_dir, "gt_valid_mask.npy")
    
    np.save(pred_path, all_pred_pts)
    np.save(gt_path, all_gt_pts)
    np.save(valid_path, all_gt_valid_masks)
    
    # 保存场景结果摘要
    scene_summary_path = os.path.join(scene_dir, "scene_results_summary.npy")
    np.save(scene_summary_path, scene_results)
    
    print(f"场景 {scene_name} 处理完成，共{len(scene_results)} 对有效结果")
    print(f"all_pred_pts形状为：{len(all_pred_pts)}")
    
    return {
        'scene_name': scene_name,
        'results': scene_results,
        'summary_path': scene_summary_path,
        'num_pairs_processed': len(scene_results),
    }

def main():
    model = init_model()
    intrinsics = np.array([[600.0, 0, 599.5],
                          [0, 600.0, 399.5],
                          [0, 0, 1]], dtype=np.float32)
    
    # 初始化随机种子（全局只需要设置一次）
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    scenes = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]
    for scene_name in scenes:
        image_pairs = []
        idx = 0
        while True:
            img1_path = f"/data3/hanning/datasets/Replica_kinectsp/{scene_name}/results/frame{idx:06d}.jpg"
            img2_path = f"/data3/hanning/datasets/Replica_kinectsp/{scene_name}/results/frame{idx+1:06d}.jpg"
            depth1_path = f"/data3/hanning/datasets/Replica/{scene_name}/results/depth{idx:06d}.png"
            depth2_path = f"/data3/hanning/datasets/Replica/{scene_name}/results/depth{idx+1:06d}.png"
            
            if all(os.path.exists(p) for p in [img1_path, img2_path, depth1_path, depth2_path]):
                image_pairs.append((img1_path, img2_path, depth1_path, depth2_path))
                idx += 1
            else:
                break

        image_pairs = random.sample(image_pairs, min(100, len(image_pairs)))

        if not image_pairs:
            print(f"场景 {scene_name} 无有效图像对，跳过")
            continue

        scene_result = process_scene(
            model=model,
            scene_name=scene_name,
            image_pairs=image_pairs,
            intrinsics=intrinsics
        )

        print(f"\n===== 场景 {scene_name} 处理完成 =====")

if __name__ == "__main__":
    main()