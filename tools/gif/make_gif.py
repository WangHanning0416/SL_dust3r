import torch
import numpy as np
import random
import os
import cv2
# import open3d as o3d # <-- 移除 Open3D 导入
import PIL.Image
from dust3r.model import load_model
from dust3r.utils.image import imread_cv2
from dust3r.datasets.utils.cropping import crop_image_depthmap, rescale_image_depthmap, camera_matrix_of_crop, bbox_from_intrinsics_in_out
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.modalities import gen_sparse_depth,gen_rays,gen_rel_pose


CONFIG = {
    "model_weight_path": "/nvme/data/hanning/checkpoints/dust3r_kinectsp_224_inject_pose/checkpoint-best.pth",
    "resolution": 224,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "conf_threshold": 0.3,
    "base_result_dir": "/data/hanning/dust3r/",  
}

def init_scene_dir(scene_name):
    """初始化场景目录，但我们只关心 npy 目录"""
    scene_dir = os.path.join(CONFIG["base_result_dir"], scene_name)
    npy_dir = os.path.join(scene_dir, "npy")

    for dir_path in [scene_dir, npy_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    print(f"场景目录已准备：{os.path.abspath(scene_dir)}")
    return scene_dir, npy_dir

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
        view['img'] = view['img'].swapaxes(1, 2)

        view['depthmap'] = view['depthmap'].swapaxes(0, 1)

        if 'pts3d' in view:
            view['pts3d'] = view['pts3d'].swapaxes(0, 1)
    
        if 'valid_mask' in view:
            view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)

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

    W, H = image.size
    assert resolution[0] >= resolution[1]
    if H > 1.1 * W:
        resolution = resolution[::-1]
    elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1] and rng is not None:
        if rng.integers(2):
            resolution = resolution[::-1]

    target_resolution = np.array(resolution)
    image, depthmap, intrinsics = rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

    intrinsics2 = camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
    
    return image, depthmap, intrinsics2

def process_view(img_path, depth_path, intrinsics, camera_pose, resolution, rng, idx, view_idx):
    """处理单个视图，完全遵循数据集的处理流程"""
    # 读取图像和深度图
    img = imread_cv2(img_path)
    depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
    
    depthmap = depthmap.astype(np.float32)
    max_val = float(np.nanmax(depthmap)) if depthmap.size > 0 else 0.0
    depthmap = depthmap / 6553.5

    depthmap[~np.isfinite(depthmap)] = 0.0

    img_pil, depthmap, intrinsics = _crop_resize_if_necessary(
        img, depthmap, intrinsics, resolution, rng=rng)

    view = {
        'img': img_pil,
        'depthmap': depthmap.astype(np.float32),
        'camera_intrinsics': intrinsics.astype(np.float32),
        'camera_pose': camera_pose,
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
    """过滤点云并提取对应颜色"""
    # 注意：view['img'] 在 process_image_pair 中已经被 unsqueeze(0) 和 to(device)
    if view['img'].ndim == 4:
        view["img"] = view['img'][0]
        
    pts_flat = pts3d.reshape(-1, 3)

    # 颜色处理 - 从标准化值转换回0-1范围
    # view['img'] 现在是 (C, H, W)
    colors_flat = view["img"].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    colors_flat = (colors_flat * 0.5 + 0.5).clip(0, 1)  # 反归一化 (RGB, 0-1)

    valid_mask = np.isfinite(pts_flat).all(axis=1)
    valid_mask &= (pts_flat[:, 2] > 0)
    
    if "conf" in model_output:
        conf_flat = model_output["conf"].squeeze(0).cpu().numpy().reshape(-1)
        valid_mask &= (conf_flat > CONFIG["conf_threshold"])

    return pts_flat[valid_mask], colors_flat[valid_mask]


def process_image_pair(model, image1_path, image2_path, depth1_path, depth2_path,
                        camera_pose1,camera_pose2,
                         pair_idx, intrinsics, resolution, rng):
    """处理单个视图对，返回预测点云和颜色"""
    view1 = process_view(image1_path, depth1_path, intrinsics,camera_pose1, resolution, rng, pair_idx, 0)
    view2 = process_view(image2_path, depth2_path, intrinsics,camera_pose2, resolution, rng, pair_idx, 1)
    view1['known_pose'] = gen_rel_pose([view1,view2])
    view2['known_pose'] = gen_rel_pose([view2,view1])

    # 设备和维度处理 (只保留模型需要的字段)
    view1_for_model = {k: v for k, v in view1.items() if k not in ['pts3d', 'valid_mask', 'depthmap']}
    view2_for_model = {k: v for k, v in view2.items() if k not in ['pts3d', 'valid_mask', 'depthmap']}

    view1_for_model['img'] = view1_for_model['img'].unsqueeze(0).to(CONFIG["device"])
    view2_for_model['img'] = view2_for_model['img'].unsqueeze(0).to(CONFIG["device"])
    view1_for_model['true_shape'] = torch.from_numpy(view1_for_model['true_shape']).unsqueeze(0).to(CONFIG["device"])
    view2_for_model['true_shape'] = torch.from_numpy(view2_for_model['true_shape']).unsqueeze(0).to(CONFIG["device"])

    # 执行推理
    with torch.no_grad():
        res1, res2 = model(view1_for_model, view2_for_model)
        
    pred_pts, pred_colors = filter_point_cloud(
        res1["pts3d"].squeeze(0).cpu().numpy(),
        view1_for_model, # 包含已加载图像的视图
        res1
    )

    print(f"已处理图像对 {pair_idx+1}，预测点云数量: {pred_pts.shape[0]}")
    return pred_pts, pred_colors


def main():
    model = init_model()
    intrinsics = np.array([[600.0, 0, 599.5],
                           [0, 600.0, 399.5],
                           [0, 0, 1]], dtype=np.float32)

    # 初始化随机种子
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed(42)

    scene_name = "office3"
    resolution = (CONFIG["resolution"], CONFIG["resolution"])
    rng = np.random.default_rng(seed=42)
    
    traj = []
    traj_base_path = "/data/hanning/SLAM3R/data/Replica"
    traj_path = os.path.join(traj_base_path,scene_name,"traj.txt")
    traj = np.loadtxt(traj_path).reshape(-1,4,4)
    print("traj shape:",traj.shape)

    idx = 0
    img1_path = f"/nvme/data/hanning/datasets/Replica_kinectsp/{scene_name}/results/frame{idx:06d}.jpg"
    img2_path = f"/nvme/data/hanning/datasets/Replica_kinectsp/{scene_name}/results/frame{idx+10:06d}.jpg"
    depth1_path = f"/nvme/data/hanning/datasets/Replica_kinectsp/{scene_name}/results/depth{idx:06d}.png"
    depth2_path = f"/nvme/data/hanning/datasets/Replica_kinectsp/{scene_name}/results/depth{idx+10:06d}.png"
    camera_pose1 = traj[idx]
    camera_pose2 = traj[idx+10] 

    if not all(os.path.exists(p) for p in [img1_path, img2_path, depth1_path, depth2_path]):
        print("错误: 场景 room0 的第一个图像对文件缺失。")
        return

    scene_dir, npy_dir = init_scene_dir(scene_name)

    print(f"\n===== 开始处理场景 {scene_name} 的第一个图像对 (索引 0) =====")
    pred_pts, pred_colors = process_image_pair(
        model, img1_path, img2_path, depth1_path, depth2_path,camera_pose1,camera_pose2,
        0, intrinsics, resolution, rng
    )

    # 4. 保存点云和颜色到 NumPy 文件
    if pred_pts is not None and pred_pts.shape[0] > 0:
        pts_path = "/data/hanning/SL_dust3r/result/pts/predicted_pts3d_pair000.npy"
        colors_path = "/data/hanning/SL_dust3r/result/pts/predicted_colors_pair000.npy"
        
        np.save(pts_path, pred_pts)
        np.save(colors_path, pred_colors)
        
        print("\n==============================================")
        print("✅ 点云数据保存成功，请下载以下文件：")
        print(f"  - 坐标 (N, 3): {os.path.abspath(pts_path)}")
        print(f"  - 颜色 (N, 3): {os.path.abspath(colors_path)}")
        print("==============================================")
    else:
        print("未生成有效的预测点云，未保存文件。")

if __name__ == "__main__":
    main()