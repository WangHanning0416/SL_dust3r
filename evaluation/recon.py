import torch
import numpy as np
import random
from tqdm import tqdm
import os
import cv2
import open3d as o3d
import PIL.Image
from dust3r.model import load_model
from dust3r.utils.image import imread_cv2
from dust3r.datasets.utils.cropping import crop_image_depthmap, rescale_image_depthmap, camera_matrix_of_crop, bbox_from_intrinsics_in_out
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from dust3r.datasets.utils.transforms import ImgNorm  # 导入图像归一化处理
from evaluation.eval import evaluate_scene_data

# 核心配置
CONFIG = {
    "model_weight_path": "/data3/hanning/dust3r1/checkpoints/dust3r_SL_224/checkpoint-best.pth",
    "resolution": 224,  
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "conf_threshold": 0.3,
    "base_result_dir": "/data3/hanning/dust3r/result_rgb/",  # 基础结果目录
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

def process_view(img_path, depth_path, intrinsics, resolution, rng, idx, view_idx):
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
        'camera_pose': np.eye(4, dtype=np.float32),
        'dataset': 'custom',
        'label': os.path.basename(img_path),
        'instance': f'{idx}_{view_idx}',
    }

    width, height = img_pil.size
    view['true_shape'] = np.int32((height, width))

    view['img'] = ImgNorm(view['img'])

    # 计算3D点云和有效掩码
    pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(** view)
    view['pts3d'] = pts3d
    view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

    # 转换为横屏格式
    transpose_to_landscape(view)

    return view

def save_pts_to_ply(pts, colors, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=False)
    return True

def process_image_pair(model, image1_path, image2_path, depth1_path, depth2_path, 
                       pair_idx, ply_dir, intrinsics, resolution, rng):
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

def stack_batch_and_move_to_device(view_list, device):
    batch = {}
    batch['img'] = torch.stack([v['img'] for v in view_list]).to(device)
    batch['true_shape'] = torch.stack([torch.from_numpy(v['true_shape']) for v in view_list]).to(device)
    # batch['camera_intrinsics'] = torch.from_numpy(
    #     np.stack([v['camera_intrinsics'] for v in view_list])
    # ).float().to(device)

    # batch['camera_pose'] = torch.from_numpy(
    #     np.stack([v['camera_pose'] for v in view_list])
    # ).float().to(device)
    return batch

def process_batch(model, image1_batch, image2_batch, depth1_batch, depth2_batch, 
                  batch_idx, intrinsics, resolution, rng):
    view1_batch_list = []
    view2_batch_list = []
    batch_size = len(image1_batch)
    
    # 1. 预处理：逐个样本调用 process_view
    for i in range(batch_size):
        # 假设 process_view 返回包含 NumPy/CPU Tensor 的 dict
        view1 = process_view(image1_batch[i], depth1_batch[i], intrinsics, resolution, rng, batch_idx * batch_size + i, 0)
        view2 = process_view(image2_batch[i], depth2_batch[i], intrinsics, resolution, rng, batch_idx * batch_size + i, 1)
        view1_batch_list.append(view1)
        view2_batch_list.append(view2)

    # 2. 堆叠和推理：Batch Inference
    view1 = stack_batch_and_move_to_device(view1_batch_list, CONFIG["device"])
    view2 = stack_batch_and_move_to_device(view2_batch_list, CONFIG["device"])
    # 假设 'instance' 是 Batch 维度的元数据，需要保持列表形式
    view1['instance'] = [v['instance'] for v in view1_batch_list]
    view2['instance'] = [v['instance'] for v in view2_batch_list]
    
    with torch.no_grad():
        res1_batch, res2_batch = model(view1, view2)

    pred_pts3d_np = res1_batch["pts3d"].cpu().numpy() # (B, H, W, 3)

    # 颜色反归一化 (B, C, H, W) -> (B, H, W, C)
    img_tensor_cpu = view1['img'].cpu() 
    colors_np = img_tensor_cpu.permute(0, 2, 3, 1).numpy()
    colors_np = (colors_np * 0.5 + 0.5).clip(0, 1) 
    
    # 置信度 (B, 1, H, W) -> (B, H, W)
    conf_np = res1_batch["conf"].cpu().numpy()

    gt_pts3d_batch = np.stack([v['pts3d'] for v in view1_batch_list]) # (B, H, W, 3) NumPy
    gt_valid_mask_batch = np.stack([v['valid_mask'] for v in view1_batch_list]) # (B, H, W) NumPy

    batch_pred_pts_flat = []
    batch_pred_colors_flat = []
    batch_pred_conf_flat = []
    batch_gt_pts_flat = []
    batch_gt_valid_flat = []
    
    for i in range(batch_size):
        # 预测结果展平 (H*W, 3)
        batch_pred_pts_flat.append(pred_pts3d_np[i].reshape(-1, 3))
        batch_pred_colors_flat.append(colors_np[i].reshape(-1, 3))
        
        if conf_np is not None:
            batch_pred_conf_flat.append(conf_np[i].reshape(-1)) # (H*W,)

        # GT 结果展平
        batch_gt_pts_flat.append(gt_pts3d_batch[i].reshape(-1, 3))
        batch_gt_valid_flat.append(gt_valid_mask_batch[i].reshape(-1)) # (H*W,)
        
    H, W = pred_pts3d_np.shape[1:3]
    #print(f"Batch {batch_idx + 1} 已处理 {batch_size} 个图像对，每个点云形状为 ({H*W}, 3)")

    return {
        'batch_idx': batch_idx,
        'image1_paths': image1_batch,
        'image2_paths': image2_batch,
        'pred_pts_list': batch_pred_pts_flat,         # List of (H*W, 3)
        'pred_colors_list': batch_pred_colors_flat,   # List of (H*W, 3)
        'pred_conf_list': batch_pred_conf_flat,       # List of (H*W,) (如果有)
        'gt_pts_list': batch_gt_pts_flat,             # List of (H*W, 3)
        'gt_valid_mask_list': batch_gt_valid_flat     # List of (H*W,)
    }

def process_scene(model, scene_name, image_pairs, intrinsics):
    scene_dir, npy_dir, ply_dir = init_scene_dir(scene_name) 

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    resolution = (CONFIG["resolution"], CONFIG["resolution"])
    rng = np.random.default_rng(seed=42)
    device = CONFIG["device"]
    batch_size = CONFIG.get("batch_size", 16) 
    
    scene_results = []
    all_pred_pts_list = []      
    all_pred_colors_list = []
    all_pred_conf_list = []   
    all_gt_pts_list = []        
    all_gt_valid_masks_list = []
    
    num_pairs = len(image_pairs)
    num_batches = (num_pairs + batch_size - 1) // batch_size
    
    # 核心修改：使用 tqdm 包装 Batch 循环
    batch_indices = range(0, num_pairs, batch_size)
    
    for batch_idx_start in tqdm(
        batch_indices, 
        total=num_batches,  #num_batches
        desc=f"Processing Scene: {scene_name}",
        unit="batch"
    ):
        # 1. 确定当前 Batch 的范围和数据
        start_idx = batch_idx_start
        end_idx = min(batch_idx_start + batch_size, num_pairs)
        current_pairs = image_pairs[start_idx:end_idx]
        
        # 解包 Batch 数据
        img1_batch = [p[0] for p in current_pairs]
        img2_batch = [p[1] for p in current_pairs]
        depth1_batch = [p[2] for p in current_pairs]
        depth2_batch = [p[3] for p in current_pairs]
        
        # 2. 调用 Batch 处理函数
        batch_results = process_batch(
            model=model, 
            image1_batch=img1_batch, 
            image2_batch=img2_batch, 
            depth1_batch=depth1_batch, 
            depth2_batch=depth2_batch,
            batch_idx=batch_idx_start // batch_size, # 传递 Batch 索引
            intrinsics=intrinsics, 
            resolution=resolution, 
            rng=rng,
        )

        if batch_results is not None:
            all_pred_pts_list.extend(batch_results['pred_pts_list'])
            all_pred_colors_list.extend(batch_results['pred_colors_list'])
            all_gt_pts_list.extend(batch_results['gt_pts_list'])
            all_gt_valid_masks_list.extend(batch_results['gt_valid_mask_list'])
            
            if 'pred_conf_list' in batch_results and batch_results['pred_conf_list']:
                all_pred_conf_list.extend(batch_results['pred_conf_list'])

    scene_name,loss = evaluate_scene_data(
        scene_name=scene_name,
        pred_pts_list=all_pred_pts_list,
        gt_pts_list=all_gt_pts_list,
        valid_mask_list=all_gt_valid_masks_list
    )

    #将结果写到指定txt文件的末尾去
    summary_path = "/data3/hanning/dust3r/evaluation/result.txt"
    with open(summary_path, 'a') as f:
        f.write(f"{scene_name}: {loss}\n")

    print(f"\n场景 {scene_name} 处理完成，共 {len(all_pred_pts_list)} 个图像对结果已保存。")

def main():
    # 假设 init_model 已定义
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
            # 确保这些路径是正确的，并且文件存在
            img1_path = f"/data3/hanning/datasets/Replica_kinectsp/{scene_name}/results/frame{idx:06d}.jpg"
            img2_path = f"/data3/hanning/datasets/Replica_kinectsp/{scene_name}/results/frame{idx+1:06d}.jpg"
            depth1_path = f"/data3/hanning/datasets/Replica/{scene_name}/results/depth{idx:06d}.png"
            depth2_path = f"/data3/hanning/datasets/Replica/{scene_name}/results/depth{idx+1:06d}.png"
            
            if all(os.path.exists(p) for p in [img1_path, img2_path, depth1_path, depth2_path]):
                image_pairs.append((img1_path, img2_path, depth1_path, depth2_path))
                idx += 1
            else:
                break

        if not image_pairs:
            print(f"场景 {scene_name} 无有效图像对，跳过")
            continue

        process_scene(
            model=model,
            scene_name=scene_name,
            image_pairs=image_pairs,
            intrinsics=intrinsics
        )
        print(f"\n===== 场景 {scene_name} 处理完成 =====")

if __name__ == "__main__":
    main()