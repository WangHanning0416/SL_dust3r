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
from dust3r.datasets.utils.transforms import ImgNorm # 导入图像归一化处理

from dust3r.utils.modalities import gen_sparse_depth,gen_rays,gen_rel_pose
from evaluation.eval import evaluate_scene_data

# 核心配置
CONFIG = {
    "model_weight_path": "/nvme/data/hanning/checkpoints/dust3r_kinectsp_224_inject_pose/checkpoint-best.pth",
    "resolution": 224,  
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "conf_threshold": 0.3,
    "base_result_dir": "/data/hanning/dust3r/result_rgb/",
    "batch_size": 16,
}

def init_scene_dir(scene_name):
    scene_dir = os.path.join(CONFIG["base_result_dir"], scene_name)
    npy_dir = os.path.join(scene_dir, "npy")
    ply_dir = os.path.join(scene_dir, "ply")
    
    for dir_path in [scene_dir, npy_dir, ply_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    print(f"场景目录已准备：{os.path.abspath(scene_dir)}")
    return scene_dir, npy_dir, ply_dir

def init_model():
    """初始化原始模型"""
    if not os.path.exists(CONFIG["model_weight_path"]):
        print(f"警告：模型权重文件未找到：{CONFIG['model_weight_path']}。请确保路径正确。")
        return None
        
    model = load_model(
        model_path=CONFIG["model_weight_path"],
        device=CONFIG["device"],
        verbose=False
    )
    model.eval()
    print(f"原始模型加载完成，运行设备：{CONFIG['device']}")
    return model

def transpose_to_landscape(view):
    """
    将视图转换为横屏格式，并确保 known_depth 和 known_rays 被正确转置。
    """
    height, width = view['true_shape']

    if width < height:
        view['img'] = view['img'].swapaxes(1, 2)
        view['depthmap'] = view['depthmap'].swapaxes(0, 1)
        
        # === 修复/新增：known_depth 和 known_rays 转置 ===
        # known_depth 形状 (2, H, W) -> (2, W, H)，交换 H, W (索引 1, 2)
        if 'known_depth' in view and view['known_depth'] is not None:
            view['known_depth'] = view['known_depth'].swapaxes(1, 2)
            
        # known_rays 形状 (3, H, W) -> (3, W, H)，交换 H, W (索引 1, 2)
        if 'known_rays' in view and view['known_rays'] is not None:
            view['known_rays'] = view['known_rays'].swapaxes(1, 2)
            
        if 'pts3d' in view:
            view['pts3d'] = view['pts3d'].swapaxes(0, 1)
        if 'valid_mask' in view:
            view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)
            
        # 交换相机内参的 fx 和 fy
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
    # 使用 NumPy 数组进行分辨率比较
    resolution_np = np.array(resolution)
    
    if H > 1.1 * W:
        resolution_np = resolution_np[::-1]
    elif 0.9 < H / W < 1.1 and resolution_np[0] != resolution_np[1] and rng is not None:
        if rng.integers(2):
            resolution_np = resolution_np[::-1]

    target_resolution = resolution_np
    image, depthmap, intrinsics = rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

    intrinsics2 = camera_matrix_of_crop(intrinsics, image.size, target_resolution, offset_factor=0.5)
    crop_bbox = bbox_from_intrinsics_in_out(intrinsics, intrinsics2, target_resolution)
    image, depthmap, intrinsics2 = crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
    
    return image, depthmap, intrinsics2

def process_view(img_path, depth_path, intrinsics, camera_pose ,resolution, rng, idx, view_idx):

    img = imread_cv2(img_path)
    depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
    
    if img is None or depthmap is None:
        print(f"Warning: Image or depth map loading failed for {os.path.basename(img_path)}. Skipping sample.")
        return None
        
    depthmap = depthmap.astype(np.float32)
    depthmap = depthmap / 6553.5 # 假设的归一化
    depthmap[~np.isfinite(depthmap)] = 0.0

    # --- 2. 裁剪和缩放 ---
    img_pil, depthmap, intrinsics = _crop_resize_if_necessary(
        img, depthmap, intrinsics, resolution, rng=rng)
    
    if img_pil is None or depthmap is None:
         print(f"Warning: Crop/Resize returned None for {os.path.basename(img_path)}. Skipping sample.")
         return None

    # --- 3. 初始化 View 字典 ---
    view = {
        'img': img_pil,
        'depthmap': depthmap.astype(np.float32),
        'camera_intrinsics': intrinsics.astype(np.float32),
        'camera_pose': camera_pose.astype(np.float32),
        'dataset': 'custom',
        'label': os.path.basename(img_path),
        'instance': f'{idx}_{view_idx}',
    }

    width, height = img_pil.size
    view['true_shape'] = np.int32((height, width))
    valid_mask = (view['depthmap'] > 0)
    
    # --- 4. 新增：生成 known_rays (光线方向) ---
    # gen_rays 返回 (3, H, W)
    view['known_rays'] = gen_rays(view)
    
    # --- 5. 生成 known_depth (稀疏深度) ---
    # gen_sparse_depth 返回 (2, H, W)
    view['known_depth'] = gen_sparse_depth(
        view,
        valid_mask,
        n_pts_min=64,
        n_pts_max=0,
    )

    # ImgNorm 将 PIL.Image 转换为 PyTorch Tensor 并进行归一化
    view['img'] = ImgNorm(view['img'])
    pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
        depthmap=view['depthmap'],
        camera_intrinsics=view['camera_intrinsics'],
        camera_pose=np.eye(4, dtype=np.float32)
    )
    view['pts3d'] = pts3d
    view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

    # --- 7. 转置处理 ---
    transpose_to_landscape(view)

    return view

def process_image_pair(model, image1_path, image2_path, depth1_path, depth2_path, 
                         pair_idx, ply_dir, intrinsics, resolution, rng):
    print(f"警告：process_image_pair 函数未被 main 调用，并且缺少 filter_point_cloud 函数定义。")
    return None

def stack_batch_and_move_to_device(view_list, device):
    """
    堆叠批次数据，确保 known_depth 和 known_rays 被正确堆叠并移动到设备。
    """
    valid_views = [v for v in view_list if v is not None]
    if not valid_views:
        return None

    batch = {}
    batch['img'] = torch.stack([v['img'] for v in valid_views]).to(device)
    batch['true_shape'] = torch.stack([torch.from_numpy(v['true_shape']) for v in valid_views]).to(device)
    
    # === 新增：堆叠 known_rays ===
    known_rays_list = [v.get('known_rays') for v in valid_views]
    if all(kr is not None for kr in known_rays_list):
        # (3, H, W) NumPy 数组 -> Tensor -> 堆叠为 (B, 3, H, W)
        known_rays_tensors = [torch.from_numpy(kr) for kr in known_rays_list]
        batch['known_rays'] = torch.stack(known_rays_tensors).to(device)
    
    # === 修复：堆叠 known_depth ===
    known_depth_list = [v.get('known_depth') for v in valid_views]
    if all(kd is not None for kd in known_depth_list):
        # (2, H, W) NumPy 数组 -> Tensor -> 堆叠为 (B, 2, H, W)
        known_depth_tensors = [torch.from_numpy(kd) for kd in known_depth_list]
        batch['known_depth'] = torch.stack(known_depth_tensors).to(device)
    
    known_pose_list = [v.get('known_pose') for v in valid_views]
    if all(kp is not None for kp in known_pose_list):
        # (B,4,4)
        known_pose_tensors = [torch.from_numpy(kp) for kp in known_pose_list]
        batch['known_pose'] = torch.stack(known_pose_tensors).to(device)

        
    return batch

def process_batch(model, image1_batch, image2_batch, depth1_batch, depth2_batch,
                  camera_pose1_batch, camera_pose2_batch, 
                  batch_idx, intrinsics, resolution, rng):
    
    batch_size = len(image1_batch)
    
    # 1. 预处理：逐个样本调用 process_view
    view1_batch_list = []
    view2_batch_list = []
    for i in range(batch_size):
        # process_view 会返回 None 如果加载失败
        view1 = process_view(image1_batch[i], depth1_batch[i], intrinsics, camera_pose1_batch[i], resolution, rng, batch_idx * batch_size + i, 0)
        view2 = process_view(image2_batch[i], depth2_batch[i], intrinsics, camera_pose2_batch[i], resolution, rng, batch_idx * batch_size + i, 1)
        view1["known_pose"] = gen_rel_pose([view1,view2])
        view2["known_pose"] = gen_rel_pose([view2,view1])
        #print("view1_pose_shape:",view1["known_pose"].shape)

        # 仅将成功处理的 view 加入列表
        if view1 is not None and view2 is not None:
             view1_batch_list.append(view1)
             view2_batch_list.append(view2)
        else:
             print(f"Skipping pair {batch_idx * batch_size + i} due to data loading/processing failure.")

    # 检查是否有有效样本
    if not view1_batch_list:
        return None
        
    # 2. 堆叠和推理：Batch Inference
    view1 = stack_batch_and_move_to_device(view1_batch_list, CONFIG["device"])
    view2 = stack_batch_and_move_to_device(view2_batch_list, CONFIG["device"])

    # 如果堆叠后的批次为空 (即所有有效样本都被过滤)，则返回 None
    if view1 is None or view2 is None:
        return None
        
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
    
    # 信心度处理
    conf_tensor = res1_batch["conf"].cpu()
    
    if conf_tensor.dim() == 4 and conf_tensor.shape[1] == 1:
        conf_np = conf_tensor.squeeze(1).numpy() # 结果为 (B, H, W)
    elif conf_tensor.dim() == 3:
        conf_np = conf_tensor.numpy()
    else:
        conf_np = conf_tensor.numpy()
        if conf_np.ndim == 4:
            conf_np = conf_np[:, 0, :, :]

    # GT 数据准备
    gt_pts3d_batch = np.stack([v['pts3d'] for v in view1_batch_list]) # (B, H, W, 3) NumPy
    gt_valid_mask_batch = np.stack([v['valid_mask'] for v in view1_batch_list]) # (B, H, W) NumPy
    gt_depth_map_batch = np.stack([v['depthmap'] for v in view1_batch_list]) # (B, H, W) NumPy
    
    batch_pred_pts_flat = []
    batch_pred_colors_flat = []
    batch_pred_conf_flat = []
    batch_gt_pts_flat = []
    batch_gt_valid_flat = []
    batch_gt_depth_flat = []
    
    valid_batch_size = len(view1_batch_list)
    
    for i in range(valid_batch_size):
        # 预测结果展平
        batch_pred_pts_flat.append(pred_pts3d_np[i].reshape(-1, 3))
        batch_pred_colors_flat.append(colors_np[i].reshape(-1, 3))
        
        if conf_np is not None and conf_np.ndim > 1:
            batch_pred_conf_flat.append(conf_np[i].reshape(-1)) 

        # GT 结果展平
        batch_gt_pts_flat.append(gt_pts3d_batch[i].reshape(-1, 3))
        batch_gt_valid_flat.append(gt_valid_mask_batch[i].reshape(-1))
        batch_gt_depth_flat.append(gt_depth_map_batch[i].reshape(-1))
        
    return {
        'batch_idx': batch_idx,
        # 使用处理后的路径列表，防止长度不匹配
        'image1_paths': [p['label'] for p in view1_batch_list],
        'image2_paths': [p['label'] for p in view2_batch_list],
        'pred_pts_list': batch_pred_pts_flat,
        'pred_colors_list': batch_pred_colors_flat, 
        'pred_conf_list': batch_pred_conf_flat,
        'gt_pts_list': batch_gt_pts_flat, 
        'gt_valid_mask_list': batch_gt_valid_flat, 
        'gt_depth_list': batch_gt_depth_flat 
    }
    
def process_scene(model, scene_name, image_pairs, intrinsics):
    scene_dir, npy_dir, ply_dir = init_scene_dir(scene_name) 

    rng = np.random.default_rng(seed=42)
    
    resolution = (CONFIG["resolution"], CONFIG["resolution"])
    batch_size = CONFIG.get("batch_size", 16) 
    
    all_pred_pts_list = []
    all_pred_colors_list = []
    all_pred_conf_list = []
    all_gt_pts_list = [] 
    all_gt_valid_masks_list = []
    all_gt_depth_list = []
    
    num_pairs = len(image_pairs)
    num_batches = (num_pairs + batch_size - 1) // batch_size

    batch_indices = range(0, num_pairs, batch_size)
    
    for batch_idx_start in tqdm(
        batch_indices, 
        total=num_batches, 
        desc=f"Processing Scene: {scene_name}",
        unit="batch"
    ):
        start_idx = batch_idx_start
        end_idx = min(batch_idx_start + batch_size, num_pairs)
        current_pairs = image_pairs[start_idx:end_idx]
        
        img1_batch = [p[0] for p in current_pairs]
        img2_batch = [p[1] for p in current_pairs]
        depth1_batch = [p[2] for p in current_pairs]
        depth2_batch = [p[3] for p in current_pairs]
        camera_pose1_batch = [p[4] for p in current_pairs]
        camera_pose2_batch = [p[5] for p in current_pairs]
        
        batch_results = process_batch(
            model=model, 
            image1_batch=img1_batch, 
            image2_batch=img2_batch, 
            depth1_batch=depth1_batch, 
            depth2_batch=depth2_batch,
            camera_pose1_batch = camera_pose1_batch,
            camera_pose2_batch = camera_pose2_batch,
            batch_idx=batch_idx_start // batch_size,
            intrinsics=intrinsics, 
            resolution=resolution, 
            rng=rng,
        )

        if batch_results is not None:
            all_pred_pts_list.extend(batch_results['pred_pts_list'])
            all_pred_colors_list.extend(batch_results['pred_colors_list'])
            all_gt_pts_list.extend(batch_results['gt_pts_list'])
            all_gt_valid_masks_list.extend(batch_results['gt_valid_mask_list'])
            all_gt_depth_list.extend(batch_results['gt_depth_list'])
            
            if 'pred_conf_list' in batch_results and batch_results['pred_conf_list']:
                all_pred_conf_list.extend(batch_results['pred_conf_list'])

    if all_gt_depth_list:
        print(f"\n场景 {scene_name} 的 {len(all_gt_depth_list)} 个真实深度图已已经过评估")

    if not all_pred_pts_list:
        print(f"\n场景 {scene_name} 没有有效图像对被处理，跳过评估和结果保存。")
        return

    scene_name,loss = evaluate_scene_data(
        scene_name=scene_name,
        pred_pts_list=all_pred_pts_list,
        gt_pts_list=all_gt_pts_list,
        valid_mask_list=all_gt_valid_masks_list
    )

    summary_path = "/data/hanning/SL_dust3r/evaluation/result.csv"
    try:
        if not os.path.exists(os.path.dirname(summary_path)):
             os.makedirs(os.path.dirname(summary_path), exist_ok=True)
             
        with open(summary_path, 'a') as f:
            f.write(f",{loss}")
    except Exception as e:
        print(f"写入评估总结文件出错: {e}")


    print(f"\n场景 {scene_name} 处理完成，共 {len(all_pred_pts_list)} 个图像对结果已保存。")

def main():
    model = init_model()
    if model is None:
        print("模型初始化失败，退出。")
        return
        
    intrinsics = np.array([[600.0, 0, 599.5],
                           [0, 600.0, 399.5],
                           [0, 0, 1]], dtype=np.float32)
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    
    scenes = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]
    for scene_name in scenes:
        traj = []
        traj_base_path = "/data/hanning/SLAM3R/data/Replica"
        traj_path = os.path.join(traj_base_path,scene_name,"traj.txt")
        traj = np.loadtxt(traj_path).reshape(-1,4,4)
        print("traj shape:",traj.shape)

        image_pairs = []
        idx = 0
        while True:
            base_dir = f"/nvme/data/hanning/datasets/Replica_kinectsp/{scene_name}/results"
            img1_path = os.path.join(base_dir, f"frame{idx:06d}.jpg")
            img2_path = os.path.join(base_dir, f"frame{idx+10:06d}.jpg")
            depth1_path = os.path.join(base_dir, f"depth{idx:06d}.png")
            depth2_path = os.path.join(base_dir, f"depth{idx+10:06d}.png")
            
            if all(os.path.exists(p) for p in [img1_path, img2_path, depth1_path, depth2_path]):
                camera_pose1 = traj[idx]
                camera_pose2 = traj[idx+10] 
                image_pairs.append((img1_path, img2_path, depth1_path, depth2_path, camera_pose1, camera_pose2))
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