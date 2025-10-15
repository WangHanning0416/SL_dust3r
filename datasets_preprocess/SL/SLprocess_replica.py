import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

# ---------------------- 1. 定义SLSim相关依赖函数 ----------------------
def depth_to_point_cloud_batch(depth, intrinsics):
    B,H,W = depth.shape
    device = depth.device

    y, x = torch.meshgrid(torch.arange(H, device=device),
                          torch.arange(W, device=device), indexing='ij')
    x = x.unsqueeze(0).expand(B,-1,-1)
    y = y.unsqueeze(0).expand(B,-1,-1)

    Z = depth
    fx = intrinsics[:,0,0].unsqueeze(-1).unsqueeze(-1)
    fy = intrinsics[:,1,1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[:,0,2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[:,1,2].unsqueeze(-1).unsqueeze(-1)

    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    return torch.stack([X,Y,Z], dim=-1)

def transform_point_cloud_batch(point_cloud, cam2proj):
    B,H,W,_ = point_cloud.shape
    pc_flat = point_cloud.view(B,-1,3)
    ones = torch.ones(B, pc_flat.shape[1],1, device=point_cloud.device)
    pc_hom = torch.cat([pc_flat, ones], dim=-1)
    proj_points_hom = torch.bmm(pc_hom, cam2proj.transpose(1,2))
    proj_points = proj_points_hom[...,:3] / proj_points_hom[...,3:4]
    return proj_points.view(B,H,W,3)

def project_to_pattern_plane_batch(proj_points, K_proj):
    B,H,W,_ = proj_points.shape
    X = proj_points[...,0]
    Y = proj_points[...,1]
    Z = proj_points[...,2].clamp(min=1e-6)

    fx = K_proj[:,0,0].unsqueeze(-1).unsqueeze(-1)
    fy = K_proj[:,1,1].unsqueeze(-1).unsqueeze(-1)
    cx = K_proj[:,0,2].unsqueeze(-1).unsqueeze(-1)
    cy = K_proj[:,1,2].unsqueeze(-1).unsqueeze(-1)

    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return u,v

def sample_pattern_batch(u, v, pattern):
    B,H,W = u.shape
    device = u.device
    pat_h, pat_w, _ = pattern.shape
    sampled = torch.zeros(B,H,W,3, dtype=torch.uint8, device=device)

    for b in range(B):
        u_b = torch.remainder(torch.round(u[b]).long(), pat_w)
        v_b = torch.remainder(torch.round(v[b]).long(), pat_h)
        sampled[b] = pattern[v_b, u_b, :].to(device)
    return sampled

def apply_depth_attenuation_batch(image, depth, Z0=0.8, gamma=1.0, min_val=0.2):
    attenuation = (Z0 / depth.clamp(0.1,5.0))**gamma
    attenuation = attenuation.clamp(min_val,1.0).unsqueeze(-1)
    out = (image.float() * attenuation).clamp(0,255).byte()
    return out

def apply_depth_blur_batch(image, depth, focus=None, strength=3):
    B,H,W,_ = image.shape
    out = torch.zeros_like(image)
    for b in range(B):
        img_np = image[b].cpu().numpy()
        depth_np = depth[b].cpu().numpy()
        norm_depth = (depth_np - depth_np.min())/(depth_np.max()-depth_np.min()+1e-6)
        blur_img = cv2.GaussianBlur(img_np, (5,5), strength)
        f = depth_np.min() if focus is None else focus
        alpha = np.clip((norm_depth - f)*5,0,1)[...,None]
        out_np = (img_np*(1-alpha)+blur_img*alpha).astype(np.uint8)
        out[b] = torch.from_numpy(out_np).to(image.device)
    return out

def SLSim_batch(rgb, depth, pattern, K_cam, K_proj, cam2proj):
    B,H,W,_ = rgb.shape
    device = rgb.device

    point_cloud = depth_to_point_cloud_batch(depth, K_cam)
    proj_points = transform_point_cloud_batch(point_cloud, cam2proj)
    u, v = project_to_pattern_plane_batch(proj_points, K_proj)
    projected = sample_pattern_batch(u, v, pattern)
    projected = apply_depth_attenuation_batch(projected, proj_points[...,2])
    projected = apply_depth_blur_batch(projected, proj_points[...,2])
    
    rgb_f = rgb.float()
    proj_f = projected.float()
    alpha = 0.5
    gray = proj_f.mean(dim=-1, keepdim=True)/255.0
    w1 = gray + (1-gray)*(1-alpha)
    w2 = (1-gray)*alpha
    overlay = (rgb_f*w1 + proj_f*w2).clamp(0,255).byte()
    return overlay

# ---------------------- 2. 处理Replica数据集的主函数 ----------------------
def process_replica_folder(root_dir, output_root, pattern, K_cam, K_proj, cam2proj, device='cuda'):
    subfolders = ["office0", "office1" ,"office2" ,"office3" ,"office4" ,"room0" ,"room1" ,"room2"]
    print(f"Find {len(subfolders)} scene folders")

    # 图案预处理：转为tensor并移动到设备（修复类型转换错误）
    if pattern.shape[-1] == 4:  # 处理RGBA图案，取RGB通道
        pattern = pattern[..., :3]
    # 错误修正：使用type()方法转换为uint8，而非.uint8()
    pattern_tensor = torch.from_numpy(pattern).to(device).type(torch.uint8)

    # 内参/外参批量扩展
    K_cam_batch = K_cam.unsqueeze(0).to(device)
    K_proj_batch = K_proj.unsqueeze(0).to(device)
    cam2proj_batch = cam2proj.unsqueeze(0).to(device)

    for folder in tqdm(subfolders, desc="processing scenes", unit="scene"):
        scene_path = os.path.join(root_dir, folder, 'results')
        color_images = sorted([f for f in os.listdir(scene_path) if f.endswith('.jpg')])
        depth_images = sorted([f for f in os.listdir(scene_path) if f.endswith('.png')])

        if len(color_images) != len(depth_images):
            print(f"[Warning] {folder} has unequal number of color and depth images")
            continue

        save_dir = os.path.join(output_root, folder, 'results')
        os.makedirs(save_dir, exist_ok=True)

        for cimg, dimg in tqdm(zip(color_images, depth_images),
                               total=len(color_images),
                               desc=f"processing {folder}",
                               leave=False,
                               unit="pair"):
            color_path = os.path.join(scene_path, cimg)
            depth_path = os.path.join(scene_path, dimg)

            color = cv2.imread(color_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if color is None or depth is None:
                print(f"[Warning] Failed to read: {color_path} or {depth_path}")
                continue

            # 保存原始深度图
            cv2.imwrite(os.path.join(save_dir, dimg), depth)

            # 深度图预处理
            depth = (depth / 6553.5).astype(np.float32)

            color_tensor = torch.from_numpy(color).unsqueeze(0).to(device).type(torch.uint8)
            depth_tensor = torch.from_numpy(depth).unsqueeze(0).to(device).float()

            # 调用SLSim_batch
            res = SLSim_batch(
                rgb=color_tensor,
                depth=depth_tensor,
                pattern=pattern_tensor,
                K_cam=K_cam_batch,
                K_proj=K_proj_batch,
                cam2proj=cam2proj_batch
            )

            # 结果保存
            res_np = res.squeeze(0).cpu().numpy()
            out_path = os.path.join(save_dir, cimg)
            cv2.imwrite(out_path, res_np)

# ---------------------- 3. 主函数入口 ----------------------
if __name__ == "__main__":
    input_root = "/data/hanning/SLAM3R/data/Replica"
    output_root = "/nvme/data/hanning/datasets/replica_SL"
    pattern_path = "/data/hanning/SLAM3R/data/patterns/alacarte.png"

    # 加载结构光图案
    pattern = cv2.imread(pattern_path, cv2.IMREAD_UNCHANGED)
    if pattern is None:
        raise FileNotFoundError(f"Failed to load pattern: {pattern_path}")

    # 相机/投影仪参数
    K_cam = torch.tensor([
        [600.0,   0.0, 599.5],
        [  0.0, 600.0, 339.5],
        [  0.0,   0.0,   1.0]
    ], dtype=torch.float32)

    K_proj = torch.tensor([
        [900.0,   0.0, 600.0],
        [  0.0, 700.0, 350.0],
        [  0.0,   0.0,   1.0]
    ], dtype=torch.float32)

    cam2proj = torch.eye(4, dtype=torch.float32)
    cam2proj[:3, 3] = torch.tensor([0.1, 0.0, 0.0])

    # 运行处理函数
    process_replica_folder(
        root_dir=input_root,
        output_root=output_root,
        pattern=pattern,
        K_cam=K_cam,
        K_proj=K_proj,
        cam2proj=cam2proj,
        device='cuda'
    )
