import os
import cv2
from tqdm import tqdm
from SLSim import SLSim
import numpy as np
import torch

pattern_path = "/data/hanning/SLAM3R/data/patterns/alacarte.png"
pattern = cv2.imread(pattern_path, cv2.IMREAD_UNCHANGED) 

def load_intrinsics(npz_path, device='cuda'):
    try:
        data = np.load(npz_path)
        K = data["camera_intrinsics"] 
        K_tensor = torch.from_numpy(K).float().to(device)  
        return K_tensor 
    except Exception as e:
        print(f"[Error] Failed to load intrinsics from {npz_path}: {e}")
        return None

K_proj = torch.tensor([[80, 0.0, 590],
                      [0.0, 60, 350],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

def random_cam2proj():
    roll = (torch.rand(1, device='cuda') - 0.5) *  np.pi * (1/18)
    pitch = (torch.rand(1, device='cuda') - 0.5) *  np.pi * (1/18)
    yaw = (torch.rand(1, device='cuda') - 0.5) *  np.pi * (1/18)
    
    # 旋转矩阵（绕X轴、Y轴、Z轴的旋转）
    R_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(roll), -torch.sin(roll)],
        [0, torch.sin(roll), torch.cos(roll)]
    ], device='cuda')
    
    R_y = torch.tensor([
        [torch.cos(pitch), 0, torch.sin(pitch)],
        [0, 1, 0],
        [-torch.sin(pitch), 0, torch.cos(pitch)]
    ], device='cuda')
    
    R_z = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0],
        [torch.sin(yaw), torch.cos(yaw), 0],
        [0, 0, 1]
    ], device='cuda')
    
    R = torch.mm(torch.mm(R_z, R_y), R_x)
    
    # 平移向量
    t = (torch.rand(3, device='cuda') - 0.5) * 0.3

    transform = torch.eye(4,device='cuda')
    transform[:3, :3] = R
    transform[:3, 3] = t
    return transform


def process_co3d_folder(root_dir, output_root):
    level1_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    print(f"Found {len(level1_folders)} top-level categories")

    for category in tqdm(level1_folders, desc="Processing categories"):
        category_path = os.path.join(root_dir, category)
        scene_folders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]

        for scene in tqdm(scene_folders, desc=f"Processing {category}", leave=False):
            cam2proj = random_cam2proj()
            scene_path = os.path.join(category_path, scene)
            image_dir = os.path.join(scene_path, "images")
            depth_dir = os.path.join(scene_path, "depths")

            if not os.path.exists(image_dir) or not os.path.exists(depth_dir):
                print(f"[Skipped] {category}/{scene} missing image or depth folders")
                continue

            image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
            save_dir = os.path.join(output_root, category, scene)
            os.makedirs(save_dir, exist_ok=True)

            for img_file in tqdm(image_files, desc=f"{category}/{scene}", leave=False):
                img_name = os.path.splitext(img_file)[0]
                color_path = os.path.join(image_dir, img_file)
                depth_path = os.path.join(depth_dir, img_name + ".jpg.geometric.png")
                intrinsics_path = os.path.join(image_dir, img_name + ".npz")
                data = np.load(intrinsics_path)
                color = cv2.imread(color_path)
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                depth = (depth / 65535.0) * data["maximum_depth"]
                depth = depth.astype(np.float32)/1000.0
                K = load_intrinsics(intrinsics_path)
                if color is None or depth is None or K is None:
                    print(f"[Warning] Failed to read image/depth/intrinsics for {img_name}")
                    continue
                res = SLSim(color, depth, pattern, K, K_proj)
                if isinstance(res, torch.Tensor):
                    res_np = res.detach().cpu().numpy()
                    if res_np.ndim == 3 and res_np.shape[0] in [1, 3]:
                        res_np = res_np.transpose(1, 2, 0)
                    out_path = os.path.join(save_dir, img_file)
                    cv2.imwrite(out_path, res_np)
                else:
                    print("res不是tensor，无法保存")
                
if __name__ == "__main__":
    process_co3d_folder("/data/yuzheng/autodl4090D_data/co3d_processed","/data/hanning/SLAM3R/data/co3d_SL")
