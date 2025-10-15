import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from SLSim import SLSim_batch

pattern_path = "/data/hanning/SLAM3R/data/patterns/alacarte.png"
pattern = cv2.imread(pattern_path, cv2.IMREAD_UNCHANGED) 
pattern = torch.from_numpy(pattern)

K_proj=torch.tensor([[680,0.0,590,0.00],
                 [0.0,580,260,0.00],
                 [0.0 ,0.0,1.0  ,0.00],
                 [0.00,0.00,0.00 ,1.00]],dtype=torch.float32).cuda()

def random_cam2proj_batch(B, device):
    transforms = []
    for _ in range(B):
        R = torch.tensor([[1,0,0],[0,1,0],[0,0,1]])
        t = torch.tensor([0.1, -0.1, 0.1])
        T = torch.eye(4)
        T[:3,:3] = R
        T[:3,3] = t
        transforms.append(T)
    return torch.stack(transforms).to(device)

def process_aria_folder(root_dir,output_root):
    subfolders = sorted(
    [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))],
    key=lambda x: int(x)
      )
    print(f"find {len(subfolders)} scene folders")
    devices = ["cuda:0", "cuda:1", "cuda:2"]
    num_gpus = len(devices)

    for folder in tqdm(subfolders, desc="processing scenes", unit="scene"):
        scene_path = os.path.join(root_dir, folder)
        color_dir = os.path.join(scene_path, 'color')
        depth_dir = os.path.join(scene_path, 'depth')
        intr_dir = os.path.join(scene_path,'intrinsic','intrinsic_color.txt')

        if not os.path.exists(color_dir) or not os.path.exists(depth_dir):
            print(f"[Skipped] {folder} missing color or depth folders")
            continue

        color_images = sorted(os.listdir(color_dir))
        depth_images = sorted(os.listdir(depth_dir))
        mat = np.loadtxt(intr_dir)
        K_cam = torch.tensor(mat, dtype=torch.float32).cuda()
        K_cam = K_cam.unsqueeze(0)
        total_pairs = len(color_images)
        idx = 0
        batch_size = 12
        save_dir = os.path.join(output_root, folder)
        os.makedirs(save_dir, exist_ok=True)
        with tqdm(total=total_pairs, desc=f"processing {folder}", leave=False, unit="pair") as pbar:
            end_idx = idx    
            while idx < total_pairs:
                batch_colors, batch_depths, batch_K_list = [], [], []
                end_idx = idx
                while end_idx < total_pairs:
                    if len(batch_colors) >= batch_size:
                        break
                    color = cv2.imread(os.path.join(color_dir, color_images[end_idx]))
                    depth = cv2.imread(os.path.join(depth_dir, depth_images[end_idx]), cv2.IMREAD_UNCHANGED)
                    batch_colors.append(torch.from_numpy(color))
                    batch_depths.append(torch.from_numpy(depth))   
                    batch_K_list.append(K_cam)
                    end_idx += 1
                B = len(batch_colors)
                gpu_batches = [[],[],[]]

                for i in range(B):
                    gpu_batches[i % num_gpus].append(i)

                overlay_results = [None]*B
                for g_id, indices in enumerate(gpu_batches):
                    if len(indices)==0:
                        continue
                    device = devices[g_id]
                    rgb_batch = torch.stack([batch_colors[i] for i in indices]).to(device)
                    depth_batch = torch.stack([batch_depths[i] for i in indices]).float().to(device)
                    K_batch = torch.cat([batch_K_list[i] for i in indices]).to(device)
                    cam2proj_batch = random_cam2proj_batch(len(indices), device)
                    pattern_dev = pattern.to(device)
                    overlay_batch = SLSim_batch(rgb_batch, depth_batch, pattern_dev, K_batch, K_proj.expand(len(indices),-1,-1).to(device), cam2proj_batch)
                    for j, idx_in_batch in enumerate(indices):
                        overlay_results[idx_in_batch] = overlay_batch[j].detach().cpu().numpy()

                for i, out_np in enumerate(overlay_results):
                    out_path = os.path.join(save_dir, color_images[idx+i])
                    if out_np.dtype != np.uint8:
                        out_np = (out_np*255).astype(np.uint8) if out_np.max() <= 1 else out_np.astype(np.uint8)
                    cv2.imwrite(out_path, out_np)

                pbar.update(B)
                idx = end_idx


if __name__ == "__main__":
    process_aria_folder("/data/yuzheng/autodl4090D_data/projectaria_ase_processed","/nvme/data/hanning/ase_SL2")
