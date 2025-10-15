import PIL
import numpy as np
import torch
import time

from slam3r.datasets.base.easy_dataset import EasyDataset
from slam3r.datasets.utils.transforms import ImgNorm
from slam3r.utils.geometry import depthmap_to_absolute_camera_coordinates, depthmap_to_world_and_camera_coordinates
import slam3r.datasets.utils.cropping as cropping

import os
import cv2
from tqdm import tqdm
from SLSim import SLSim_batch

# 设置模式路径和模式
pattern_path = "/data/hanning/SLAM3R1/data/patterns/alacarte.png"
pattern = cv2.imread(pattern_path, cv2.IMREAD_UNCHANGED)
pattern = torch.from_numpy(pattern)

# 投影矩阵
Kproj = torch.tensor([
    [400, 0.0, 390],
    [0.0, 350, 350],
    [0.0, 0.0, 1.0]
]).unsqueeze(0)  # 添加批次维度

def crop_resize_if_necessary(image, depthmap, intrinsics, resolution, rng=None, info=None):
    """ This function:
        - first downsizes the image with LANCZOS inteprolation,
            which is better than bilinear interpolation in
    """
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image) 

    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W-cx)
    min_margin_y = min(cy, H-cy)
    assert min_margin_x > W/5, f'Bad principal point in view={info}'
    assert min_margin_y > H/5, f'Bad principal point in view={info}'
    # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (l, t, r, b)
    image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    W, H = image.size  # new size
    assert resolution[0] >= resolution[1]
    if H > 1.1*W:
        resolution = resolution[::-1]
    elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
        if rng.integers(2):
            resolution = resolution[::-1]

    target_resolution = np.array(resolution)

    image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    return image, depthmap, intrinsics2

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

def load_intrinsics(npz_path, device='cuda'):
    """加载内参矩阵"""
    data = np.load(npz_path)
    K = data["intrinsics"]
    return K

def process_single_image(root_dir, output_root):
    subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    folder = subfolders[0]
    print(f"Processing folder: {folder}")
    scene_path = os.path.join(root_dir, folder)
    color_dir = os.path.join(scene_path, 'images')
    depth_dir = os.path.join(scene_path, 'depth')
    DSC_path = os.path.join(scene_path, 'scene_dslr_metadata.npz')
    frame_path = os.path.join(scene_path, 'scene_iphone_metadata.npz')

    color_images = sorted(os.listdir(color_dir))
    depth_images = sorted(os.listdir(depth_dir))
    save_dir = os.path.join(output_root, folder)

    K_DSC = load_intrinsics(DSC_path)
    K_frame = load_intrinsics(frame_path)

    color_img_name = color_images[0]
    depth_img_name = depth_images[0]
    
    img_type = "frame" if color_img_name.startswith("frame") else "DSC"
    print(f"处理图片: {color_img_name}, 类型: {img_type}")
    SL_path = "/nvme/data/hanning/vis_crop/sincos_e3c1da58dd/DSC01286.jpg"
    color = cv2.imread(SL_path)
    depth = cv2.imread(os.path.join(depth_dir, depth_img_name), cv2.IMREAD_UNCHANGED)
    
    if color is None or depth is None:
        print(f"[错误] 无法读取图片: {color_img_name} 或 {depth_img_name}")
        return
    
    if img_type == "frame":
        K = K_frame[0]  # 使用第一个内参
    else:
        K = K_DSC[0]    # 使用第一个内参
    print(K.shape)
    rgb_image, depthmap, intrinsics = crop_resize_if_necessary(
                color, depthmap = depth, intrinsics=K,resolution = (224,224), rng=None, info=None)
    folder = "/nvme/data/hanning/vis_crop"
    save_path = os.path.join(folder, f"{os.path.basename(folder)}.jpg")
    cv2.imwrite(save_path, np.array(rgb_image))


if __name__ == "__main__":
    process_single_image(
        "/data/yuzheng/data/scannetpp_v2/scannetpp_processed",
        "/nvme/data/hanning/scannetpp_SL2_single"
    )
