#裁剪输入图片为224*224
import cv2
import numpy as np
import torch
import PIL
import os.path as osp
from dust3r.datasets.base.easy_dataset import EasyDataset
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
import dust3r.datasets.utils.cropping as cropping

path = "/data3/hanning/datasets/Replica_kinectsp/office0/results/frame000001.jpg"


def _crop_resize_if_necessary(image, depthmap, intrinsics, resolution, rng=None, info=None):
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)

    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (l, t, r, b)
    image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    W, H = image.size  # new size
    assert resolution[0] >= resolution[1]
    if H > 1.1 * W:
        resolution = resolution[::-1]
    elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
        if rng.integers(2):
            resolution = resolution[::-1]

    target_resolution = np.array(resolution)
    image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    return image, depthmap, intrinsics2

rgb_image = cv2.imread(path)
depthmap = cv2.imread(path)
H, W = rgb_image.shape[:2]  # OpenCV 格式是 (H, W, 3)
intrinsics = np.array([
    [max(W, H), 0, W/2],    # 焦距 fx，主点 cx=W/2
    [0, max(W, H), H/2],    # 焦距 fy，主点 cy=H/2
    [0, 0, 1]
], dtype=np.float64)
resolution = (224,224)

rgb_image, depthmap, intrinsics = _crop_resize_if_necessary(
    rgb_image, depthmap, intrinsics, resolution, rng=None, info=1)
            
if isinstance(rgb_image, torch.Tensor):
    rgb_image = rgb_image.cpu().detach().numpy()

if isinstance(rgb_image, PIL.Image.Image):
    rgb_image = np.array(rgb_image)

rgb_image = np.asarray(rgb_image)

if rgb_image.ndim == 3 and rgb_image.shape[0] in (1, 3):
    rgb_image = np.transpose(rgb_image, (1, 2, 0))

if rgb_image.ndim == 2:
    rgb_image = np.stack([rgb_image] * 3, axis=-1)
if rgb_image.dtype != np.uint8:
    # if values in [0,1], scale up
    maxv = float(np.nanmax(rgb_image))
    if maxv <= 1.0:
        rgb_image = (rgb_image * 255.0).round()
    # clip to [0,255] and convert
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)


if rgb_image.shape[2] == 3:
    if rgb_image[..., 0].mean() < rgb_image[..., 2].mean():
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

# 6. 保存图像
cv2.imwrite("./tools/pattern000001.png", rgb_image)