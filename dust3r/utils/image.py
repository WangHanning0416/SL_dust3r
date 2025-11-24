# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os
import torch
import torch.nn.functional as F
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa
import megfile
import imageio.v3 as iio
from typing import Optional, Union
import kornia

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _dbg_save_img_tensor(path, img:torch.Tensor):
    '''img: (CHW) or (HW)'''
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

    if img.ndim == 4:
        assert img.shape[0] == 1
        img = img.squeeze(0)
    
    if img.ndim == 3:
        img = img.permute(1,2,0)
    elif img.ndim == 2:
        img = img.unsqueeze(-1).expand(img.shape + (3,))
    else:
        raise NotImplementedError
    img = (img.clip(0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
    iio.imwrite(path, img)


def _dbg_show_depth_tensor(path, dep:torch.Tensor):
    import matplotlib.pyplot as plt
    if isinstance(dep, np.ndarray):
        dep = torch.from_numpy(dep)
    
    if dep.ndim > 2:
        dep = dep.squeeze()
    assert dep.ndim == 2, f"invalid depth shape: {dep.shape}"
    
    dep = dep.detach().cpu().numpy()
    vmin = dep[dep > 0].min()
    vmax = dep.max()
    plt.imshow(dep, cmap='Spectral_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(path)
    plt.close()


def img_to_arr(img):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        if options & cv2.IMREAD_ANYDEPTH == 0:
            options = cv2.IMREAD_ANYDEPTH  # options中没有cv2.IMREAD_ANYDEPTH, 没被考虑到，覆盖掉
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imread_iio(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with imageio.v3.
    """
    ext = os.path.splitext(path)[1].lower()
    f = megfile.smart_open(path, 'rb')

    if ext == '.exr':
        options = cv2.IMREAD_ANYDEPTH
    img = iio.imread(f, extension=ext, plugin='opencv', flags=options)
    f.close()

    # if path.endswith(('.exr', 'EXR')):
    #     img = iio.imread(path, extension='EXR-FI')
    # else:
    #     img = iio.imread(path)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        return img[..., :3]  # exclude alpha channel
    else:
        return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(folder_or_list, size, square_ok=False, verbose=True, patch_size=16):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw = ((2 * cx) // patch_size) * patch_size / 2
            halfh = ((2 * cy) // patch_size) * patch_size / 2
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs


# def estimate_reflectance_multiscale_retinex(img: np.ndarray, sigma_list=[15, 80, 250]) -> np.ndarray:
#     """ Estimate reflectance using multi-scale retinex algorithm
#         Input:
#             img: torch.Tensor of shape (H,W,3), in range [0,255]
#         Output:
#             reflectance: torch.Tensor of shape (B,3,H,W), in range [0,1]
#     """
#     assert img.max() >= 1.0 and img.min() >= 0., "图片像素强度范围应该是 [0,255]."
#     img_np = img.astype(np.uint8)  # H,W,3

#     img_b = img_np
#     log_img = np.log1p(img_b.astype(np.float32))

#     msr = np.zeros_like(img_b, dtype=np.float32)
#     for sigma in sigma_list:
#         blur = cv2.GaussianBlur(img_b, (0, 0), sigma)
#         log_blur = np.log1p(blur.astype(np.float32))
#         msr += log_img - log_blur
#     msr /= len(sigma_list)

#     # Normalize to [0,255]
#     msr_min = msr.min()
#     msr_max = msr.max()
#     msr_norm = 255 * (msr - msr_min) / (msr_max - msr_min + 1e-8)

#     reflectance_np = msr_norm
#     return reflectance_np


def estimate_reflectance_numpy(img: np.ndarray, sigma=75, use_guided=False):
    """
    使用Y通道（亮度）估计光照，返回反射率R与光照S。  
    img是np.ndarray, 认为它是刚读入的HWC numpy 数组，范围是[0,255]  
    numpy版本，所有函数使用opencv实现  

    return: Reflectance R (HWC, float32), Illumination S_Y (HW, float32)
    """
    assert img.max() >= 1.0 and img.min() >= 0., "图片像素强度范围应该是 [0,255]."
    I = img.astype(np.float32) / 255.
    I = np.clip(I, 1e-4, 1.0)

    # 转换到 YCrCb 颜色空间
    YCrCb = cv2.cvtColor(I, cv2.COLOR_RGB2YCrCb)
    Y = YCrCb[..., 0]  # hw

    logY = np.log(Y + 1e-6)

    # 估计平滑光照分量 logS_Y
    if use_guided:
        if not hasattr(cv2, "ximgproc"):
            raise RuntimeError("需要安装 opencv-contrib-python 才能使用 guided filter")
        logS_Y = cv2.ximgproc.guidedFilter(guide=Y, src=logY, radius=int(sigma), eps=1e-3)
    else:
        logS_Y = cv2.GaussianBlur(logY, (0, 0), sigma)

    # 光照分量 S_Y
    S_Y = np.exp(logS_Y)  # hw

    # 反射率: R = I / S_Y
    R = I / (S_Y[..., None])  # hwc

    # norm R
    norm = R.max()

    return R / norm, S_Y * norm


def estimate_reflectance_tensor(img: torch.Tensor, sigma=75, use_guided=False, S_scalefactor:float = 1):
    """
    使用Y通道（亮度）估计光照，返回反射率R与光照S。  
    img是torch.Tensor, 认为它是 BCHW 数组，范围是[0,1]  
    torch.Tensor版本.  
    S_scalefactor, 在算S分量的时候是个高斯滤波，可以提前降采样来加速..

    return: Reflectance R (BCHW, float32), Illumination S_Y (BHW, float32)
    """
    assert not use_guided, "torch版本暂不支持 guided filter."
    b, c, h, w = img.shape

    # 转换到 YCrCb 颜色空间
    YCrCb = kornia.color.rgb_to_ycbcr(img)
    Y = YCrCb[:, 0:1]  # B,1,H,W

    logY = torch.log(Y + 1e-6)

    # 估计平滑光照分量 logS_Y
    if S_scalefactor < 1:
        logY = F.interpolate(logY, scale_factor=S_scalefactor, mode='bilinear', align_corners=False)
        sigma = sigma * S_scalefactor
        
    if use_guided:
        raise NotImplementedError
    else:
        ksize = int(2 * round(3*sigma) + 1)
        min_dim = min(logY.shape[-2], logY.shape[-1])
        max_ksize = 2 * int(min_dim * 0.9)  # 0.9: 避免ksize==imgsize
        ksize = min(max_ksize, ksize)
        if ksize % 2 == 0:
            ksize -= 1
        logS_Y = kornia.filters.gaussian_blur2d(logY, (ksize, ksize), (sigma, sigma))

    if S_scalefactor < 1:
        logS_Y = F.interpolate(logS_Y, size=img.shape[-2:], mode='bilinear', align_corners=False)

    # 光照分量 S_Y
    S_Y = torch.exp(logS_Y)  # b1hw

    # 反射率: R = I / S_Y
    R = img / S_Y  # bchw

    # norm R.
    norm = R.max()

    return R / norm, (S_Y * norm).squeeze(1)


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='input image path')
    parser.add_argument('-o', '--output', type=str, default=None, help='output reflectance path')
    parser.add_argument('-s', '--sigma', type=float, default=25.0, help='sigma for Gaussian blur')
    parser.add_argument('-g', '--use_guided', action='store_true', help='use guided filter instead of Gaussian blur')
    parser.add_argument("-t", "--tensor", action='store_true', help='use torch tensor version')
    args = parser.parse_args()

    img = imread_cv2(args.input, options=cv2.IMREAD_COLOR)

    func = estimate_reflectance_numpy
    if args.tensor:
        img = torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
        func = estimate_reflectance_tensor

    s = time.time()
    R, S = func(img, sigma=args.sigma, use_guided=args.use_guided)
    print(f'Estimated reflectance in {time.time()-s} seconds.')

    if args.tensor:
        R = R.squeeze(0).permute(1, 2, 0).numpy()  # H,W,3
        S = S.squeeze(0).numpy()  # H,W

    print(R.max(axis=(0,1)), R.min(axis=(0,1)), S.max(), S.min())

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + '_reflectance' + os.path.splitext(args.input)[1]

    # 在同一个figue中可视化反射率图R和光照图S
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Estimated Reflectance R')
    # plt.imshow(np.clip(R, 0, 1))
    plt.imshow(R / R.max())
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Estimated Illumination S (Y channel)')
    plt.imshow(S, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(args.output)
    print(f'Saved reflectance and illumination visualization to {args.output}')