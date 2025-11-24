# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# PatchEmbed implementation for DUST3R & POW3R compatibility
# --------------------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F

import dust3r.utils.path_to_croco  # noqa: F401
from models.blocks import PatchEmbed  # noqa
from croco.models.blocks import Mlp


# --- [修改] 兼容 Pow3r 的 get_patch_embed ---
def get_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3):
    # 扩展允许的类名列表，包含 Dust3r 原版和 Pow3r 新增的类型
    valid_classes = [
        'PatchEmbedDust3R', 
        'ManyAR_PatchEmbed', 
        'PatchEmbed_Mlp', 
        'PatchEmbedDust3R_Mlp', # Pow3r 别名
        'ManyAR_PatchEmbed_Mlp' # Pow3r 新增
    ]
    
    # 稍微放宽检查，或者你可以直接注释掉这行 assert
    if patch_embed_cls not in valid_classes:
        print(f"Warning: {patch_embed_cls} is not in the standard list: {valid_classes}")

    # 动态实例化：传入 in_chans
    patch_embed = eval(patch_embed_cls)(img_size, patch_size, in_chans, enc_embed_dim)
    
    return patch_embed


# --- 辅助类 (保持不变) ---
class Permute(torch.nn.Module):
    dims: tuple[int, ...]
    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = tuple(dims)

    def __repr__(self):
        return f"Permute{self.dims}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.permute(*self.dims)


class PixelUnshuffle (nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        if input.numel() == 0:
            # this is not in the original torch implementation
            C,H,W = input.shape[-3:]
            assert H and W and H % self.downscale_factor == W%self.downscale_factor == 0
            return input.view(*input.shape[:-3], C*self.downscale_factor**2, H//self.downscale_factor, W//self.downscale_factor)
        else:
            return F.pixel_unshuffle(input, self.downscale_factor)


# --- Dust3r 原版类 ---
class PatchEmbedDust3R(PatchEmbed):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        # 显式接受 in_chans 并传递给父类
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)

    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos


class ManyAR_PatchEmbed (PatchEmbed):
    """ Handle images with non-square aspect ratio.
        All images in the same batch have the same aspect ratio.
        true_shape = [(height, width) ...] indicates the actual shape of each image.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        self.embed_dim = embed_dim
        # 显式传递 in_chans
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)

    def forward(self, img, true_shape):
        B, C, H, W = img.shape
        assert W >= H, f'img should be in landscape mode, but got {W=} {H=}'
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        assert true_shape.shape == (B, 2), f"true_shape has the wrong shape={true_shape.shape}"

        W //= self.patch_size[0]
        H //= self.patch_size[1]
        n_tokens = H * W

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape
        x = img.new_zeros((B, n_tokens, self.embed_dim))
        pos = img.new_zeros((B, n_tokens, 2), dtype=torch.int64)

        # linear projection, transposed if necessary
        x[is_landscape] = self.proj(img[is_landscape]).permute(0, 2, 3, 1).flatten(1, 2).float()
        x[is_portrait] = self.proj(img[is_portrait].swapaxes(-1, -2)).permute(0, 2, 3, 1).flatten(1, 2).float()

        pos[is_landscape] = self.position_getter(1, H, W, pos.device)
        pos[is_portrait] = self.position_getter(1, W, H, pos.device)

        x = self.norm(x)
        return x, pos


class PatchEmbed_Mlp (PatchEmbedDust3R):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)

        # 使用 PixelUnshuffle + MLP 替代 Conv2d
        self.proj = nn.Sequential(
            PixelUnshuffle(patch_size), 
            Permute((0,2,3,1)),
            Mlp(in_chans * patch_size**2, 4*embed_dim, embed_dim),
            Permute((0,3,1,2)),
            )

class ManyAR_PatchEmbed_Mlp (ManyAR_PatchEmbed):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)
    
        # 使用 PixelUnshuffle + MLP 替代 Conv2d
        self.proj = nn.Sequential(
            PixelUnshuffle(patch_size), 
            Permute((0,2,3,1)),
            Mlp(in_chans * patch_size**2, 4*embed_dim, embed_dim),
            Permute((0,3,1,2)),
            )


PatchEmbedDust3R_Mlp = PatchEmbed_Mlp