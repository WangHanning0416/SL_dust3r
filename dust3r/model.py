# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub
from torch import nn
import torch.nn.functional as F
import cv2

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed,PatchEmbed_Mlp

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), ("Outdated huggingface_hub version, "
                                                                     "please reinstall requirements.txt")
DECODER = False
ENCODER = False

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)

class PatternCNN(nn.Module):
    def __init__(self, in_channels, cnn_channels, out_dim, img_size, patch_size):
        super().__init__()
        self.in_channels = in_channels
        self.cnn_channels = cnn_channels
        self.out_dim = out_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.target_HW = img_size[0] // patch_size  # 目标输出空间维度（与图像patch一致）

        layers = []
        prev_channels = in_channels
        current_HW = list(img_size)[0]
        for channels in cnn_channels:
            layers.extend([
                nn.Conv2d(prev_channels, channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ])
            prev_channels = channels
            current_HW = current_HW // 2
        layers.append(nn.Conv2d(prev_channels, out_dim, kernel_size=1, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(out_dim))
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, pattern_tensor):
        """
        前向传播：输入(B, 3, H, W) → 输出(B, N, out_dim)（N=target_HW*target_HW）
        """
        # 1. 卷积下采样：(B, 3, H, W) → (B, out_dim, target_HW, target_HW)
        conv_feat = self.conv_layers(pattern_tensor)
        # 2. 维度转换：适配图像特征格式 (B, C, H, W) → (B, H*W, C)
        B, C, H, W = conv_feat.shape
        patch_feat = conv_feat.view(B, C, H*W).permute(0, 2, 1)  # 最终形状(B, N, C)
        return patch_feat

class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(** croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode,** croco_kwargs)
        self.set_freeze(freeze)
        self.pattern_channels = 3
        self.pattern_cnn_channels = [64, 128, 256, 512]

        self.img_size = croco_kwargs.get('img_size', 224)  # 从croco参数获取图像尺寸（默认224）
        self.patch_size = croco_kwargs.get('patch_size', 16)  # 从croco参数获取patch大小（默认16）
        # self.pattern_cnn = PatternCNN(
        #     in_channels=self.pattern_channels,
        #     cnn_channels=self.pattern_cnn_channels,
        #     out_dim=self.enc_embed_dim,  # 与图像编码器输出通道一致
        #     img_size=self.img_size,
        #     patch_size=self.patch_size
        # )
        # self.pattern_embed = nn.Linear(self.enc_embed_dim, self.dec_embed_dim, bias=True)
        self.pattern_encoder_embed = PatchEmbed_Mlp(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=3,  # pattern是3通道
            embed_dim=self.enc_embed_dim,  # 与图像嵌入维度一致（1024）
            flatten=True  # 最终输出(B, 196, 1024)
        )
        self.pattern_decoder_embed = PatchEmbed_Mlp(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=3,  # pattern是3通道
            embed_dim=self.dec_embed_dim,  # 与解码器嵌入维度一致（768）
            flatten=True  # 最终输出(B, 196, 768)
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            try:
                model = super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError as e:
                raise Exception(f'tried to load {pretrained_model_name_or_path} from huggingface, but failed')
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_size = patch_size
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt,** kw):
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        
        if ENCODER:
            B = x.shape[0]
            pattern_path = "/data3/hanning/dust3r/tools/cropped_image.png"
            pattern = cv2.imread(pattern_path)
            device = image.device
            pattern = torch.tensor(pattern, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, H, W)
            pattern = pattern.unsqueeze(0).to(device)
            pattern = pattern.repeat(B, 1, 1, 1)  # (B, 3, H, W)
            pattern_embed,pos_embed = self.pattern_encoder_embed(pattern)  # (B, N, enc_embed_dim)
            # print("x,Patch特征形状:", x.shape) #(2,196,1024)
            # print("pattern特征形状:", pattern_embed.shape) #(2,196,1024)
            x = x + pattern_embed

        assert self.enc_pos_embed is None

        total_blocks = len(self.enc_blocks)
        half_blocks = int(total_blocks * 0.5)  # 前50%的层数

        # now apply the transformer encoder and normalization
        for i,blk in enumerate(self.enc_blocks):
            x = blk(x, pos)
            if ENCODER and i < half_blocks:
                x = x + pattern_embed  # 注入pattern特征

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2, pattern):
        img1, img2 = view1['img'], view2['img']
        B = img1.shape[0]

        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # 第一步：先编码图像，获取 feat1/feat2（之前顺序反了，导致feat1未定义）
        if is_symmetrized(view1, view2):
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2, pattern):
        final_output = [(f1, f2)] 

        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2) #shape=(B,196,768)
        
        if DECODER:
            B = f1.shape[0]
            pattern = pattern.repeat(B, 1, 1, 1)
            pattern_feat,_ = self.pattern_decoder_embed(pattern)  # (B, N, dec_embed_dim)
            

        final_output.append((f1, f2))
        i = 1
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)      
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            if DECODER:
                if i < 6:
                    f1 += pattern_feat
                    f2 += pattern_feat
                i += 1
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        pattern_path = "/data3/hanning/dust3r/tools/cropped_image.png"
        pattern = cv2.imread(pattern_path)
        if pattern is None:
            raise FileNotFoundError(f"未找到pattern图像：{pattern_path}")
            
        device = view1['img'].device
        pattern = torch.tensor(pattern, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, H, W)
        pattern = pattern.to(device)
        
        B = view1['img'].shape[0]
        pattern = pattern.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 3, H, W)

        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2, pattern)
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, pattern)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
            res2['pts3d_in_other_view'] = res2.pop('pts3d')

        return res1, res2