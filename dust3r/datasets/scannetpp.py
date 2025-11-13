# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed scannet++
# dataset at https://github.com/scannetpp/scannetpp - non-commercial research and educational purposes
# https://kaldir.vc.in.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf
# See datasets_preprocess/preprocess_scannetpp.py
# --------------------------------------------------------
import os.path as osp
import cv2
import numpy as np
import json

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.datasets.base.pattern_synth import PatternSynthMixin
from dust3r.utils.image import imread_cv2, imread_iio


class ScanNetpp(BaseStereoViewDataset, PatternSynthMixin):
    def __init__(self, *args, split='train', ROOT, pattern_config=None, pattern_split=None, **kwargs):
        '''subtrain, subval: subval的场景比例是0.05, 大约42个.'''
        self.ROOT = ROOT
        assert split in ['train', 'subtrain', 'subval', 'test'] 

        self.pattern_config = pattern_config
        self.pattern_split = pattern_split
        
        super().__init__(*args, **kwargs)
        assert self.num_views <= 2, "only support num_views in [1,2] now."
        self.split = split
        self.subval_ratio = 0.05
        
        if split == 'test':
            self.split = 'subval' # 在 _load_data 中将其映射到 subval 场景分割逻辑
        
        self._load_data()
        
        if self.pattern_config is not None:
            # 自动推断 pattern_split 的合理默认值
            if self.pattern_split is None:
                default_pattern_split = 'train' if split in ['train', 'subtrain'] else 'test'
                self.pattern_split = default_pattern_split
                print(f"[ScanNetpp] 自动推断 pattern_split 为: '{self.pattern_split}' (基于数据集 split='{split}')")
            
            print(f"[ScanNetpp] 激活结构光合成. Config: {self.pattern_config}, Split: {self.pattern_split}")
            # 调用继承自 PatternSynthMixin 的方法，立即激活功能
            self.initialize_pattern_config(self.pattern_config, self.pattern_split)

    def _load_data(self):
        with np.load("/nvme/data/hanning/datasets/Scannetpp/scannetpp_processed/all_metadata.npz") as data:
            self.scenes = data['scenes']
            self.sceneids = data['sceneids']
            self.images = data['images']
            self.intrinsics = data['intrinsics'].astype(np.float32)
            self.trajectories = data['trajectories'].astype(np.float32)
            self.pairs = data['pairs'][:, :2].astype(int)
        
        if self.split == 'train':
            return   # 所有都拿来训练，直接返回.
        
        # subval or subtrain.
        subval_num = int(len(self.scenes) * self.subval_ratio)
        if self.split == 'subtrain':
            selected_scenes = self.scenes[:-subval_num]
        else: # split == 'subval' (也包括我们映射的 'test' split)
            selected_scenes = self.scenes[-subval_num:]

        selected_image_indices = []
        for i, scene_id in enumerate(self.sceneids):
            scene_name = self.scenes[scene_id]
            if scene_name in selected_scenes:
                selected_image_indices.append(i)
        selected_image_set = set(selected_image_indices)

        valid_pairs = []
        for pair in self.pairs.tolist():
            if pair[0] in selected_image_set and pair[1] in selected_image_set:
                valid_pairs.append(pair)
        
        self.pairs = np.array(valid_pairs, dtype=int)
        return self.pairs

    def __len__(self):
        return len(self.pairs)

    def _get_views(self, idx, resolution, rng):
        image_idx1, image_idx2 = self.pairs[idx]

        views = []
        for view_idx in [image_idx1, image_idx2] if self.num_views == 2 else [image_idx1]:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]

            # 读取原始 RGB 图像 (会被 PatternSynthMixin 替换)
            rgb_image = imread_iio(osp.join(self.ROOT,self.scenes[scene_id], 'images', basename + '.jpg'))  
            depthmap = imread_iio(osp.join(scene_dir, 'depth', basename + '.png'), cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='ScanNet++',
                label=self.scenes[scene_id] + '|' + basename,
                instance=f'{str(idx)}_{str(view_idx)}',
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    pattern_config = "/data/hanning/dust3r/configs/pattern_config.yaml"

    train_dataset = ScanNetpp(
        split='train', 
        ROOT="/data/yuzheng/data/scannetpp_v2/scannetpp_processed", 
        resolution=224, 
        aug_crop=16
    )

    train_dataset.initialize_pattern_config(pattern_config, split='train')

    for idx in np.random.permutation(min(5, len(train_dataset))):
        views = train_dataset[idx]
        assert len(views) == 2
        print(f"train {idx}: {view_name(views[0])}, {view_name(views[1])}")
        
        valid = views[0]['pattern_valid_mask'] 
        true_count = np.sum(valid)
        print(f"pattern valid count: {true_count} / {valid.size}" )
        