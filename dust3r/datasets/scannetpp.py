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
from dust3r.utils.image import imread_cv2
from dust3r.utils.modalities import gen_sparse_depth,gen_rays,gen_rel_pose
from dust3r.datasets.base.pattern_synth import PatternSynthMixin

class ScanNetpp(BaseStereoViewDataset):
    def __init__(self, *args, split='train', ROOT, **kwargs):
        self.ROOT = ROOT
        self.metadata_ROOT = "/nvme/data/hanning/datasets/Scannetpp/scannetpp_processed" 
        self.SL_ROOT = "/nvme/data/hanning/datasets/Scannetpp/scannetpp_SL"
        self.split_file = osp.join(self.metadata_ROOT, 'train_test_split.json')
        assert osp.exists(self.split_file), f"non exist: {self.split_file}"
        
        with open(self.split_file, 'r') as f:
            split_data = json.load(f)
        self.train_scenes = split_data['train']
        self.test_scenes = split_data['test']
        
        super().__init__(*args, **kwargs)
        self.split = split
        self.loaded_data = self._load_data()

    def _load_data(self):
        with np.load(osp.join(self.metadata_ROOT, 'all_metadata.npz')) as data:
            self.scenes = data['scenes']
            self.sceneids = data['sceneids']
            self.images = data['images']
            self.intrinsics = data['intrinsics'].astype(np.float32)
            self.trajectories = data['trajectories'].astype(np.float32)
            all_pairs = data['pairs'][:, :2].astype(int)
        print("-----------",self.split,"------------")
        if self.split == 'train':
            selected_scenes = set(self.train_scenes)
        elif self.split == 'test':
            selected_scenes = set(self.test_scenes)
        else:
            raise ValueError(f"unsupported: {self.split}, valuable: 'train', 'test'")

        selected_image_indices = []
        for i, scene_id in enumerate(self.sceneids):
            scene_name = self.scenes[scene_id]
            if scene_name in selected_scenes:
                selected_image_indices.append(i)
        selected_image_set = set(selected_image_indices)

        valid_pairs = []
        for pair in all_pairs:
            if pair[0] in selected_image_set and pair[1] in selected_image_set:
                valid_pairs.append(pair)
        
        self.pairs = np.array(valid_pairs, dtype=int)
        return self.pairs

    def __len__(self):
        return len(self.pairs)

    def _get_views(self, idx, resolution, rng):
        image_idx1, image_idx2 = self.pairs[idx]

        views = []
        for view_idx in [image_idx1, image_idx2]:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]
           
            rgb_image = imread_cv2(osp.join(self.SL_ROOT,self.scenes[scene_id], basename + '.jpg'))
            depthmap = imread_cv2(osp.join(scene_dir, 'depth', basename + '.png'), cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)

            view_dict = dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='ScanNet++',
                label=self.scenes[scene_id] + '_' + basename,
                instance=f'{str(idx)}_{str(view_idx)}',
            )
            valid_mask = (depthmap > 0)
            view_dict['known_depth'] = gen_sparse_depth(
                view_dict,
                valid_mask,
                n_pts_min=64,
                n_pts_max=0,
            )
            view_dict['known_rays'] = gen_rays(view_dict)
            views.append(view_dict)
        views[0]['known_pose'] = gen_rel_pose(views)
        views[1]['known_pose'] = gen_rel_pose([views[1],views[0]])
        return views

if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    train_dataset = ScanNetpp(
        split='train', 
        ROOT="/data/yuzheng/data/scannetpp_v2/scannetpp_processed", 
        resolution=224, 
        aug_crop=16
    )

    for idx in np.random.permutation(min(1, len(train_dataset))):
        views = train_dataset[idx]
        assert len(views) == 2
        print(f"train {idx}: {view_name(views[0])}, {view_name(views[1])}")
        
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        # print("known_depth:",views[0]['known_depth'])
        # print("camera:",views[0]['camera_intrinsics'])
        # print("known_rays",views[0]['known_rays'])
        print("camera_pose1:",views[0]['camera_pose'])
        print("camera_pose2:",views[1]['camera_pose'])
        print("known_pose1",views[0]['known_pose'])
        print("known_pose2",views[1]['known_pose'])
        # print(views[0]['known_pose'] @ views[1]['known_pose'])
           