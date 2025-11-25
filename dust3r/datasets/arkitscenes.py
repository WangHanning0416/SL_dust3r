# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed arkitscenes
# dataset at https://github.com/apple/ARKitScenes - Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License https://github.com/apple/ARKitScenes/tree/main?tab=readme-ov-file#license
# See datasets_preprocess/preprocess_arkitscenes.py
# --------------------------------------------------------
import os.path as osp
import cv2
import numpy as np
import megfile

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.datasets.base.pattern_synth import PatternSynthMixin
from dust3r.utils.image import imread_cv2, imread_iio
from dust3r.utils.modalities import gen_sparse_depth,gen_rays,gen_rel_pose


class ARKitScenes(BaseStereoViewDataset, PatternSynthMixin):
    def __init__(self, *args, split, ROOT="/nvme/data/jiaheng/dust3r/arkitscenes_preprocessed/", **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        if split == "train":
            self.split = "Training"
        elif split == "test":
            self.split = "Test"
        else:
            raise ValueError("")
        
        self.pattern_config_path="/data/hanning/SL_dust3r/configs/pattern_config.yaml"

        self.loaded_data = self._load_data(self.split)
        assert self.num_views <= 2, "目前只支持单视角或双视角！"
        self.initialize_pattern_config(config=self.pattern_config_path, split=split, **kwargs)

    def _load_data(self, split):
        with megfile.smart_open(osp.join(self.ROOT, split, 'all_metadata.npz'), 'rb') as f:
            with np.load(f) as data:
                self.scenes = data['scenes']
                self.sceneids = data['sceneids']
                self.images = data['images']
                self.intrinsics = data['intrinsics'].astype(np.float32)
                self.trajectories = data['trajectories'].astype(np.float32)
                self.pairs = data['pairs'][:, :2].astype(int)

    def __len__(self):
        return len(self.pairs)

    def _get_views(self, idx, resolution, rng):

        image_idx1, image_idx2 = self.pairs[idx]

        views = []
        for view_idx in [image_idx1, image_idx2] if self.num_views == 2 else [image_idx1]:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.split, self.scenes[scene_id])

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_iio(osp.join(scene_dir, 'vga_wide', basename.replace('.png', '.jpg')))
            # Load depthmap
            depthmap = imread_iio(osp.join(scene_dir, 'lowres_depth', basename), cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)

            view_dict = dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='arkitscenes',
                label=self.scenes[scene_id] + '|' + basename,
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

    dataset = ARKitScenes(split='train', resolution=224, aug_crop=16)

    for idx in np.random.permutation(min(1, len(dataset))):
        views = dataset[idx]
        assert len(views) == 2
        print(f"train {idx}: {view_name(views[0])}, {view_name(views[1])}")
        
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        print("known_depth:",views[0]['known_depth'])
        print("camera:",views[0]['camera_intrinsics'])
        print("known_rays",views[0]['known_rays'])
        print("camera_pose1:",views[0]['camera_pose'])
        print("camera_pose2:",views[1]['camera_pose'])
        print("known_pose1",views[0]['known_pose'])
        print("known_pose2",views[1]['known_pose'])