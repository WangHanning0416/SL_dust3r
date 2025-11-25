#用于制造：view1是正常的结构光数据集，view2被替换为pattern且相应地增加一些量用于后续的监督用
import numpy as np
import megfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from numbers import Number
import warnings

from dust3r.utils.image import imread_iio, estimate_reflectance_tensor, _dbg_save_img_tensor
from dust3r.utils.geometry import warp_coord, intrinsics_from_fov, extrinsics_from_RT, rotation_matrix_from_euler, fov_from_intrinsics, extrinsics_from_euler_transl
from dust3r.utils.config import load_config, random_sample_properties_from_config
from dust3r.datasets.utils.cropping import resize_crop_img_tensor


class PatternSynthMixin:
    def initialize_pattern_config(self, config:Union[dict, str], split:str, **kwargs):
        '''config can be overwritten by kwargs.'''
        assert split in ['train','test']
        if isinstance(config, str):
            assert megfile.smart_isfile(config), f"inexistent file: {config}"
            config = load_config(config)
        else:
            assert isinstance(config, dict)
        
        self.pattern_paths = config['pattern_paths']
        self.pattern_config = config.get(split, None)
        if self.pattern_config is None:
            raise NotImplementedError(f"config[{split}] unfinished.")
        self.patterns = self.load_pattern()

        # dynamically define the class that process pattern properly.

        if hasattr(self, '_original_class') and self._original_class is not None:
            warnings.warn(f"[PatternSynthMixin] warning: it is already initialized! ", UserWarning)
            return
        
        print(f"[{self.__class__.__name__} instance]: PatternSynthMixin initialized. __getitem__ will be overrode and cross_view_invariancy will be enabled.")

        # save instance's original class.
        self._original_class = self.__class__

        # toggle cross view invariant
        if hasattr(self, '_crop_resize_if_necessary'):
            self.invariancy_before = self.cross_view_invariant
            self._toggle_cross_view_invariant(True)

        class PatternSynthPatched(self._original_class):
            def __getitem__(self, idx):
                views = super().__getitem__(idx)
                cfgs = random_sample_properties_from_config(self.pattern_config)
                pat = self.patterns[cfgs['pattern_idx']].clone()
                
                # 获取最终图像尺寸 H, W
                H, W = views[0]['img'].shape[-2:]
                # resize and crop pat first !
                pat = resize_crop_img_tensor(pat, (H, W))

                # 先处理view1，计算投影仪深度图
                view1 = views[0]
                img1 = view1['img'] * 0.5 + 0.5  # unnormalized.
                dep1 = view1['depthmap'] # (H, W)
                cam_fov = fov_from_intrinsics(view1['camera_intrinsics'])
                proj_fov = cfgs['fov'] + cam_fov
                x_transl = cfgs['x_transl_abs'] * (1 if cfgs['x_transl_sign']==1 else -1)
                
                # Intri (K_proj)
                proj_intri = intrinsics_from_fov(proj_fov, pat.shape[-2:])
                # C2P Extri (T_c2p)
                c2p = extrinsics_from_euler_transl(
                    cfgs['x_rot'], cfgs['y_rot'], cfgs['z_rot'], 
                    x_transl, cfgs['y_transl'], cfgs['z_transl'], degree=True, order='xyz'
                )

                synthesized_img, warpped_pattern = structured_light_synthesize(
                    img1, dep1, pat, dep1 > 0, view1['camera_intrinsics'], proj_intri, c2p,
                    cfgs['gamma'], cfgs['alpha'], cfgs['beta'], None, cfgs['noise_scale']
                )
                
                img1 = synthesized_img * 2 - 1
                
                # 计算投影仪深度图（用于view2的监督）
                cam_intri = view1['camera_intrinsics']
                depth_np = dep1
                K_cam_np = cam_intri
                c2p_np = c2p # 4x4 T_c2p
                R_cp = c2p_np[:3, :3]
                t_cp = c2p_np[:3, 3].reshape(3, 1)

                u, v = np.meshgrid(np.arange(W), np.arange(H))
                points_2d = np.stack([u.flatten(), v.flatten(), np.ones(H*W)], axis=1).T
                K_cam_inv = np.linalg.inv(K_cam_np)
                P_normalized = K_cam_inv @ points_2d
                depth_flat = depth_np.flatten()
                valid_depth_mask = depth_flat > 1e-6 
                P_normalized_valid = P_normalized[:, valid_depth_mask]
                depth_valid = depth_flat[valid_depth_mask]
                P_c_valid = P_normalized_valid * depth_valid
                
                # 2. 变换到投影仪坐标系 (P_p)
                P_p_valid = (R_cp @ P_c_valid) + t_cp
                if not isinstance(P_p_valid, np.ndarray):
                    # 如果它是 Tensor，先转 numpy
                    P_p_valid = P_p_valid.cpu().numpy() if P_p_valid.is_cuda else P_p_valid.numpy()
                
                K_proj_np = proj_intri.cpu().numpy() 
                P_proj_2d_homo = K_proj_np @ P_p_valid
                Z_p = P_proj_2d_homo[2, :]
        
                final_projection_mask = Z_p > 1e-6 
                P_p_filtered_by_z = P_p_valid[:, final_projection_mask]
                P_proj_2d_valid = P_proj_2d_homo[:, final_projection_mask]
                Z_p_final = Z_p[final_projection_mask]
                
                u_proj = P_proj_2d_valid[0, :] / Z_p_final
                v_proj = P_proj_2d_valid[1, :] / Z_p_final

                valid_pattern_bounds_mask = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H)

                depth_valid_indices = np.where(valid_depth_mask)[0]
                valid_projection_indices = np.where(final_projection_mask)[0]
                final_valid_indices = depth_valid_indices[valid_projection_indices][np.where(valid_pattern_bounds_mask)[0]]

                u_proj_final = u_proj[valid_pattern_bounds_mask]
                v_proj_final = v_proj[valid_pattern_bounds_mask]
                
                coords_int_pat = np.round(np.stack([u_proj_final, v_proj_final], axis=1)).astype(np.int32)
                coords_int_pat[:, 0] = np.clip(coords_int_pat[:, 0], 0, W - 1)
                coords_int_pat[:, 1] = np.clip(coords_int_pat[:, 1], 0, H - 1)

                pattern_final_valid_mask_HW = np.zeros((H, W), dtype=bool) # 尺寸为 H x W

                pattern_final_valid_mask_HW[coords_int_pat[:, 1], coords_int_pat[:, 0]] = True
                
                dense_projector_coords_3d = np.full((H*W, 3), np.nan, dtype=np.float32)
                P_p_final_coords = P_p_filtered_by_z.T[valid_pattern_bounds_mask, :] 
                dense_projector_coords_3d[final_valid_indices] = P_p_final_coords
                dense_projector_coords_3d = dense_projector_coords_3d.reshape(H, W, 3)
                
                projector_depthmap_from_pattern_view = np.full((H, W), np.nan, dtype=np.float32)
                
                Z_p_final_coords = P_p_filtered_by_z[2, :][valid_pattern_bounds_mask]

                projector_depthmap_from_pattern_view[coords_int_pat[:, 1], coords_int_pat[:, 0]] = Z_p_final_coords

                # 更新view1
                view1.update(dict(
                    img=img1, 
                    pat=pat * 2 - 1, 
                    proj_intrinsics=proj_intri, 
                    # --- [修改: 添加 known_pose] ---
                    known_pose=c2p.cpu().numpy().astype(np.float32), 
                    # --- [修改结束] ---
                    c2p=c2p,
                    gamma=cfgs['gamma'], alpha=cfgs['alpha'], beta=cfgs['beta'], noise_scale=cfgs['noise_scale'],
                    # 输出 1 (相机视图索引): Pattern 上的有效性掩码 (H_Pat x W_Pat)
                    pattern_valid_mask=pattern_final_valid_mask_HW.astype(np.bool_), 
                    # 输出 2 (相机视图索引): view1每个点对应的深度 (H_RGB x W_RGB x 3)
                    projector_coords_3d=dense_projector_coords_3d.astype(np.float32),
                ))
                
                new_views = [view1]
                
                # 处理view2：使用pattern图像，完全在投影仪坐标系下
                if len(views) > 1:
                    view2 = views[1]
                    # view2使用pattern图像
                    img2 = pat * 2 - 1  # pattern图像，归一化到[-1, 1]
                    
                    # 为view2创建depthmap：使用projector_depthmap_from_pattern_view
                    # 由于camera_intrinsics是proj_intri，camera_pose是单位矩阵，
                    # 所以BaseStereoViewDataset计算出的pts3d会在投影仪坐标系下
                    # （虽然这个pts3d不会被使用，因为model.py中会删除它，只保留projector_depth）
                    depthmap2 = projector_depthmap_from_pattern_view.copy()
                    depthmap2[~pattern_final_valid_mask_HW] = 0  # 无效区域设为0
                    depthmap2[~np.isfinite(depthmap2)] = 0  # nan区域设为0
                    
                    # view2更新：
                    # 1. 使用pattern图像作为输入
                    # 2. 将camera_intrinsics改为proj_intrinsics（投影仪内参），使模型在投影仪坐标系下工作
                    # 3. camera_pose设为单位矩阵（因为view2在投影仪坐标系下，不需要额外的pose变换）
                    # 4. depthmap使用投影仪深度图（这样计算出的pts3d会在投影仪坐标系下）
                    # 5. 添加投影仪深度图作为监督信号（每个pattern点对应的RGB点在投影仪坐标系下的深度）
                    view2.update(dict(
                        img=img2,
                        depthmap=depthmap2.astype(np.float32),  # 使用投影仪深度图作为depthmap
                        pat=pat * 2 - 1,
                        camera_intrinsics=proj_intri,  # 使用投影仪内参替代相机内参
                        camera_pose=np.eye(4, dtype=np.float32),  # 单位矩阵，因为view2在投影仪坐标系下
                        proj_intrinsics=proj_intri,
                        c2p=c2p,
                        gamma=cfgs['gamma'], alpha=cfgs['alpha'], beta=cfgs['beta'], noise_scale=cfgs['noise_scale'],
                        projector_depthmap_from_pattern_view=projector_depthmap_from_pattern_view.astype(np.float32),
                        # Pattern 上的有效性掩码
                        pattern_valid_mask=pattern_final_valid_mask_HW.astype(np.bool_),
                        # 标记这是pattern view
                        is_pattern_view=True,
                    ))
                    new_views.append(view2)
                
                return new_views
        

        # point the instance's __class__ to our new PatchedClass
        self.__class__ = PatternSynthPatched

    def deinitialize(self):
        if not hasattr(self, '_original_class') or self._original_class is None:
            warnings.warn(f'[PatternSynthMixin] warning: it is not initialized, so no need to perform deinitialize')
            return
        
        print(f"[{self._original_class.__name__} instance]: PatternSynthMixin deinitialized.")

        self.__class__ = self._original_class

        # recover
        # toggle cross view invariant
        if hasattr(self, '_crop_resize_if_necessary'):
            self._toggle_cross_view_invariant(self.invariancy_before)
            del self.invariancy_before

        del self._original_class

    def load_pattern(self):
        self.patterns = []
        for path in self.pattern_paths:
            pat = imread_iio(path)
            pat = torch.from_numpy(pat).to(torch.float32) / 255.  # [0, 1]
            pat = pat.permute(2,0,1)
            self.patterns.append(pat)
        return self.patterns



def __validate_img(img:torch.Tensor):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if img.ndim == 3:
        img = img.unsqueeze(0)
    assert img.ndim == 4, f"{img.shape}"
    return img

def __validate_depth(img_valid:torch.Tensor, depth:torch.Tensor):
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth)
    if depth.ndim == 3:  # B,H,W
        assert depth.shape[0] == img_valid.shape[0]
        assert depth.shape[1:] == img_valid.shape[2:], "unmatched resolution between image and depth."
        return depth
    else:
        assert depth.ndim == 2 and img_valid.shape[0] == 1, "image is batched while depth is not."
        return depth.unsqueeze(0)

def __validate_pattern(img_valid:torch.Tensor, pattern:torch.Tensor):
    if isinstance(pattern, np.ndarray):
        pattern = torch.from_numpy(pattern)
    if pattern.ndim == 4:  # batched, bchw
        assert pattern.shape[0] == img_valid.shape[0] or pattern.shape[0] == 1, \
            "pattern's batch dim must either match the image or be 1 (indicating broadcasting)"
        pattern = pattern.expand((img_valid.shape[0], *pattern.shape[1:]))
    else:
        assert pattern.ndim == 3, "pattern.ndim must be either 4 (bchw) or 3 (chw, need broadcasting)"
        pattern = pattern.unsqueeze(0).expand((img_valid.shape[0], *pattern.shape))
    return pattern.to(img_valid.device)
    
def __validate_intri(img_valid:torch.Tensor, intri: Union[torch.Tensor, Number, List[Number]]):
    b, c, h, w = img_valid.shape
    if isinstance(intri, np.ndarray):
        intri = torch.from_numpy(intri)
    if isinstance(intri, torch.Tensor):
        assert intri.shape[-2:] == (3,3), "intrinsic matrix must be 3x3."
        if intri.ndim == 3:
            assert intri.shape[0] == b or intri.shape[0] == 1, \
                "intri's batch dim must either match the image or be 1 (indicating broadcasting)"
            intri = intri  # it will be broadcast automatically.
        else:
            assert intri.ndim == 2, f"intrinsic.shape should be either (3,3) or (b,3,3), but found {intri.shape}"  # 3x3
            intri = intri  # it will be broadcast automatically.
    elif isinstance(intri, Number):  # a scalar indicating fov, in degree!
        intri = intrinsics_from_fov(intri, (h, w))
    else:
        assert isinstance(intri, (list, tuple)) and (len(intri) == b or len(intri) == 1), f"unkown intrinsic type, or invalid length. {intri}"
        intri = [intrinsics_from_fov(fov, (h,w)) for fov in intri]
        intri = torch.stack(intri, dim=0)
    return intri.to(img_valid.device)
    
def __validate_pose(img_valid:torch.Tensor, pose: Union[torch.Tensor, tuple, List[tuple]]):
    b, c, h, w = img_valid.shape
    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose)
    if isinstance(pose, torch.Tensor):
        assert pose.shape[-2:] in [(3,3), (3,4), (4,4)], "transformation matrix should be one of [(3,3),(3,4),(4,4)]"
        if pose.ndim == 3:
            assert pose.shape[0] == b or pose.shape[0] == 1, \
                "pose's batch dim must either match the image or be 1 (indicating broadcasting)"
            pose = pose
        else:
            assert pose.ndim == 2, f"intrinsic.shape should be either (_,_) or (b,_,_), but found {pose.shape}"  # 3x3
            pose = pose
    elif isinstance(pose, (list, tuple)) and isinstance(pose[0], Number):
        pose = extrinsics_from_euler_transl(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], degree=True, order='xyz')
    else:
        assert isinstance(pose, (list, tuple)) and isinstance(pose[0], (list, tuple)) and len(pose) in [1, b]
        pose = [extrinsics_from_euler_transl(p[0], p[1], p[2], p[3], p[4], p[5], degree=True, order='xyz') for p in pose]
        pose = torch.stack(pose, dim=0)
    return pose.to(img_valid.device)

def __validate_scalar(img_valid:torch.Tensor, val:Union[Number, List[Number]]):
    b, c, h, w = img_valid.shape
    if isinstance(val, Number):
        return val
    else:
        assert isinstance(val, (list, tuple)) and (len(val) == b or len(val) == 1), \
            "scalar parameter must either be a number or a list/tuple of length equal to batch size."
        return val[0] if len(val) == 1 else torch.tensor(val, dtype=torch.float32).reshape((b,) + (1,)*(img_valid.ndim-1)).to(img_valid.device)


def structured_light_synthesize(
        img: torch.Tensor, depth:torch.Tensor, pattern: torch.Tensor, valid_mask:torch.Tensor,
        cam_intri:Union[torch.Tensor, Number, List[Number]], 
        proj_intri:Union[torch.Tensor, Number, List[Number]], 
        c2p:Union[torch.Tensor, Tuple[Number], List[Tuple[Number]]],
        gamma:Union[Number, List[Number]] = 1, alpha:Union[Number, List[Number]] = 1, beta:Union[Number,List[Number]] = 0.5,
        reflectance:Optional[torch.Tensor] = None, noise_scale: Number = 0.):
    """ Synthesize a pattern image from a depth map and camera parameters.
        
        delta(gamma * (alpha * I + beta / (1+0.1*d^2) * P * R) + N(0, noise_scale)), delta = torch.clamp()  
        beta: projector energy.
        alpha: ambient energy.
        gamma: exposure

    Args:
        img: ((B,) C, H, W), in range (0,1)
        depth: ((B,) H, W) depth map
        pattern: ((B,) C, H, W) pattern image in range(0,1)
        valid_mask: ((B,) H, W), valid mask
        cam_intri: ((B, ) 3, 3), or number/list[number] (representing fov **in degree**).
        proj_intri: ((B,) 3, 3), or number/list[number] (representing fov **in degree**).
        c2p: ((B,) 4,4) camera extrinsics, or (x_rot, y_rot, z_rot, x_move, y_move, z_move), can be list. rotation should be in degree!
            note: if a tuple is used, the rot & trans should be "rot and trans that move the projector to align with the camera."
        reflectance: ((B,) C, H, W) pre-computed reflectance image. If provided, the reflectance will be used directly instead of estimating from img. (reflectance can be img itself for acceleration.)
    Returns:
        synthesized_pattern: ((B,) C, H, W) synthesized pattern image, warpped pattern: ((B,) C, H, W)
    """
    unbatched = img.ndim == 3
    ph, pw = pattern.shape[-2:]
    ih, iw = img.shape[-2:]
    # validate and conversion.
    img = __validate_img(img)
    depth = __validate_depth(img, depth)
    pattern = __validate_pattern(img, pattern)
    valid_mask = __validate_depth(img, valid_mask)
    cam_intri = __validate_intri(img, cam_intri)
    proj_intri = __validate_intri(pattern, proj_intri)
    c2p = __validate_pose(img, c2p)
    # relectance:
    if reflectance is None:
        reflectance, s_y = estimate_reflectance_tensor(img, sigma=75, use_guided=False, S_scalefactor=0.5)  # b,c,h,w
    else:
        reflectance = __validate_img(reflectance)
    # # DEBUG
    # out = 'tmp/reflectance.png'
    # _dbg_save_img_tensor(out, reflectance)
    # # END DEBUG
    gamma = __validate_scalar(img, gamma)
    alpha = __validate_scalar(img, alpha)
    beta = __validate_scalar(img, beta)

    warpped_coord, new_depth = warp_coord((ih, iw), depth, cam_intri, proj_intri, c2p, (ph, pw), normalize_type='grid_sample') # B,H,W,2, 注意存在nan.(depth=0)

    warpped_coord[(~valid_mask).unsqueeze(-1).expand_as(warpped_coord)] = -2.  # 这样grid_sample的时候会自动因为超界变成0.
    warpped_pattern = F.grid_sample(pattern, warpped_coord, 'bilinear', align_corners=False) # b,c,h,w

    # print(pattern.shape, beta, warpped_pattern.shape, reflectance.shape, new_depth.shape, img.shape, alpha)
    # depth应该需要是射线长度，而不是z轴距离，但这里简化了.
    attenuation = 0.04
    result = gamma * ( \
            alpha * img + \
            beta * warpped_pattern * reflectance * (1 / (1 + attenuation * new_depth.permute(0, 3, 1, 2)**2)) \
        )  # B,C,H,W
    # # DEBUG
    # out = "tmp/reflected_attenuated_warpped_pattern.png"
    # p = (beta * warpped_pattern * reflectance * (1 / (1 + 0.1 * new_depth.permute(0, 3, 1, 2)**2)))
    # _dbg_save_img_tensor(out, p)
    # # END DEBUG
    if noise_scale > 0:
        result = result + torch.randn_like(result) * noise_scale
    result = torch.clamp(result, 0, 1)
    return result.squeeze(0) if unbatched else result, warpped_pattern.squeeze(0) if unbatched else warpped_pattern


if __name__ == '__main__':
    import cv2
    import shutil
    import time

    inp_img = '/nvme/data/jiaheng/data/Replica/office0/results/frame000000.jpg'
    inp_dep = '/nvme/data/jiaheng/data/Replica/office0/results/depth000000.png'
    inp_pat = '/nvme/data/jiaheng/dev/slslam/SL_dust3r/dust3r/datasets/patterns/alacarte.png'
    out = 'tmp/sl_synth.png'

    cam_intri = torch.tensor([[600., 0, 599.5], [0, 600, 339.5], [0, 0, 1]], dtype=torch.float32)
    depth_scale = 6553.5

    proj_intri = 60 # 90 # fov
    pose = [5, 5, 2, -0.1, 0, 0]

    gamma = 1
    alpha = 1. # 1.2
    beta = 2
    noise = 0


    img = torch.from_numpy(imread_iio(inp_img).astype(np.float32) / 255.).permute(2,0,1)
    dep = torch.from_numpy(imread_iio(inp_dep, cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale)
    print('dep:', dep.min(), dep.max())
    valid_mask = dep > 0
    pat = torch.from_numpy(imread_iio(inp_pat).astype(np.float32) / 255.).permute(2,0,1)

    with torch.no_grad():
        s = time.time()
        ret, warpped_pattern = structured_light_synthesize(
            img, dep, pat, valid_mask, cam_intri, proj_intri, pose, gamma=gamma, alpha=alpha, beta=beta, noise_scale=noise
        )
        print(f"synthesizing time: {time.time() - s} s.")

    ret = np.uint8(ret.permute(1,2,0).detach().cpu().numpy() * 255)
    h,w,c = ret.shape

    figure = np.zeros((h, 3*w, 3), dtype=np.uint8)
    figure[:,:w] = (img.permute(1,2,0) * 255).detach().cpu().numpy().astype(np.uint8)
    figure[:,w:2*w] = (warpped_pattern.permute(1,2,0) * 255).detach().cpu().numpy().astype(np.uint8)
    figure[:,2*w:] = ret

    cv2.imwrite(out, figure)
    print(f"result saved to {out}")