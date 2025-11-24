# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# geometry utilitary functions
# --------------------------------------------------------
import torch
import numpy as np
from typing import Union
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree as KDTree

from dust3r.utils.misc import invalid_to_zeros, invalid_to_nans
from dust3r.utils.device import to_numpy

def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def intrinsics_from_fov(fov, image_size) -> torch.Tensor:
    """ Compute camera intrinsics from field of view and image size.

    fov: field of view in degree
    image_size: (height, width) tuple

    Returns a 3x3 camera intrinsics matrix.
    """
    H, W = image_size
    focal_length = 0.5 * W / np.tan(0.5 * np.deg2rad(fov))
    K = np.array([[focal_length, 0, W / 2],
                  [0, focal_length, H / 2],
                  [0, 0, 1]], dtype=np.float32)
    return torch.from_numpy(K)


def fov_from_intrinsics(intri, img_size=None):
    cx = intri[0, 2]
    fx = intri[0, 0] if intri[0,0] > 0 else intri[0,1]  # 可能是把一个竖向的图片打横了，变成 [0, fx, cx], [fy,0,cy]
    if isinstance(intri, torch.Tensor):
        cx = cx.item()
        fx = fx.item()
    if img_size is not None:
        w = img_size[-1]
    else:
        w = cx * 2
    fov = np.rad2deg(np.arctan(0.5 * w / fx) * 2)
    return float(fov)

def extrinsics_from_RT(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """ Build a 4x4 extrinsics matrix from rotation and translation.

    R: 3x3 rotation matrix
    T: 3x1 translation vector

    Returns a 4x4 extrinsics matrix.
    """
    extrinsics = torch.eye(4, dtype=R.dtype, device=R.device)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = T.squeeze()
    return extrinsics

def extrinsics_from_euler_transl(x_rot, y_rot, z_rot, x_transl, y_transl, z_transl, degree=True, order='xyz'):
    R = rotation_matrix_from_euler(x_rot, y_rot, z_rot, degrees=degree, order=order)
    T = torch.tensor([x_transl, y_transl, z_transl], dtype=torch.float32)
    return extrinsics_from_RT(R, T)


def rotation_matrix_from_euler(x, y, z, degrees: bool = False, order='xyz') -> torch.Tensor:
    """ Convert Euler angles to a rotation matrix.

    x, y, z: rotation angles around the x, y, z axes
    degrees: whether the input angles are in degrees
    order: order of rotations, e.g., 'xyz', 'zyx', etc.

    Returns a 3x3 rotation matrix.
    """
    if degrees:
        r = R.from_euler(order, [x, y, z], degrees=True)
    else:
        r = R.from_euler(order, [x, y, z], degrees=False)
    R_mat = r.as_matrix()
    return torch.from_numpy(R_mat).to(dtype=torch.float32)


def warp_coord(grid: Union[tuple, torch.Tensor], z: torch.Tensor, 
                         Ksource:torch.Tensor, Ktarget:torch.Tensor, 
                         Trf:torch.Tensor,
                         target_size:list = None,
                         normalize_type:str = None):
    """ Apply a warp operation to a pixel grid.
        **z must be batched. its 1st dim should be batch_dim

    grid: tuple: (H, W) or tensor (...,2) or (...,3)
    Ksource: 3x3 source camera intrinsics. pixel unit.
    Ktarget: 3x3 target camera intrinsics
    Trf: 4x4, 3x4 or 3x3 (rotation only) matrix transforming points from source to target camera coordinates
    z: (..., 1) or (...,) ;  if grid is a tuple, z must be (..., H, W). If Trf is rotation-only, z can be ones.
    
    normalize_type: [None, 'none', 'grid_sample']
    target_size: (H, W)

    Returns an array of projected 2d points (B,H,W,2) and tranformed new z (B,H,W,1).  
    """
    def reshape_matrix(mat:torch.Tensor, grid_shape:tuple, num_base_dim=2):  # grid_shape: (b, ..., 3) target: (b, ..., nh, nw)
        if mat.ndim == num_base_dim:
            return mat  # automatically broadcast.
        assert mat.ndim == 3 and mat.shape[0] == grid_shape[0]
        fill = len(grid_shape) - 2
        new_shape = (mat.shape[0], ) + (1, )*fill + mat.shape[-num_base_dim:]
        return mat.reshape(new_shape)

    b = z.shape[0]
    if isinstance(grid, tuple):
        H, W = grid
        grid = xy_grid(W, H, z.device if hasattr(z, 'device') else None).to(z.dtype)  # (H, W, 2)
    else:
        assert grid.shape[-1] in [2,3] and grid.shape[0] == b, "tensor grid must be batched and match the batch_num of z."
    if grid.shape[-1] == 2:  # homogeneous coords not given
        ones = torch.ones((*grid.shape[:-1], 1), dtype=grid.dtype, device=grid.device)
        grid = torch.concat((grid, ones), dim=-1)  # (..., 3)

    if z.shape[-1] != 1:
        z = z.unsqueeze(-1)
    
    Ksource_inv = torch.linalg.inv(Ksource)
    coord = torch.einsum('...ij, ...j -> ...i', reshape_matrix(Ksource_inv, grid.shape), grid) * z  # (..., 3)
    Trf = reshape_matrix(Trf, grid.shape)
    if Trf.shape[-1] == 4:
        R, T = Trf[..., :3, :3], Trf[..., :3, 3]
    else:
        assert Trf.shape[-2:] == (3,3)
        R = Trf
        T = 0.
    coord = torch.einsum('...ij, ...j -> ...i', R, coord) + T  # (..., 3)
    new_z = coord[..., 2:3]
    coord = torch.einsum('...ij, ...j -> ...i', reshape_matrix(Ktarget, coord.shape), coord)  # (..., 3)
    coord = coord[..., :2] / coord[..., 2:3]  # (..., 2)

    if normalize_type is None or normalize_type.lower() == 'none':
        return coord, new_z
    
    if target_size is None:
        target_size = Ktarget[..., :2, -1] * 2  # (b,) 2. [w, h]
    else:
        target_size = torch.tensor(target_size, dtype=coord.dtype, device=coord.device).flip(dims=(-1,)) # flip last dim. now (w,h)
    
    if normalize_type == 'grid_sample':
        coord = coord / reshape_matrix(target_size, coord.shape, num_base_dim=1) * 2 - 1
    else:
        raise NotImplementedError

    return coord, new_z



def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')


def depthmap_to_pts3d(depth, pseudo_focal, pp=None, **_):
    """
    Args:
        - depthmap (BxHxW array):
        - pseudo_focal: [B,H,W] ; [B,2,H,W] or [B,1,H,W]
    Returns:
        pointmap of absolute coordinates (BxHxWx3 array)
    """

    if len(depth.shape) == 4:
        B, H, W, n = depth.shape
    else:
        B, H, W = depth.shape
        n = None
    if len(pseudo_focal.shape) == 3:  # [B,H,W]
        pseudo_focalx = pseudo_focaly = pseudo_focal
    elif len(pseudo_focal.shape) == 4:  # [B,2,H,W] or [B,1,H,W]
        pseudo_focalx = pseudo_focal[:, 0]
        if pseudo_focal.shape[1] == 2:
            pseudo_focaly = pseudo_focal[:, 1]
        else:
            pseudo_focaly = pseudo_focalx
    if pseudo_focal.ndim == 1: # [B,]
        pseudo_focalx = pseudo_focal[:, None, None]
        pseudo_focaly = pseudo_focal[:, None, None]  # [B,1,1]
    else:
        raise NotImplementedError("Error, unknown input focal shape format.")

    # assert pseudo_focalx.shape == depth.shape[:3]
    # assert pseudo_focaly.shape == depth.shape[:3]
    grid_x, grid_y = xy_grid(W, H, cat_dim=0, device=depth.device)[:, None]
    # grid = xy_grid(W, H, cat_dim=0, device=depth.device)

    # set principal point
    if pp is None:
        grid_x = grid_x - (W - 1) / 2
        grid_y = grid_y - (H - 1) / 2
    else:
        grid_x = grid_x.expand(B, -1, -1) - pp[:, 0, None, None]
        grid_y = grid_y.expand(B, -1, -1) - pp[:, 1, None, None]

    if n is None:
        # depth: B,H,W, focal: B,H,W or B,1,1
        pts3d = torch.empty((B, H, W, 3), device=depth.device)
        pts3d[..., 0] = depth * grid_x / pseudo_focalx
        pts3d[..., 1] = depth * grid_y / pseudo_focaly
        pts3d[..., 2] = depth
    else:
        pts3d = torch.empty((B, H, W, 3, n), device=depth.device)
        pts3d[..., 0, :] = depth * (grid_x / pseudo_focalx)[..., None]
        pts3d[..., 1, :] = depth * (grid_y / pseudo_focaly)[..., None]
        pts3d[..., 2, :] = depth
    return pts3d


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    return X_cam, valid_mask


def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def normalize_pointcloud(pts1, pts2, norm_mode='avg_dis', valid1=None, valid2=None, ret_factor=False):
    """ renorm pointmaps pts1, pts2 with norm_mode
    """
    assert pts1.ndim >= 3 and pts1.shape[-1] == 3
    assert pts2 is None or (pts2.ndim >= 3 and pts2.shape[-1] == 3)
    norm_mode, dis_mode = norm_mode.split('_')

    if norm_mode == 'avg':
        # gather all points together (joint normalization)
        nan_pts1, nnz1 = invalid_to_zeros(pts1, valid1, ndim=3)
        nan_pts2, nnz2 = invalid_to_zeros(pts2, valid2, ndim=3) if pts2 is not None else (None, 0)
        all_pts = torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)
        if dis_mode == 'dis':
            pass  # do nothing
        elif dis_mode == 'log1p':
            all_dis = torch.log1p(all_dis)
        elif dis_mode == 'warp-log1p':
            # actually warp input points before normalizing them
            log_dis = torch.log1p(all_dis)
            warp_factor = log_dis / all_dis.clip(min=1e-8)
            H1, W1 = pts1.shape[1:-1]
            pts1 = pts1 * warp_factor[:, :W1 * H1].view(-1, H1, W1, 1)
            if pts2 is not None:
                H2, W2 = pts2.shape[1:-1]
                pts2 = pts2 * warp_factor[:, W1 * H1:].view(-1, H2, W2, 1)
            all_dis = log_dis  # this is their true distance afterwards
        else:
            raise ValueError(f'bad {dis_mode=}')

        norm_factor = all_dis.sum(dim=1) / (nnz1 + nnz2 + 1e-8)
    else:
        # gather all points together (joint normalization)
        nan_pts1 = invalid_to_nans(pts1, valid1, ndim=3)
        nan_pts2 = invalid_to_nans(pts2, valid2, ndim=3) if pts2 is not None else None
        all_pts = torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)

        if norm_mode == 'avg':
            norm_factor = all_dis.nanmean(dim=1)
        elif norm_mode == 'median':
            norm_factor = all_dis.nanmedian(dim=1).values.detach()
        elif norm_mode == 'sqrt':
            norm_factor = all_dis.sqrt().nanmean(dim=1)**2
        else:
            raise ValueError(f'bad {norm_mode=}')

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts1.ndim:
        norm_factor.unsqueeze_(-1)

    res = pts1 / norm_factor
    if pts2 is not None:
        res = (res, pts2 / norm_factor)
    if ret_factor:
        res = res + (norm_factor,)
    return res


@torch.no_grad()
def get_joint_pointcloud_depth(z1, z2, valid_mask1, valid_mask2=None, quantile=0.5):
    # set invalid points to NaN
    _z1 = invalid_to_nans(z1, valid_mask1).reshape(len(z1), -1)
    _z2 = invalid_to_nans(z2, valid_mask2).reshape(len(z2), -1) if z2 is not None else None
    _z = torch.cat((_z1, _z2), dim=-1) if z2 is not None else _z1

    # compute median depth overall (ignoring nans)
    if quantile == 0.5:
        shift_z = torch.nanmedian(_z, dim=-1).values
    else:
        shift_z = torch.nanquantile(_z, quantile, dim=-1)
    return shift_z  # (B,)


@torch.no_grad()
def get_joint_pointcloud_center_scale(pts1, pts2, valid_mask1=None, valid_mask2=None, z_only=False, center=True):
    # set invalid points to NaN
    _pts1 = invalid_to_nans(pts1, valid_mask1).reshape(len(pts1), -1, 3)
    _pts2 = invalid_to_nans(pts2, valid_mask2).reshape(len(pts2), -1, 3) if pts2 is not None else None
    _pts = torch.cat((_pts1, _pts2), dim=1) if pts2 is not None else _pts1

    # compute median center
    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values  # (B,1,3)
    if z_only:
        _center[..., :2] = 0  # do not center X and Y

    # compute median norm
    _norm = ((_pts - _center) if center else _pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values
    return _center[:, None, :, :], scale[:, None, None, None]


def find_reciprocal_matches(P1, P2):
    """
    returns 3 values:
    1 - reciprocal_in_P2: a boolean array of size P2.shape[0], a "True" value indicates a match
    2 - nn2_in_P1: a int array of size P2.shape[0], it contains the indexes of the closest points in P1
    3 - reciprocal_in_P2.sum(): the number of matches
    """
    tree1 = KDTree(P1)
    tree2 = KDTree(P2)

    _, nn1_in_P2 = tree2.query(P1, workers=8)
    _, nn2_in_P1 = tree1.query(P2, workers=8)

    reciprocal_in_P1 = (nn2_in_P1[nn1_in_P2] == np.arange(len(nn1_in_P2)))
    reciprocal_in_P2 = (nn1_in_P2[nn2_in_P1] == np.arange(len(nn2_in_P1)))
    assert reciprocal_in_P1.sum() == reciprocal_in_P2.sum()
    return reciprocal_in_P2, nn2_in_P1, reciprocal_in_P2.sum()


def get_med_dist_between_poses(poses):
    from scipy.spatial.distance import pdist
    return np.median(pdist([to_numpy(p[:3, 3]) for p in poses]))
