import numpy as np
import torch.nn.functional as F
import torch.nn.functional as F
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import geotrf, inv

ALLOWED_MODALITIES = ['K1','K2','D1','D2','RT']

def dot(A, B): 
    return (A.unsqueeze(-2) @ B.unsqueeze(-1)).squeeze(-1)

def compute_rays(pix, K):
    return geotrf(inv(K), pix, ncol=3)

def _gen_rays(true_shape, K):
    
    H, W = true_shape
    pix = np.mgrid[:W, :H].T.astype(np.float32)
    rays = compute_rays(pix, K) # H,W,3
    return rays.transpose(2,0,1) # 3,H,W
    
def gen_rays(view):
    return _gen_rays(view['depthmap'].shape, view['camera_intrinsics'])


def gen_sparse_depth( view, valid_mask, n_pts_min, n_pts_max, rng=np.random, norm=True ):
    H, W = view['depthmap'].shape
    n_pts_max = n_pts_max or H*W

    # random sampling across valid points only
    valid_pixels = valid_mask.ravel()
    valid_indices = valid_pixels.ravel().nonzero()[0]
    num_valid_pixels = len(valid_indices)
            
    # select points
    n_pts_min = min(n_pts_min, num_valid_pixels)
    n_pts_max = min(n_pts_max, num_valid_pixels)
    n_pts = randint_logspace(rng, n_pts_min, n_pts_max, bias=1)
    unmasked_indices = rng.choice(valid_indices, n_pts, replace=False)

    mask = np.zeros((H,W), dtype=np.float32)
    mask.ravel()[unmasked_indices] = 1

    sparse_depth = view['depthmap'] * mask
    if norm and n_pts: # normalize
        sparse_depth /= sparse_depth[mask > 0].mean()

    return np.stack((sparse_depth, mask)) # (2,H,W)


def randint_logspace(rng, a, b, bias=1):
    if not 0 < a <= b:
        return 0
    r = rng.random()

    f = lambda r, bias: (1-bias)*r**2 + bias*r

    r = f(r, bias)
    r = (1-r)*np.log(a) + r*np.log(b+0.999)
    return int(np.exp(r))


def gen_rel_pose(views, norm=False):
    assert len(views) == 2
    cam1_to_w, cam2_to_w = [view['camera_pose'] for view in views]
    w_to_cam1 = np.linalg.inv(cam1_to_w)

    cam2_to_cam1 = w_to_cam1 @ cam2_to_w

    if norm: # normalize
        T = cam2_to_cam1[:3,3]
        T /= max(1e-5, np.linalg.norm(T))

    return cam2_to_cam1.astype(np.float32)


def gen_c2p(norm = True):
    cam2proj = torch.eye(4, dtype=torch.float32)
    cam2proj[:3, 3] = torch.tensor([0.1, 0.0, 0.0])
    if norm:
        T = cam2proj[:3,3]
        T /= max(1e-5, np.linalg.norm(T))
    return 
    
