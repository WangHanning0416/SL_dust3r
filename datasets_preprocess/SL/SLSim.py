import torch
import torch.nn.functional as F
import numpy as np
import cv2

def depth_to_point_cloud_batch(depth, intrinsics):
    # depth: (B,H,W), intrinsics: (B,3,3)
    B,H,W = depth.shape
    device = depth.device

    y, x = torch.meshgrid(torch.arange(H, device=device),
                          torch.arange(W, device=device), indexing='ij')
    x = x.unsqueeze(0).expand(B,-1,-1)
    y = y.unsqueeze(0).expand(B,-1,-1)

    Z = depth
    fx = intrinsics[:,0,0].unsqueeze(-1).unsqueeze(-1)
    fy = intrinsics[:,1,1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[:,0,2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[:,1,2].unsqueeze(-1).unsqueeze(-1)

    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    return torch.stack([X,Y,Z], dim=-1)  # (B,H,W,3)

def transform_point_cloud_batch(point_cloud, cam2proj):
    # point_cloud: (B,H,W,3), cam2proj: (B,4,4)
    B,H,W,_ = point_cloud.shape
    pc_flat = point_cloud.view(B,-1,3)
    ones = torch.ones(B, pc_flat.shape[1],1, device=point_cloud.device)
    pc_hom = torch.cat([pc_flat, ones], dim=-1)  # (B,N,4)
    proj_points_hom = torch.bmm(pc_hom, cam2proj.transpose(1,2))  # (B,N,4)
    proj_points = proj_points_hom[...,:3] / proj_points_hom[...,3:4]
    return proj_points.view(B,H,W,3)

def project_to_pattern_plane_batch(proj_points, K_proj):
    # proj_points: (B,H,W,3), K_proj: (B,3,3)
    B,H,W,_ = proj_points.shape
    X = proj_points[...,0]
    Y = proj_points[...,1]
    Z = proj_points[...,2].clamp(min=1e-6)

    fx = K_proj[:,0,0].unsqueeze(-1).unsqueeze(-1)
    fy = K_proj[:,1,1].unsqueeze(-1).unsqueeze(-1)
    cx = K_proj[:,0,2].unsqueeze(-1).unsqueeze(-1)
    cy = K_proj[:,1,2].unsqueeze(-1).unsqueeze(-1)

    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return u,v

def sample_pattern_batch(u, v, pattern):
    B,H,W = u.shape
    device = u.device
    sampled = torch.zeros(B,H,W,3, dtype=torch.uint8, device=device)

    for b in range(B):
        u_b = torch.remainder(torch.round(u[b]).long(), pattern.shape[1])
        v_b = torch.remainder(torch.round(v[b]).long(), pattern.shape[0])
        for c in range(3):
            sampled[b,:,:,c] = pattern[v_b,u_b,c].to(device)
    return sampled

def apply_depth_attenuation_batch(image, depth, Z0=0.8, gamma=1.0, min_val=0.2):
    attenuation = (Z0 / depth.clamp(0.1,5.0))**gamma
    attenuation = attenuation.clamp(min_val,1.0).unsqueeze(-1)
    out = (image.float() * attenuation).clamp(0,255).byte()
    return out

def apply_depth_blur_batch(image, depth, focus=None, strength=3):
    B,H,W,_ = image.shape
    out = torch.zeros_like(image)
    for b in range(B):
        img_np = image[b].cpu().numpy()
        depth_np = depth[b].cpu().numpy()
        norm_depth = (depth_np - depth_np.min())/(depth_np.max()-depth_np.min()+1e-6)
        blur_img = cv2.GaussianBlur(img_np, (5,5), strength)
        if focus is None:
            f = depth_np.min()
        else:
            f = focus
        alpha = np.clip((norm_depth - f)*5,0,1)[...,None]
        out_np = (img_np*(1-alpha)+blur_img*alpha).astype(np.uint8)
        out[b] = torch.from_numpy(out_np).to(image.device)
    return out

def SLSim_batch(rgb, depth, pattern, K_cam, K_proj, cam2proj):
    # rgb: (B,H,W,3), depth: (B,H,W), K_cam: (B,3,3), K_proj: (B,3,3), cam2proj: (B,4,4)
    B,H,W,_ = rgb.shape
    depth = depth.clone().float()/1000.0
    point_cloud = depth_to_point_cloud_batch(depth, K_cam)
    proj_points = transform_point_cloud_batch(point_cloud, cam2proj)
    u,v = project_to_pattern_plane_batch(proj_points, K_proj)
    projected = sample_pattern_batch(u,v,pattern)
    projected = apply_depth_attenuation_batch(projected, proj_points[...,2])
    projected = apply_depth_blur_batch(projected, proj_points[...,2])
    rgb_f = rgb.float()
    proj_f = projected.float()
    alpha = 0.4
    gray = proj_f.mean(dim=-1, keepdim=True)/255.0
    w1 = gray + (1-gray)*(1-alpha)
    w2 = (1-gray)*alpha
    overlay = (rgb_f*w1 + proj_f*w2).clamp(0,255).byte()
    return overlay
