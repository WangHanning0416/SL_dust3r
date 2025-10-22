import numpy as np
from scipy.spatial import KDTree
from os.path import join


def colored_pred_ply(pred1, gt1, valid1, pred2, gt2, valid2,output_ply_path):
    """
    生成带颜色的预测点云PLY文件，颜色亮度随与GT点云的距离增加而提高
    :param pred_npy_path: 预测点云npy文件路径 (形状可为(H,W,3)或(N,3))
    :param gt_npy_path: GT点云npy文件路径 (形状可为(H,W,3)或(M,3))
    :param output_ply_path: 输出PLY文件路径
    :param color_mode: 颜色模式: 'gray'(灰度, 越远越白) / 'red'(红色, 越远越红)
    """
    pred_flat1 = pred1.reshape(-1, 3)  # 预测点: (N, 3)
    gt_flat1 = gt1.reshape(-1, 3)      # GT点: (M, 3)
    valid_flat1 = valid1.reshape(-1)   # 有效掩码: (N,)
    pred1 = pred_flat1[valid_flat1]  
    gt1 = gt_flat1[valid_flat1] 

    pred_flat2 = pred2.reshape(-1, 3)  # 预测点: (N, 3)
    gt_flat2 = gt2.reshape(-1, 3)      # GT点: (M, 3)
    valid_flat2 = valid2.reshape(-1)   # 有效掩码:
    pred2 = pred_flat2[valid_flat2]
    gt2 = gt_flat2[valid_flat2]
    
    distances1 = np.linalg.norm(pred1 - gt1, axis=1)
    distances2 = np.linalg.norm(pred2 - gt2, axis=1)

    dist_max = max(distances1.max(), distances2.max())

    normalized_dist1 = distances1 / dist_max
    brightness1 = (normalized_dist1 * 255).astype(np.uint8) 
    normalized_dist2 = distances2 / dist_max
    brightness2 = (normalized_dist2 * 255).astype(np.uint8)


    colors1 = np.stack([brightness1, brightness1, brightness1], axis=1)
    colors2 = np.stack([brightness2, brightness2, brightness2], axis=1)

    total_points1 = len(pred1)
    out_path1 = join(output_ply_path, "pred1_colored.ply")
    out_path2 = join(output_ply_path, "pred2_colored.ply")
    with open(out_path1, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {total_points1}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in zip(pred1, colors1):
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n')
    total_points2 = len(pred2)
    with open(out_path2, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {total_points2}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in zip(pred2, colors2):
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n')

if __name__ == "__main__":
    npy_dir =  "/data3/hanning/dust3r/result_rgb/room0/npy"
    pred_pcd1 = np.load(join(npy_dir, "predicted_pts3d.npy")).astype(np.float32)
    gt_pcd1 = np.load(join(npy_dir, "gt_pts3d.npy")).astype(np.float32)
    valid1 = np.load(join(npy_dir, "gt_valid_mask.npy")).astype(bool)
    
    npy_dir =  "/data3/hanning/dust3r/result_kinectsp/room0/npy"
    pred_pcd2 = np.load(join(npy_dir, "predicted_pts3d.npy")).astype(np.float32)
    gt_pcd2 = np.load(join(npy_dir, "gt_pts3d.npy")).astype(np.float32)
    valid2 = np.load(join(npy_dir, "gt_valid_mask.npy")).astype(bool)

    colored_pred_ply(
        pred1=pred_pcd1[0],
        gt1=gt_pcd1[0],
        valid1=valid1[0],
        pred2=pred_pcd2[0],
        gt2=gt_pcd2[0],
        valid2=valid2[0],
        output_ply_path="/data3/hanning/dust3r/vis/",
    )