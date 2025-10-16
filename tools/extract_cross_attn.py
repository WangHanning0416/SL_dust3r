import torch
import numpy as np
import os

def extract_cross_parts(pth_file_path, output_dir='cross_attn_npy'):
    """
    提取pth文件中key包含"cross"的部分，并保存为npy文件
    
    参数:
        pth_file_path: pth文件的路径
        output_dir: 保存npy文件的目录，默认为'cross_parts'
    """
    # 检查文件是否存在
    if not os.path.exists(pth_file_path):
        print(f"错误: 文件 {pth_file_path} 不存在")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载pth文件
    print(f"正在加载pth文件: {pth_file_path}")
    data = torch.load(pth_file_path, map_location=torch.device('cpu'))

    if not isinstance(data, dict):
        print("错误: 加载的pth文件内容不是字典类型")
        return
    
    # 统计符合条件的key数量
    count = 0
    
    # 遍历所有key，查找包含"cross"的
    for key in data["model"]:
        if 'cross' in key.lower():  # 不区分大小写
            count += 1
            if isinstance(data["model"][key], torch.Tensor):
                np_array = data["model"][key].cpu().numpy()
            else:
                np_array = np.array(data[key])

            safe_key = key.replace('/', '_').replace('\\', '_').replace(':', '_')
            npy_file_name = f"{safe_key}.npy"
            npy_file_path = os.path.join(output_dir, npy_file_name)
            
            # 保存为npy文件
            np.save(npy_file_path, np_array)
            print(f"已保存: {npy_file_path}")
    
    if count == 0:
        print("未找到包含'cross'的key")
    else:
        print(f"处理完成，共提取 {count} 个包含'cross'的部分，保存至 {output_dir} 目录")


if __name__ == "__main__":
    pth_file = '/data3/hanning/dust3r1/checkpoints/dust3r_SL_224/checkpoint-best.pth'
    extract_cross_parts(pth_file)
