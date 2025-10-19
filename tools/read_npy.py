import numpy as np

npy_path = "/data3/hanning/dust3r/cross_attn_npy/layer_0/img1_to_img2_attn.npy"

attn = np.load(npy_path)
print(f"Loaded attention shape: {attn.shape}")