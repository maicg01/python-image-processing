import cv2
import torch
import numpy as np
import os
import torch_pruning as tp

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from thop import profile, clever_format


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vits' # or 'vits', 'vitb'
dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 100 # 20 for indoor model, 80 for outdoor model

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})

# # inference old
# model.load_state_dict(torch.load(f'/home/lab-02/Documents/maicg/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth', map_location='cpu'))
# model.to(DEVICE).eval()

# inference train new v1
# state_dict = torch.load(f'/home/lab-02/Documents/maicg/Depth-Anything-V2/metric_depth/exp/ddos_small_10_epoch_bs_11/latest.pth', map_location='cpu')
state_dict = torch.load(f'/mnt/data/shared/maicg/lab_01/Depth-Anything-V2/metric_depth/exp/airsim_v2_10_epoch_50000_more/latest.pth', map_location='cpu')
# state_dict = torch.load(f'/mnt/data/shared/maicg/lab_01/Depth-Anything-V2/metric_depth/exp/airsim_v2_10_epoch/latest.pth', map_location='cpu')

my_state_dict = {} 
for key in state_dict['model'].keys(): 
    my_state_dict[key.replace('module.', '')] = state_dict['model'][key]
model.load_state_dict(my_state_dict)
model.to(DEVICE).eval()


# print_layer_flops(model, input)

# filename = '/home/lab-02/Documents/maicg/depth_anything/data_yolo_format/data_demo/images/image_1410.jpg'
# filename = '/home/lab-02/Documents/maicg/depth_anything/data_yolo_format/data_demo/images/image_2790.jpg'
# filename = '/home/lab-02/Documents/maicg/Depth-Anything-V2/assets/examples/demo01.jpg'
filename = '/mnt/data/shared/airsim_image1/raw/13660.png'
# filename = '/mnt/data/shared/airsim_image1/raw/14660.png'
# filename = '/mnt/data/shared/airsim_image1/raw/15660.png'
# filename = '/mnt/data/shared/airsim_image1/raw/16660.png'
# filename = '/mnt/data/shared/airsim_image1/raw/17660.png'
# filename = '/mnt/data/shared/DDOS/data/test/park/0/depth/0.png'
raw_img = cv2.imread(filename)
depth = model.infer_image(raw_img) # HxW depth map in meters in numpy
# depth = depth*4
print("depth meters: ", depth)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import cv2

# plt.title("INFER DEPTH")
# plt.imshow(depth, cmap=cm.viridis)  # Sử dụng colormap 'viridis'
# plt.colorbar()  # Hiển thị thang màu
# # plt.show()

# depth_path = '/mnt/data/shared/airsim_image1/depth/13660.png'
# depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  
# data = depth * (100.0 / 65535.0) # convert in meters
# plt.title("Goc")
# plt.imshow(data, cmap=cm.viridis)  # Sử dụng colormap 'viridis'
# plt.colorbar()  # Hiển thị thang màu
# plt.show()

fig, ax = plt.subplots(1, 2, figsize=(20, 4))

color_depth = ax[0].imshow(depth, cmap=cm.viridis)
plt.colorbar(color_depth, ax=ax[0], label='Depth')
ax[0].axis('off')  # Tắt các trục

#goc

depth_path = '/mnt/data/shared/airsim_image1/depth/13660.png'
# depth_path = '/mnt/data/shared/airsim_image1/depth/14660.png'
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  
data = depth * (100.0 / 65535.0) # convert in meters

color_goc = ax[1].imshow(data, cmap=cm.viridis)  # Sử dụng colormap 'viridis'
plt.colorbar(color_goc, ax=ax[1], label='Orin')
ax[1].axis('off')

plt.show()

