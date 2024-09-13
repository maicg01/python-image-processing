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
# model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})

# # inference old
# model.load_state_dict(torch.load(f'/home/lab-02/Documents/maicg/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth', map_location='cpu'))
# model.to(DEVICE).eval()

# # inference train new v1
# state_dict = torch.load(f'/home/lab-02/Documents/maicg/Depth-Anything-V2/metric_depth/exp/ddos_small_10_epoch_bs_11/latest.pth', map_location='cpu')
# my_state_dict = {} 
# for key in state_dict['model'].keys(): 
#     my_state_dict[key.replace('module.', '')] = state_dict['model'][key]
# model.load_state_dict(my_state_dict)
# model.to(DEVICE).eval()

#pruning load
# model = torch.load('/home/lab-02/Documents/maicg/Torch-Pruning/weights_prune/prune_depth_50_struct.pth')
# model = torch.load('/mnt/data/shared/maicg/lab_01/Depth-Anything-V2/exp/ddos_finetune_50_struct_v1/latest.pth')
# model = torch.load('/mnt/data/shared/maicg/lab_01/Depth-Anything-V2/exp/prune_depth_30_struct_prune_attention/latest.pth')
# model = torch.load('/mnt/data/shared/maicg/lab_01/Depth-Anything-V2/exp/prune_depth_50_struct_prune_attention/latest.pth')
model = torch.load('/home/lab-02/Documents/maicg/Torch-Pruning/weights_prune/prune_depth_70_struct_prune_attention.pth')
model.to(DEVICE).eval()

# input = torch.randn(1, 3, 518, 518).to(DEVICE)

# print_layer_flops(model, input)

# filename = '/home/lab-02/Documents/maicg/depth_anything/data_yolo_format/data_demo/images/image_1410.jpg'
# filename = '/home/lab-02/Documents/maicg/depth_anything/data_yolo_format/data_demo/images/image_2790.jpg'
# filename = '/home/lab-02/Documents/maicg/Depth-Anything-V2/assets/examples/demo01.jpg'
# raw_img = cv2.imread(filename)
# depth = model.infer_image(raw_img) # HxW depth map in meters in numpy
# print("depth meters: ", depth)


# depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
# depth = depth.astype(np.uint8)
# depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
# print("depth numpy: ", depth)



# h, w = raw_img.shape[:2]

# split_region = np.ones((h, 50, 3), dtype=np.uint8) * 255
# combined_frame = cv2.hconcat([raw_img, split_region, depth])
# combined_frame = cv2.resize(combined_frame, (1920,1080))

# # cv2.imwrite(os.path.join("/home/lab-02/Documents/maicg/Depth-Anything-V2/metric_depth/infer_data", os.path.splitext(os.path.basename(filename))[0] + '_ddos_v1.png'), combined_frame)
# # cv2.imshow("raw", raw_img)
# cv2.imshow("depth_anything_v2_metric", combined_frame)
# cv2.waitKey(3000)


import torch
import torch.autograd.profiler as profiler

# # Giả sử model và data đã được định nghĩa
# with profiler.profile(record_shapes=True) as prof:
#     with profiler.record_function("model_inference"):
#         output = model.infer_image(raw_img)

# # Xem kết quả profiling
# print(prof.key_averages().table(sort_by="cpu_time_total"))

# from torchinfo import summary

# # Kiểm tra tóm tắt của mô hình
# summary(model, input_size=(1, 3, 518, 518))

# from ptflops import get_model_complexity_info

# flops, params = get_model_complexity_info(model, (3, 518, 518), as_strings=True, print_per_layer_stat=True)
# print(f"FLOPs: {flops}, Params: {params}")

# import torch
# import torchvision.models as models

# model = models.densenet121(pretrained=True)
x = torch.randn((1, 3, 518, 518), requires_grad=True).to('cuda')
# for _ in range(50):
#     model(x)
# with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     model(x)
# print(prof) 
for _ in range(50):
     model(x)

# Sử dụng profiler
with profiler.profile(use_cuda=True) as prof:
    with profiler.record_function("model_inference"):
        output = model(x)

# In kết quả profiler
print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"))

def estimate_latency(model, example_inputs, repetitions=50):
    import numpy as np
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings=np.zeros((repetitions,1))

    for _ in range(5):
        _ = model(example_inputs)

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(example_inputs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn

example_input = torch.rand(1, 3, 518, 518).to('cuda:0')

latency_mu, latency_std = estimate_latency(model, example_input)
macs, params = tp.utils.count_ops_and_params(model, example_input)

print(f"\tMACs: {macs/1e9:.2f} G, \tParams: {params/1e6:.2f} M, \tLatency: {latency_mu:.2f} ms +- {latency_std:.2f} ms")
