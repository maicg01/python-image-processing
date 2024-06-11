import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import math
from torch import nn
# from depth_anything.dpt import DepthAnything
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix #hai format de luu tru ma tran thu nham tinh toan duoc de dang do tiet kiem bo nho.

def apply_weight_sharing(model, bits=6): # co toi da 2^5 = 32 cum cua kmeans
    for module in model.parameters():
        dev = module.device
        weight = module.data.cpu().numpy()
        shape_orin = weight.shape
        if(len(shape_orin) == 3):
            print("shape_orin: ", shape_orin)
            weight = weight[0]
            shape = weight.shape
            print("shape: ", shape)
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
            # print("mat: ", mat)
            min_ = min(mat.data)
            max_ = max(mat.data)
            space = np.linspace(min_, max_, num=2**bits)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, algorithm="elkan")
            kmeans.fit(mat.data.reshape(-1,1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1) # share lại centroid vào các vị trí weight
            mat.data = new_weight
            module.data = torch.from_numpy(mat.toarray().reshape(1, shape_orin[1], shape_orin[2])).to(dev)
        elif (len(shape_orin) == 2):
            shape = weight.shape
            print("shape: ", shape)
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
            # print("mat: ", mat)
            try: 
                min_ = min(mat.data)
                max_ = max(mat.data)
                space = np.linspace(min_, max_, num=2**bits)
                kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, algorithm="lloyd")
                kmeans.fit(mat.data.reshape(-1,1))
                new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1) # share lại centroid vào các vị trí weight
                mat.data = new_weight
                module.data = torch.from_numpy(mat.toarray()).to(dev)
                print("done else")
            except Exception as e:
                print(e)

    return model


encoder = 'vits'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE)

from depth_anything.dpt import DPT_DINOv2
depth_anything = DPT_DINOv2('vits', features=64, out_channels=[48, 96, 192, 384])
ckpt = torch.load('/home/lab-02/Documents/maicg/depth_anything/checkpoints/depth_anything_vits14.pth')  
depth_anything.load_state_dict(ckpt)
model = depth_anything.to(device=DEVICE)

def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')


# print_nonzeros(model)

# Customize of prune function with mask
# def prune(module, threshold): #tinh toan lai cac trong so co weight nho hon nguong quy dinh, cap nhat lai mask va weight tai cac vi tri do ve gia tri 0
#     weight_dev = module.weight.device
#     mask_module =  Parameter(torch.ones([module.out_features, module.in_features]), requires_grad=False)
#     mask_dev = mask_module.device 
#     # Convert Tensors to numpy and calculate
#     tensor = module.weight.data.cpu().numpy()
#     mask = mask_module.data.cpu().numpy()
#     new_mask = np.where(abs(tensor) < threshold, 0, mask)
#     # Apply new weight and mask
#     module.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
#     mask_module = torch.from_numpy(new_mask).to(mask_dev)

def prune(parameter, threshold): #tinh toan lai cac trong so co weight nho hon nguong quy dinh, cap nhat lai mask va weight tai cac vi tri do ve gia tri 0
    device_param = parameter.device
    
    mask_module = torch.ones_like(parameter)
    mask_dev = mask_module.device 
    mask = mask_module.data.cpu().numpy()

    tensor = parameter.data.cpu().numpy()
    new_mask = np.where(abs(tensor) < threshold, 0, mask)
    # print("mask sparity: ", new_mask)

    parameter.data = torch.from_numpy(tensor * new_mask).to(device_param)
    mask_module = torch.from_numpy(new_mask).to(mask_dev)

def prune_by_std(module, n, s=0.5): # tuy chinh tham so s=0.25 de tinh toan gia tri cua threshold can cat tia, 25% của average standard deviation: gia tri do lech chuan trung binh
    # Note that module here is the layer
    # ex) fc1, fc2, fc3
    threshold = np.std(module.data.cpu().numpy()) * s
    if (threshold < 0.05):
        print(f'Pruning with threshold : {threshold} for layer {n}')
        prune(module, threshold)

if __name__=="__main__":

    for n, parameter in model.named_parameters():
        prune_by_std(parameter, n)
    print_nonzeros(model)

    torch.save(model.state_dict(), 'prune.pth')

    model = apply_weight_sharing(model)
    torch.save(model.state_dict(), 'prune_quanti.pth')
