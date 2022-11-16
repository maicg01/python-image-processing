import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

net = InceptionResnetV1(pretrained='vggface2').eval().to(device)
def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('/home/maicg/Documents/python-image-processing/code-edit-insightFace/facenet_al/faceMTCNN')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

print(dataset.idx_to_class)

aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    print("dau vao MTCNN:", x_aligned.shape)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

print(len(aligned))
print(names)

print(type(aligned))
aligned = torch.stack(aligned).to(device)
print(type(aligned))
print(aligned.shape)
print(aligned)
embeddings = net(aligned).detach().cpu()

print(len(embeddings))