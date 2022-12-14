import os
import math
from sklearn import neighbors
import pickle
import cv2
import numpy as np
import torch
import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1

from facenetPreditctFunction import SCRFD, process_image, computeEmb, fixed_image_standardization, change_brightness

pathkk =  '/home/maicg/Documents/python-image-processing/code-edit-insightFace/clasification_FR/faiss_class/dataMe/dataDemo'

def train(train_dir):
    X=[]
    y=[]
    #lap tung anh trong co so du lieu
    #load moder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = SCRFD(model_file='/home/maicg/Documents/python-image-processing/code-edit-insightFace/onnx/scrfd_2.5g_bnkps.onnx')
    detector.prepare(-1)

    net = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    for image in os.listdir(train_dir):
        pathName = os.path.join(train_dir,image)

        img2 = cv2.imread(pathName)
        # img2 = change_brightness(img2, 1.0, 5)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img_origin, remember1 = process_image(img2, detector)
        # img_origin = cv2.dnn.blobFromImage(img_origin, size=(96,96), scalefactor=1, mean=(0, 0, 0), swapRB=False, crop=False)
        emb = computeEmb(img_origin, net)
        # print(emb)
        # print(type(emb))
        # print(emb.shape)
        emb = np.array(emb,dtype=np.float32).reshape(512,)
        print(type(emb))
        print(emb.shape)
        # print(path_img)
        index_label = image.find(".")
        directory = image[:index_label]
        print(directory)

        X.append(emb)
        y.append(directory)
        # print(type(X))
        # count = count + 1 
    # print(X)
    print(y)  
    return X, y

# with open('x', 'rb'):
#     pickle.dump()
X, y = train(train_dir=pathkk)
faceEncode = np.array(X,dtype=np.float32)
print(faceEncode.shape)


with open('X.pkl', 'wb') as f:
    pickle.dump(X, f)  

with open('y.pkl', 'wb') as f:
    pickle.dump(y, f) 

