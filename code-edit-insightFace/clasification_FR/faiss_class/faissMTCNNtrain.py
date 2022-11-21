import os
import math
from sklearn import neighbors
import pickle
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from facenetPreditctFunction import computeEmbMTCNN

pathkk =  '/home/maicg/Documents/python-image-processing/code-edit-insightFace/dataExper/dataDemo'

def train(train_dir):
    X=[]
    y=[]
    #lap tung anh trong co so du lieu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
    )

    net = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    for image in os.listdir(train_dir):
        pathName = os.path.join(train_dir,image)

        img2 = cv2.imread(pathName)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        emb = computeEmbMTCNN(img2, mtcnn, net)
        print(emb)
        # print(path_img)
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

with open('X1.pkl', 'wb') as f:
    pickle.dump(X, f)  

with open('y1.pkl', 'wb') as f:
    pickle.dump(y, f) 

