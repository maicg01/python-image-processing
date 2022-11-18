import os
import math
from sklearn import neighbors
import pickle
import cv2
import numpy as np

from facenetPreditctFunction import computeEmbMTCNN

pathkk =  '/home/maicg/Documents/python-image-processing/code-edit-insightFace/dataExper/dataDemo'

def train(train_dir):
    X=[]
    y=[]
    #lap tung anh trong co so du lieu
    for image in os.listdir(train_dir):
        pathName = os.path.join(train_dir,image)

        img2 = cv2.imread(pathName)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        emb = computeEmbMTCNN(img2)
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

