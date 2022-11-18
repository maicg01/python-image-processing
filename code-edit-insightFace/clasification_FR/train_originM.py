import os
import math
from sklearn import neighbors
import pickle
import cv2

from facenetPreditctFunction import process_image, computeEmb, fixed_image_standardization

pathkk =  '/home/maicg/Documents/python-image-processing/code-edit-insightFace/dataExper/dataDemo'

def train(train_dir):
    X=[]
    y=[]
    #lap tung anh trong co so du lieu
    for image in os.listdir(train_dir):
        pathName = os.path.join(train_dir,image)

        img2 = cv2.imread(pathName)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img_origin, remember1 = process_image(img2)
        emb = computeEmb(img_origin)
        print(emb)
        # print(path_img)
        index_label = image.find(".")
        directory = image[:index_label]
        print(directory)

        X.append(emb)
        y.append(directory)
        # count = count + 1 
    print(X)
    print(y)  
    return X, y

# with open('x', 'rb'):
#     pickle.dump()
X, y = train(train_dir=pathkk)

with open('X.pkl', 'wb') as f:
    pickle.dump(X, f)  

with open('y.pkl', 'wb') as f:
    pickle.dump(y, f) 

