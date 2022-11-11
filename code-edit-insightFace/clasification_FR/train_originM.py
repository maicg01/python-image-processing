import os
import math
from sklearn import neighbors
import pickle
import cv2

from facenet_preditctM import process_image, computeEmb

pathkk =  '/home/maicg/Documents/python-image-processing/code-edit-insightFace/dataExper/dataDemo'

def train(train_dir, path_save_model):
    X=[]
    y=[]
    #lap tung anh trong co so du lieu
    for image in os.listdir(train_dir):
        pathName = [os.path.join(train_dir,image)]
        for pathtest in pathName:
            img2 = cv2.imread(pathtest)
            img_origin, remember1 = process_image(img2)
            emb = computeEmb(img_origin)
            # print('done')
            path_img = pathtest[-16:-4]
            # print(path_img)
            index_label = path_img.find("/")
            directory = path_img[index_label+1:]
            print(directory)

            X.append(emb)
            y.append(directory)
            # count = count + 1 
    print(X)
    print(y)  

train(train_dir=pathkk)

