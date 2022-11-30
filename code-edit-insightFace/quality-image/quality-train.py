import os
import math
from sklearn import neighbors
import pickle
import cv2
import numpy as np
import torch
import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1

from facenetPreditctFunction import SCRFD, process_image
from fuction_compute import take_image, load_net, load_model_onnx, process_onnx
from facenetPreditctFunction import SCRFD, alignment, process_image_package, xyz_coordinates


pathkk =  '/home/maicg/Documents/python-image-processing/code-edit-insightFace/clasification_FR/faiss_class/dataMe/dataDemo'

def train(train_dir):
    import glob
    X=[]
    y=[]
    #lap tung anh trong co so du lieu
    #load moder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = SCRFD(model_file='/home/maicg/Documents/python-image-processing/code-edit-insightFace/onnx/scrfd_2.5g_bnkps.onnx')
    detector.prepare(-1)

    # net = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    BACKBONE = load_model_onnx('./onnx/Resnet2F.onnx')
    QUALITY = load_model_onnx('./onnx/Quality.onnx')
    for image in os.listdir(train_dir):
        pathName = os.path.join(train_dir,image)

        img2 = cv2.imread(pathName)
        # img2 = change_brightness(img2, 1.0, 5)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        bboxes, kpss = process_image_package(img2, detector)
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1,y1,x2,y2,_ = bbox.astype(np.int)
            _,_,_,_,score = bbox.astype(np.float)

            crop_img = img2[y1:y2, x1:x2]
            if kpss is not None:
                    kps = kpss[i]
                    distance12, distance_nose1, distance_nose2, distance_center_eye_mouth, distance_nose_ceye, distance_nose_cmouth, distance_eye, distance_mouth, l_eye, r_eye = xyz_coordinates(kps)

                    rotate_img = alignment(crop_img, l_eye, r_eye)
                    rotate_img = cv2.resize(rotate_img, (112,112))
                    try:
                        _, emb = process_onnx(rotate_img, BACKBONE, QUALITY)
                    except:
                        continue

                    print(emb.shape)
                    # print(emb)
                    # print(type(emb))
                    # print(emb.shape)
                    emb = emb.cpu().detach().numpy()
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

