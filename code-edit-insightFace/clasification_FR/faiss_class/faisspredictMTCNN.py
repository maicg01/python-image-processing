import cv2
import os
import numpy as np
import time
import pickle
import torch 
from torchvision import transforms
import math
from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1
from facenetPreditctFunction import computeCosinMTCNN, computeEmbMTCNN
import faiss




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    faceEncode = []
    labelOriginSet = []
    with open('X1.pkl', 'rb') as f:
        faceEncode = pickle.load(f)

    with open('y1.pkl', 'rb') as f:
        labelOriginSet = pickle.load(f)
    
    faceEncode = np.array(faceEncode,dtype=np.float32)
    # create index with faiss
    face_index = faiss.IndexFlatL2(512)
    # add vector
    face_index.add(faceEncode)

    # cam_port=1
    # cap = cv2.VideoCapture(cam_port)
    # cap = cv2.VideoCapture('rtsp://ai_dev:123654789@@@192.168.15.10:554/Streaming/Channels/1601')
    cap = cv2.VideoCapture('/home/maicg/Documents/python-image-processing/video_gt.avi')
    
    path = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/clasification_FR/faiss_class/dataMTCNN/dataDemo'
    path_dir = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/clasification_FR/faiss_class/dataMTCNN/dataTest'
    k=0
    if cap.isOpened():
        while True:
            for i in range(4):
                result, img = cap.read()
            # plt.imshow(img[:,:,::-1])
            # plt.show()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mtcnn = MTCNN(
                image_size=160, margin=0, min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                device=device
            )
            path_detect = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/clasification_FR/faiss_class/dataMTCNN/dataTest/all/frame' + str(k) + '.jpg'
            img_detect = mtcnn(img, save_path=path_detect)
            avr_time = 0
            
            
            if img_detect is not None:
                time_start = time.time()
                result = computeEmbMTCNN(img)
                emb = np.array(result,dtype=np.float32)
                w,result = face_index.search(emb, k=1)
                label = [labelOriginSet[i] for i in result[0]]

                if w[0][0] <= 1:
                    directory = label[0]
                    print(directory)
                    try:
                        dir_fold = os.path.join(path_dir, directory)
                        os.makedirs(dir_fold, exist_ok = True)
                        frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(w[0][0], 2))  + '.jpg'
                        img_read = cv2.imread(path_detect, cv2.COLOR_BGR2RGB)
                        # print(frame_img_path)
                        cv2.imwrite(frame_img_path, img_read)
                        
                        print("Directory created successfully")
                        k=k+1
                    except OSError as error:
                        print("Directory can not be created")
                else:
                    print("unknow")
                    try:
                        dir_fold = os.path.join(path_dir, 'unknow')
                        os.makedirs(dir_fold, exist_ok = True)
                        frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(w[0][0], 2))  + '.jpg'
                        img_read = cv2.imread(path_detect, cv2.COLOR_BGR2RGB)
                        # print(frame_img_path)
                        cv2.imwrite(frame_img_path, img_read)
                        k=k+1
                    except OSError as error:
                        print("Directory can not be created")
                
                
                time_end = time.time()
                avr_time = round(((time_end-time_start)), 2)
            
                print(avr_time)
                print('Doneeeee')
        cap.release()
    cv2.destroyAllWindows()

main()