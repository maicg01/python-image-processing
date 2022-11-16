import cv2
import os
import time
import pickle
import torch 
from torchvision import transforms
import math
from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1
from facenetPreditctFunction import computeCosinMTCNN, computeEmbMTCNN





def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imageOriginSet = []
    labelOriginSet = []
    with open('X1.pkl', 'rb') as f:
        imageOriginSet = pickle.load(f)

    with open('y1.pkl', 'rb') as f:
        labelOriginSet = pickle.load(f)

    # cam_port=1
    # cap = cv2.VideoCapture(cam_port)
    # cap = cv2.VideoCapture('rtsp://ai_dev:123654789@@@192.168.15.10:554/Streaming/Channels/1601')
    cap = cv2.VideoCapture('/home/maicg/Documents/python-image-processing/video_gt.avi')
    
    path = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/clasification_FR/dataMTCNNFacenet/dataDemo'
    path_dir = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/clasification_FR/dataMTCNNFacenet/dataTest'
    k=0
    if cap.isOpened():
        while True:
            for i in range(4):
                result, img = cap.read()
            # plt.imshow(img[:,:,::-1])
            # plt.show()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mtcnn = MTCNN(
                image_size=160, margin=0, min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                device=device
            )
            img_detect, remember = mtcnn(img, return_prob=True)
            avr_time = 0
            
            predict=[]
            label = []
            if img_detect is not None:
                count = 0
                time_start = time.time()
                for embOrigin in imageOriginSet:
                    # print('done')
                    result = computeCosinMTCNN(embOrigin, img)
                    print("===============", result)
                    predict.append(result.item())
                    label.append(labelOriginSet[count])
                    count = count + 1
                print('ket qua cuoi cung', max(predict))
                if max(predict) >= 0.65:
                    # print("vi tri anr: ", label[predict.index(max(predict))])
                    directory = label[predict.index(max(predict))]
                    print(directory)
                    try:
                        dir_fold = os.path.join(path_dir, directory)
                        os.makedirs(dir_fold, exist_ok = True)
                        frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(max(predict), 2))  + '.jpg'
                        print(frame_img_path)
                        cv2.imwrite(frame_img_path, img)
                        print("Directory created successfully")
                        k=k+1
                    except OSError as error:
                        print("Directory can not be created")
                else:
                    print("unknow")
                    try:
                        dir_fold = os.path.join(path_dir, 'unknow')
                        os.makedirs(dir_fold, exist_ok = True)
                        frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(max(predict), 2))  + '.jpg'
                        print(frame_img_path)
                        cv2.imwrite(frame_img_path, img)
                        k=k+1
                    except OSError as error:
                        print("Directory can not be created")
                
                
                time_end = time.time()
                avr_time = round(((time_end-time_start)/count), 2)
            
                print(avr_time)
                # print('Doneeeee')
        cap.release()
    cv2.destroyAllWindows()

main()