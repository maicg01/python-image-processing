import cv2
import os
import time
import pickle
import numpy as np
import faiss
import torch
import torchvision
import shutil
from facenet_pytorch import MTCNN, InceptionResnetV1

from facenetPreditctFunction import SCRFD, process_image, computeEmb
from fuction_compute import computeCosinQuality, load_net, take_image


def main():
    #load moder
    detector = SCRFD(model_file='/home/maicg/Documents/python-image-processing/code-edit-insightFace/onnx/scrfd_2.5g_bnkps.onnx')
    detector.prepare(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    BACKBONE, QUALITY, DEVICE = load_net()

    faceEncode = []
    labelOriginSet = []
    with open('X.pkl', 'rb') as f:
        faceEncode = pickle.load(f)

    with open('y.pkl', 'rb') as f:
        labelOriginSet = pickle.load(f)
    
    faceEncode = np.array(faceEncode,dtype=np.float32)
    # create index with faiss
    face_index = faiss.IndexFlatIP(512)
    # add vector
    face_index.add(faceEncode)

    # cam_port=1
    # cap = cv2.VideoCapture(cam_port)
    # cap = cv2.VideoCapture('rtsp://ai_dev:123654789@@@192.168.15.10:554/Streaming/Channels/1601')
    cap = cv2.VideoCapture('/home/maicg/Documents/python-image-processing/video_gt.avi')
    
    path = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/quality-image/dataMe/dataDemo'
    path_dir = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/quality-image/dataMe/dataTest'
    k=0
    if cap.isOpened():
        while True:
            for i in range(4):
                result, img = cap.read()
            # plt.imshow(img[:,:,::-1])
            # plt.show()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            avr_time = 0
            img_detect, remember = process_image(img,detector=detector)
            # img_detect = change_brightness(img_detect, 1.0, 10)
            # img_detect = cv2.dnn.blobFromImage(img_detect, scaleFactor=1/255.0, mean=(0, 0, 0))
            
            if img_detect is not None:
                time_start = time.time()
                try:
                    quality, emb = take_image(BACKBONE, QUALITY, DEVICE, img_detect)
                except:
                    continue
                print(emb.shape)
                emb = emb.cpu().detach().numpy()
                emb = np.array(emb,dtype=np.float32)
                w, result = face_index.search(emb, k=1)
                label = [labelOriginSet[i] for i in result[0]]

                if quality[0] < 0.2:
                    output_path = 'quality_result_bad'
                    dir_fold = os.path.join(path_dir, output_path)
                    os.makedirs(dir_fold, exist_ok = True)

                    frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(quality[0], 4))  + '.jpg'
                    cv2.imwrite(frame_img_path, img_detect)
                else:
                    output_path = 'quality_result_good'
                    dir_fold = os.path.join(path_dir, output_path)
                    os.makedirs(dir_fold, exist_ok = True)

                    frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(quality[0], 4))  + '.jpg'
                    cv2.imwrite(frame_img_path, img_detect)

                if w[0][0] >= 0.35:
                    directory = label[0]
                    print(directory)
                    try:
                        dir_fold = os.path.join(path_dir, directory)
                        os.makedirs(dir_fold, exist_ok = True)
                        frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(w[0][0], 2))  + '.jpg'
                        print(frame_img_path)
                        # img_save = cv2.resize(img_detect, (160,160))
                        cv2.imwrite(frame_img_path, img_detect)
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
                        print(frame_img_path)
                        # img_save = cv2.resize(img_detect, (160,160))
                        cv2.imwrite(frame_img_path, img_detect)
                        k=k+1
                    except OSError as error:
                        print("Directory can not be created")
                
                
                time_end = time.time()
                avr_time = round(((time_end-time_start)), 2)
            
                print(avr_time)
                # print('Doneeeee')
        cap.release()
    cv2.destroyAllWindows()

main()