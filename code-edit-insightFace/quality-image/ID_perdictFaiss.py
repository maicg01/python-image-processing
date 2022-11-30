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

from facenetPreditctFunction import SCRFD, alignment, process_image_package, xyz_coordinates
from fuction_compute import process_onnx, load_model_onnx, take_image


def main():
    import glob
    #load moder
    detector = SCRFD(model_file='/home/maicg/Documents/python-image-processing/code-edit-insightFace/onnx/scrfd_2.5g_bnkps.onnx')
    detector.prepare(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    BACKBONE = load_model_onnx('./onnx/Resnet2F.onnx')
    QUALITY = load_model_onnx('./onnx/Quality.onnx')

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
    cap = cv2.VideoCapture('/home/maicg/Documents/python-image-processing/video_59s.avi')
    
    path = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/quality-image/dataMe/dataDemo'
    path_dir = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/quality-image/test_quality/ID/test2'
    k=0
    name_id = len(labelOriginSet)
    if cap.isOpened():
        while True:
            for i in range(4):
                result, img = cap.read()
            # plt.imshow(img[:,:,::-1])
            # plt.show()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxes, kpss = process_image_package(img, detector)
            h, w, c = img.shape
            area_base = h*w
            tl = 0
            tl1 = 0
            for i in range(bboxes.shape[0]):
                time_start = time.time()
                bbox = bboxes[i]
                x1,y1,x2,y2,_ = bbox.astype(np.int)
                _,_,_,_,score = bbox.astype(np.float)

                crop_img = img[y1:y2, x1:x2]
                
                h1 = int(crop_img.shape[0])
                w1 = int(crop_img.shape[1])
                area_crop = h1*w1
                if kpss is not None:
                    kps = kpss[i]
                    distance12, distance_nose1, distance_nose2, distance_center_eye_mouth, distance_nose_ceye, distance_nose_cmouth, distance_eye, distance_mouth, l_eye, r_eye = xyz_coordinates(kps)
                    if (distance_nose1-distance_nose2) <= 0:
                        # print("=====================dt1,dt2",distance_nose1,distance_nose2)
                        tl = distance_nose1/distance_nose2
                    else: 
                        # print("else=====================dt1,dt2",distance_nose1,distance_nose2)
                        tl = distance_nose2/distance_nose1
                    
                    if (distance_nose_ceye - distance_nose_cmouth) <= 0:
                        tl1 = distance_nose_ceye/distance_nose_cmouth
                    else:
                        tl1 = distance_nose_cmouth/distance_nose_ceye

                    # print(tl)

                    if area_crop == 0:
                        break
                    elif (area_base/area_crop) > ((1080*1920)/(64*64)):
                        print("hinh nho")
                        cv2.putText(img, 'Hinh nho', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
                    else:
                        if distance12 >= distance_nose1 and distance12 >= distance_nose2:
                            if distance_center_eye_mouth >= distance_nose_ceye and distance_center_eye_mouth >= distance_nose_cmouth:
                                # if tl >= 0.6 and tl1 >= 0.6
                                rotate_img = alignment(crop_img, l_eye, r_eye)
                                rotate_img = cv2.resize(rotate_img, (112,112))
                                try:
                                    quality, emb = process_onnx(rotate_img, BACKBONE, QUALITY)
                                except:
                                    continue
                                print(emb.shape)
                                emb = emb.cpu().detach().numpy()
                                emb = np.array(emb,dtype=np.float32)
                                w, result = face_index.search(emb, k=1)
                                label = [labelOriginSet[i] for i in result[0]]

                                if w[0][0] >= 0.4:
                                    directory = label[0]
                                    print(directory)
                                    
                                    cv2.putText(img, directory, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2 )
                                    try:
                                        dir_fold = os.path.join(path_dir, directory)
                                        os.makedirs(dir_fold, exist_ok = True)
                                        frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(w[0][0], 2))  + '.jpg'
                                        print(frame_img_path)
                                        # img_save = cv2.resize(img_detect, (160,160))
                                        cv2.imwrite(frame_img_path, rotate_img)
                                        print("Directory created successfully")
                                        k=k+1
                                    except OSError as error:
                                        print("Directory can not be created")
                                else:
                                    print("unknow")
                                    cv2.putText(img, 'unknow', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2 )

                                    if quality[0] < 0.25:
                                        try:
                                            output_path = 'quality_result_bad'
                                            dir_fold = os.path.join(path_dir, output_path)
                                            os.makedirs(dir_fold, exist_ok = True)
                                            frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(quality[0], 4))  + '.jpg'
                                            cv2.imwrite(frame_img_path, rotate_img)

                                        except OSError as error:
                                            print("Directory can not be created")
                                    elif quality[0] > 0.5:
                                        try:
                                            name_path_id = 'ID' + str(name_id)
                                            output_path = 'quality_result_good/' + name_path_id
                                            dir_fold = os.path.join(path_dir, output_path)
                                            os.makedirs(dir_fold, exist_ok = True)

                                            frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(quality[0], 4))  + '.jpg'
                                            cv2.imwrite(frame_img_path, rotate_img)

                                            face_index.add(emb)
                                            labelOriginSet.append(name_path_id)

                                            name_id = name_id+1
                                        except OSError as error:
                                            print("Directory can not be created")
                                        
                                    else:
                                        try: 
                                            output_path = 'quality_result_bad'
                                            dir_fold = os.path.join(path_dir, output_path)
                                            os.makedirs(dir_fold, exist_ok = True)
                                            frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(quality[0], 4))  + '.jpg'
                                            cv2.imwrite(frame_img_path, rotate_img)
                                        except OSError as error:
                                            print("Directory can not be created")

                        else:
                            try: 
                                output_path = 'quality_result_bad'
                                dir_fold = os.path.join(path_dir, output_path)
                                os.makedirs(dir_fold, exist_ok = True)
                                frame_img_path = dir_fold + '/frame' + str(k) + '_' + str(round(quality[0], 4))  + '.jpg'
                                cv2.imwrite(frame_img_path, rotate_img)
                            except OSError as error:
                                print("Directory can not be created")

                time_end = time.time()
                avr_time = round(((time_end-time_start)), 2)
                print(avr_time)
                # print('Doneeeee')                                    
                                    
                #them hien thi video
                cv2.rectangle(img, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
            cv2.imshow("image", img)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
    cv2.destroyAllWindows()
    with open('X.pkl', 'wb') as f:
        pickle.dump(faceEncode, f)  

    with open('y.pkl', 'wb') as f:
        pickle.dump(labelOriginSet, f) 

main()