import argparse
import os
from time import time
from datetime import datetime
import pickle
import faiss

import cv2
import numpy as np
import tensorflow as tf
from lib.face_utils import judge_side_face
from lib.utils import Logger, mkdir
from project_root_dir import project_dir
from src.sort import Sort


from utils.facenetPreditctFunction import SCRFD, alignment, process_image_package, xyz_coordinates
from utils.fuction_compute import process_onnx, load_model_onnx 
# logger = Logger()

def search(list, platform):
    for i in range(len(list)):
        if list[i] == platform:
            return True
    return False

def main():
    path_dir = './data_test/test5'
    global colours, img_size
    args = parse_args()
    videos_dir = 'videos/video_gt.avi'
    # videos_dir = 'videos/1_video_AH.avi'
    # videos_dir = 'videos/2_Obama.mp4'
    # videos_dir = '/home/maicg/Documents/python-image-processing/pexels-artem-podrez-7956859.mp4'
    # videos_dir = 'rtsp://ai_dev:123654789@@@192.168.15.10:554/Streaming/Channels/1601'
    output_path = args.output_path
    no_display = args.no_display
    print("=======no display: ", no_display)
    detect_interval = args.detect_interval  # you need to keep a balance between performance and fluency
    margin = args.margin  # if the face is big in your video ,you can set it bigger for tracking easiler
    scale_rate = args.scale_rate  # if set it smaller will make input frames smaller
    show_rate = args.show_rate  # if set it smaller will dispaly smaller frames
    face_score_threshold = args.face_score_threshold

    detector = SCRFD(model_file='/home/maicg/Documents/python-image-processing/code-edit-insightFace/onnx/scrfd_2.5g_bnkps.onnx')
    detector.prepare(-1)
    BACKBONE = load_model_onnx('./onnx/Resnet2F.onnx')
    QUALITY = load_model_onnx('./onnx/Quality.onnx')

    # set origin
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
    faceEncode = list(faceEncode)
    print("ty==========", type(faceEncode))

    #set new data ID
    faceEncode_newID = []
    label_newID = []
    try:
        with open('X1.pkl', 'rb') as f:
            faceEncode_newID = pickle.load(f)
            

        with open('y1.pkl', 'rb') as f:
            label_newID = pickle.load(f)  
    except:
        print("No file unknown")  

    faceEncode_newID = np.array(faceEncode_newID,dtype=np.float32)
    # create index with faiss
    face_index_newID = faiss.IndexFlatIP(512)
    # add vector
    try:
        face_index_newID.add(faceEncode_newID)
    except:
        pass
    faceEncode_newID = list(faceEncode_newID)
    print("ty==========", type(faceEncode_newID))
    
    
    mkdir(output_path)
    # for display
    if not no_display:
        colours = np.random.rand(32, 3)

    # init tracker
    tracker = Sort()  # create instance of the SORT tracker

    directoryname = os.path.join(output_path, 'scrfd_detect')
    print("===directoryname: ", directoryname)
    cam = cv2.VideoCapture(videos_dir)

    list_id = []
    name_list = []
    # name_id = len(labelOriginSet)
    name_id = len(label_newID)
    print("len data origin: ", len(labelOriginSet))
    print("name ID: ", name_id)
    quit_loop = False
    remember_tracking = 0
    while True:
        final_faces = []
        addtional_attribute_list = []
        for i in range(2):
            try:
                ret, frame = cam.read()
            except:
                quit_loop = True
        
        # take date and time
        # dd/mm/YY H:M:S
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y-%H:%M:%S")
        print("date and time =", dt_string)

        # frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
        scrdf_starttime = time()
        try:
            bboxes, kpss = process_image_package(frame, detector)
        except:
            quit_loop = True
        # logger.info("MTCNN detect face cost time : {} s".format(round(time() - scrdf_starttime, 3)))

        if quit_loop:
            with open('X1.pkl', 'wb') as f:
                pickle.dump(faceEncode_newID, f)  
            print("runningnnn")
            with open('y1.pkl', 'wb') as f:
                pickle.dump(label_newID, f)

            with open('X.pkl', 'wb') as f:
                pickle.dump(faceEncode, f)  
            print("runningnnn")
            with open('y.pkl', 'wb') as f:
                pickle.dump(labelOriginSet, f)
            break
        h, w, c = frame.shape
        area_base = h*w
        tl = 0
        tl1 = 0
        
        img_size = np.asarray(frame.shape)[0:2]

        face_list = []
        facial_landmarks = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1,y1,x2,y2,_ = bbox.astype(np.int)
            _,_,_,_,score = bbox.astype(np.float)

            crop_img = frame[y1:y2, x1:x2]
            
            h1 = int(crop_img.shape[0])
            w1 = int(crop_img.shape[1])
            face = np.array([x1,y1,x2,y2,score])
            face_list.append(face)

            if kpss is not None:
                kps = kpss[i]
                # print("kpslllllllll: ", type(kps))
                for j in range(5):
                    facial_landmarks.append(kps[j].tolist())
                dist_rate, high_ratio_variance, width_rate = judge_side_face(np.array(facial_landmarks))
                item_list = [crop_img, score, dist_rate, high_ratio_variance, width_rate]
                addtional_attribute_list.append(item_list)

        final_faces = np.array(face_list)
        
        # print('===============addtional_attribute_list', addtional_attribute_list)
        trackers = tracker.update(final_faces, img_size, directoryname, addtional_attribute_list, detect_interval)
        # print("==============tracker: ", trackers)

        remember = 0
        for d in trackers:
            # print(len(trackers))
            new_kps = facial_landmarks[remember:remember+5]
            new_kps = np.array(new_kps)
            # print("new_kps: ", new_kps[0])
            remember = remember + 5
            # print("no display: ", no_display)

            distance12, distance_nose1, distance_nose2, distance_center_eye_mouth, distance_nose_ceye, distance_nose_cmouth, distance_eye, distance_mouth, l_eye, r_eye = xyz_coordinates(new_kps)
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

            if not no_display:
                d = d.astype(np.int32)
                cropImg = frame[int(d[1]):int(d[3]),int(d[0]):int(d[2])]
                h1 = int(cropImg.shape[0])
                w1 = int(cropImg.shape[1])
                area_crop = h1*w1

                if area_crop == 0:
                    break

                if search(list_id, d[4]) is False:
                    list_id.append(d[4])
                    if (area_base/area_crop) > ((1080*1920)/(64*64)):
                        print("hinh nho")
                        cv2.putText(frame, 'Hinh nho', (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0,0,255), thickness=2)
                        list_id.pop()
                    else:
                        if distance12 >= distance_nose1 and distance12 >= distance_nose2:
                            if distance_center_eye_mouth >= distance_nose_ceye and distance_center_eye_mouth >= distance_nose_cmouth:
                                # if tl >= 0.6 and tl1 >= 0.6
                                rotate_img = alignment(cropImg, l_eye, r_eye)
                                rotate_img = cv2.resize(rotate_img, (112,112))
                                try:
                                    quality, emb = process_onnx(rotate_img, BACKBONE, QUALITY)
                                except:
                                    continue
                                # print(emb.shape)
                                emb = emb.cpu().detach().numpy()
                                emb = np.array(emb,dtype=np.float32)

                                # search in database
                                w, result = face_index.search(emb, k=1)
                                label = [labelOriginSet[i] for i in result[0]]

                                print("=========listid", list_id)
                                print("=========name_id", name_id)
                                if w[0][0] >= 0.53: #test vs 0.45
                                    directory = label[0]
                                    print(directory)
                                    name_list.append(directory)

                                    # # add emb in datebase
                                    # face_index.add(emb)
                                    # labelOriginSet.append(directory)
                                    # emb = np.array(emb,dtype=np.float32).reshape(512,)
                                    # faceEncode.append(emb)

                                    cv2.putText(frame, directory, (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=colours[d[4] % 32, :] * 255, thickness=2 )
                                    try:
                                        dir_fold = os.path.join(path_dir, directory)
                                        os.makedirs(dir_fold, exist_ok = True)
                                        frame_img_path = dir_fold + '/' + str(dt_string) + '_' + str(round(quality[0], 4)) + '_' + str(round(w[0][0], 2))  + '.jpg'
                                        print(frame_img_path)
                                        # img_save = cv2.resize(img_detect, (160,160))
                                        cv2.imwrite(frame_img_path, rotate_img)
                                        print("Directory created successfully")
                                    except OSError as error:
                                        print("Directory can not be created")
                                else:
                                    # search in data unknow
                                    w, result = face_index_newID.search(emb, k=1)
                                    print("=================results[0]: ", result[0])
                                    print("=================w[0][0]: ", w[0][0])
                                    print("label new id: ", label_newID)
                                    try:
                                        label = [label_newID[i] for i in result[0]]
                                    except:
                                        pass
                                    if w[0][0] >= 0.53: #test vs 0.45
                                        directory = label[0]
                                        # print(directory)
                                        name_list.append(directory)

                                        # # add emb in data unknown
                                        # face_index_newID.add(emb)
                                        # label_newID.append(directory)
                                        # emb = np.array(emb,dtype=np.float32).reshape(512,)
                                        # faceEncode_newID.append(emb)

                                        cv2.putText(frame, directory, (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=colours[d[4] % 32, :] * 255, thickness=2 )
                                        try:
                                            dir_fold = os.path.join(path_dir, directory)
                                            os.makedirs(dir_fold, exist_ok = True)
                                            frame_img_path = dir_fold + '/' + str(dt_string) + '_' + str(round(quality[0], 4)) + '_' + str(round(w[0][0], 2))  + '.jpg'
                                            print(frame_img_path)
                                            # img_save = cv2.resize(img_detect, (160,160))
                                            cv2.imwrite(frame_img_path, rotate_img)
                                            print("Directory created successfully")
                                        except OSError as error:
                                            print("Directory can not be created")
                                    else: 
                                        if quality[0] > 0.52:
                                            try:
                                                name_path_id = 'ID' + str(name_id)
                                                print("chay 33333333333333333333333333333333333333333333333333333333")

                                                cv2.putText(frame, name_path_id, (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=colours[d[4] % 32, :] * 255, thickness=2 )
                                                output_path = 'quality_result_good/' + name_path_id
                                                dir_fold = os.path.join(path_dir, output_path)
                                                os.makedirs(dir_fold, exist_ok = True)

                                                frame_img_path = dir_fold + '/' + str(dt_string) + '_' + str(round(quality[0], 4)) + '_' + str(round(w[0][0], 2))  + '.jpg'
                                                cv2.imwrite(frame_img_path, rotate_img)

                                                face_index_newID.add(emb)
                                                label_newID.append(name_path_id)

                                                emb = np.array(emb,dtype=np.float32).reshape(512,)
                                                faceEncode_newID.append(emb)
                                                name_list.append(name_path_id)

                                                name_id = name_id+1
                                            except OSError as error:
                                                print("Directory can not be created")
                                        
                                        else:
                                            list_id.pop()
                                            cv2.putText(frame, 'bad_qlt', (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0,0,255), thickness=2 )
                                            try: 
                                                output_path = 'quality_result_bad'
                                                dir_fold = os.path.join(path_dir, output_path)
                                                os.makedirs(dir_fold, exist_ok = True)
                                                frame_img_path = dir_fold + '/' + str(dt_string) + '_' + str(round(quality[0], 4)) + '_' + str(round(w[0][0], 2))  + '.jpg'
                                                cv2.imwrite(frame_img_path, rotate_img)
                                            except OSError as error:
                                                print("Directory can not be created")
                            else:
                                list_id.pop()
                                cv2.putText(frame, 'bad_img', (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0,0,255), thickness=2 )

                        else:
                            list_id.pop()
                            cv2.putText(frame, 'bad_img', (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0,0,255), thickness=2 )
                else:
                    remember_tracking = remember_tracking + 1
                    try:
                        f_name = name_list[list_id.index(d[4])]
                    except:
                        print("sai o f_name")
                    cv2.putText(frame, f_name, (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=colours[d[4] % 32, :] * 255, thickness=2 )

                    if remember_tracking == 10:
                        if distance12 >= distance_nose1 and distance12 >= distance_nose2:
                            if distance_center_eye_mouth >= distance_nose_ceye and distance_center_eye_mouth >= distance_nose_cmouth:
                                rotate_img = alignment(cropImg, l_eye, r_eye)
                                rotate_img = cv2.resize(rotate_img, (112,112))
                                try:
                                    quality, emb = process_onnx(rotate_img, BACKBONE, QUALITY)
                                except:
                                    continue
                                # print(emb.shape)
                                emb = emb.cpu().detach().numpy()
                                emb = np.array(emb,dtype=np.float32)
                                if quality[0] > 0.4:
                                    if search(labelOriginSet, f_name) is True:
                                        face_index.add(emb)
                                        labelOriginSet.append(f_name)
                                        emb = np.array(emb,dtype=np.float32).reshape(512,)
                                        faceEncode.append(emb)
                                        print("labelOriginSet: ", labelOriginSet)

                                        #save img 
                                        output_path = 'quality_result_good/' + f_name
                                        dir_fold = os.path.join(path_dir, output_path)
                                        os.makedirs(dir_fold, exist_ok = True)
                                        frame_img_path = dir_fold + '/' + str(dt_string) + '_' + str(round(quality[0], 4)) + '.jpg'
                                        cv2.imwrite(frame_img_path, rotate_img)

                                    else:
                                        print("running add new label data........................................")
                                        face_index_newID.add(emb)
                                        label_newID.append(f_name)
                                        emb = np.array(emb,dtype=np.float32).reshape(512,)
                                        faceEncode_newID.append(emb)
                                        print("label_newID: ", label_newID)

                                        #save img 
                                        output_path = 'quality_result_good/' + f_name
                                        dir_fold = os.path.join(path_dir, output_path)
                                        os.makedirs(dir_fold, exist_ok = True)
                                        frame_img_path = dir_fold + '/' + str(dt_string) + '_' + str(round(quality[0], 4))  + '.jpg'
                                        cv2.imwrite(frame_img_path, rotate_img)

                        remember_tracking = 0

                cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 3)

        if not no_display:
            frame = cv2.resize(frame, (0, 0), fx=show_rate, fy=show_rate)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", type=str,
                        help='Path to the data directory containing aligned your face patches.', default='videos')
    parser.add_argument('--output_path', type=str,
                        help='Path to save face',
                        default='facepics')
    parser.add_argument('--detect_interval',
                        help='how many frames to make a detection',
                        type=int, default=1)
    parser.add_argument('--margin',
                        help='add margin for face',
                        type=int, default=10)
    parser.add_argument('--scale_rate',
                        help='Scale down or enlarge the original video img',
                        type=float, default=0.7)
    parser.add_argument('--show_rate',
                        help='Scale down or enlarge the imgs drawn by opencv',
                        type=float, default=1)
    parser.add_argument('--face_score_threshold',
                        help='The threshold of the extracted faces,range 0<x<=1',
                        type=float, default=0.65)
    parser.add_argument('--face_landmarks',
                        help='Draw five face landmarks on extracted face or not ', action="store_true")
    parser.add_argument('--no_display',
                        help='Display or not', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
