import argparse
import os
from time import time

import align.detect_face as detect_face
import cv2
import numpy as np
import tensorflow as tf
from lib.face_utils import judge_side_face
from lib.utils import Logger, mkdir
from project_root_dir import project_dir
from src.sort import Sort


from utils.facenetPreditctFunction import SCRFD, alignment, process_image_package, xyz_coordinates
from utils.fuction_compute import process_onnx, load_model_onnx 
logger = Logger()


def main():
    global colours, img_size
    args = parse_args()
    videos_dir = 'videos/1_video_AH.avi'
    output_path = args.output_path
    no_display = args.no_display
    detect_interval = args.detect_interval  # you need to keep a balance between performance and fluency
    margin = args.margin  # if the face is big in your video ,you can set it bigger for tracking easiler
    scale_rate = args.scale_rate  # if set it smaller will make input frames smaller
    show_rate = args.show_rate  # if set it smaller will dispaly smaller frames
    face_score_threshold = args.face_score_threshold

    detector = SCRFD(model_file='/home/maicg/Documents/python-image-processing/code-edit-insightFace/onnx/scrfd_2.5g_bnkps.onnx')
    detector.prepare(-1)
    mkdir(output_path)
    # for display
    if not no_display:
        colours = np.random.rand(32, 3)

    # init tracker
    tracker = Sort()  # create instance of the SORT tracker

    directoryname = os.path.join(output_path, videos_dir.split('.')[0])
    print("===directoryname: ", directoryname)
    cam = cv2.VideoCapture(videos_dir)
    c = 0
    while True:
        final_faces = []
        addtional_attribute_list = []
        ret, frame = cam.read()
        if not ret:
            logger.warning("ret false")
            break
        if frame is None:
            logger.warning("frame drop")
            break

        frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
        h, w, c = frame.shape
        # r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if c % detect_interval == 0:
            print("=====detect_interval: ", detect_interval)
            img_size = np.asarray(frame.shape)[0:2]
            mtcnn_starttime = time()
            # faces, points = detect_face.detect_face(r_g_b_frame, minsize, pnet, rnet, onet, threshold,
            #                                         factor)
            bboxes, kpss = process_image_package(frame, detector)

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
                    for j in range(5):
                        facial_landmarks.append(kps[j].tolist())
                    dist_rate, high_ratio_variance, width_rate = judge_side_face(np.array(facial_landmarks))
                    item_list = [crop_img, score, dist_rate, high_ratio_variance, width_rate]
                    addtional_attribute_list.append(item_list)

            print("facial landmarks: ", facial_landmarks)
            final_faces = np.array(face_list)
        
        print('===============addtional_attribute_list', addtional_attribute_list)
        trackers = tracker.update(final_faces, img_size, directoryname, addtional_attribute_list, detect_interval)
        c += 1

        for d in trackers:
            if not no_display:
                d = d.astype(np.int32)
                cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 3)
                if final_faces != []:
                    cv2.putText(frame, 'ID : %d  DETECT' % (d[4]), (d[0] - 10, d[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                colours[d[4] % 32, :] * 255, 2)
                    cv2.putText(frame, 'DETECTOR', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (1, 1, 1), 2)
                else:
                    cv2.putText(frame, 'ID : %d' % (d[4]), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                colours[d[4] % 32, :] * 255, 2)

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
