import argparse
import os
from time import time
from datetime import datetime
import pickle
import faiss
import torch

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN

def txt_format_yolo(pil_bbox, width, height):
    xcenter = ((pil_bbox[0] + pil_bbox[2]) / 2) / width
    ycenter = ((pil_bbox[1] + pil_bbox[3]) / 2) / height
    w = (pil_bbox[2] - pil_bbox[0]) / width
    h = (pil_bbox[3] - pil_bbox[1]) / height
    return [xcenter, ycenter, w, h]

def main():
    import glob
    #load moder 
    detector = MTCNN()
    pathdir = '/home/maicg/Downloads/archive/Train/Train/JPEGImages'
    path_txt = '/home/maicg/Documents/Me/YOLO/yolov5/runs/detect/exp2/labels'
    save_path = '/home/maicg/Documents/Me/YOLO/yolov5/runs/detect/exp2/save_path'
    save_txt = '/home/maicg/Documents/Me/YOLO/yolov5/runs/detect/exp2/save_txt'
    for dirImage in sorted(os.listdir(pathdir)):
        pathName = os.path.join(pathdir,dirImage)
        name_image = dirImage.split('.')[0]
        print(name_image)
        image = cv2.imread(pathName)
        dir_txt = path_txt + '/' + str(name_image) + '.txt'
        dir_save_txt = save_txt + '/' + str(name_image) + '.txt'
        faces = detector.detect_faces(image)
        try:
            with open(dir_txt, 'r') as f:
                lines = f.readlines()
        except:
            os.remove(pathName)
            continue

        for face in faces:
            x, y, w, h = face['box']
            x1,y1,x2,y2 = x, y, x+w, y+h
            
            cv2.rectangle(image, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
            print("x1,y1,x2,y2: ", x1,y1,x2,y2)
            if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                print("error outside")
                continue

            height, width, channels = image.shape
            pil_bbox = [x1,y1,x2,y2]
            yolo_fm = txt_format_yolo(pil_bbox, width, height)
            new_line = str(1) + ' ' + str(yolo_fm[0]) + ' ' + str(yolo_fm[1]) + ' ' + str(yolo_fm[2]) + ' ' + str(yolo_fm[3]) + '\n'
            # print(new_line)
            lines.append(new_line)

        with open(dir_save_txt, 'w') as f:
            f.writelines(lines)
        cv2.imwrite(save_path + '/' + str(name_image) + '.jpg' , image)
        
    

main()