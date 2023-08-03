import os
import random
import shutil

SOURCES_DIR = '/home/maicg/Documents/Me/YOLO/yolov5/runs/detect/data_face_pose/'
DES = '/home/maicg/Documents/Me/YOLO/yolov5/runs/detect/data_face_pose/data/'
images = os.listdir(SOURCES_DIR + 'images')
random.shuffle(images)
# print(len(images))

labels = os.listdir(SOURCES_DIR + 'labels')
# print(len(labels))

os.makedirs(DES + 'images/train/')
os.makedirs(DES + 'images/val/')
os.makedirs(DES + 'labels/train/')
os.makedirs(DES + 'labels/val/')

for image in images[:int(len(images)*0.8)]:
    shutil.copy(SOURCES_DIR + 'images/' + image, DES + 'images/train/' + image)
    if os.path.exists(SOURCES_DIR + 'labels/' + image[:-3] + 'txt'):
        shutil.copy(SOURCES_DIR + 'labels/' + image[:-3] + 'txt', DES + 'labels/train/' + image[:-3] + 'txt')
for image in images[int(len(images)*0.8):]:
    shutil.copy(SOURCES_DIR + 'images/' + image, DES + 'images/val/' + image)
    if os.path.exists(SOURCES_DIR + 'labels/' + image[:-3] + 'txt'):
        shutil.copy(SOURCES_DIR + 'labels/' + image[:-3] + 'txt', DES + 'labels/val/' + image[:-3] + 'txt')

print("done")