import glob
import random
import os
import shutil

PATH_IMG = 'runs/detect/danhnhan/train/img_lb'
PATH_TXT = 'runs/detect/danhnhan/train/lb'
img_paths = glob.glob(PATH_IMG+'/*.jpg')
txt_paths = glob.glob(PATH_TXT+'/*.txt')

# print(len(img_paths))

# Shuffle two list
img_txt = list(zip(img_paths, txt_paths))
random.seed(43)
random.shuffle(img_txt)
img_paths, txt_paths = zip(*img_txt)

data_size = len(img_paths)
r = 0.8
train_size = int(data_size * r)

#split
train_img_paths = img_paths[:train_size]
train_txt_paths = txt_paths[:train_size]

valid_img_paths = img_paths[train_size:]
valid_txt_paths = txt_paths[train_size:]

# copy them to images, labels folders
path = '/home/maicg/Documents/Me/YOLO/yolov5/datatrain'
images = path+'/images' 
labels = path+'/labels'
os.mkdir(images)
os.mkdir(labels)

train_img = images + '/train'
val_img = images + '/val'

train_label = labels + '/train'
val_label = labels + '/val'

os.mkdir(train_img)
os.mkdir(val_img)
os.mkdir(train_label)
os.mkdir(val_label)

def copy(paths, folder):
    for p in paths:
        shutil.copy(p, folder)

copy(train_img_paths, train_img)
copy(valid_img_paths, val_img)
copy(train_txt_paths, train_label)
copy(valid_txt_paths, val_label)

print("done")