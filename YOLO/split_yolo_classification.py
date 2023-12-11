import os
import random
import shutil

SOURCES_DIR = '/home/maicg/Documents/Me/CERBERUS/NghienCuu/ultralytics/jig_datatrain/train/'
DES = '/home/maicg/Documents/Me/CERBERUS/NghienCuu/ultralytics/data_train/'
os.makedirs(DES)

images = os.listdir(SOURCES_DIR + 'correct')
images_wrong = os.listdir(SOURCES_DIR + 'wrong')
random.shuffle(images)
random.shuffle(images_wrong)
# print(len(images))


os.makedirs(DES + 'train/correct/')
os.makedirs(DES + 'val/correct/')
os.makedirs(DES + 'train/wrong/')
os.makedirs(DES + 'val/wrong/')

for image in images[:int(len(images)*0.8)]:
    shutil.copy(SOURCES_DIR + 'correct/' + image, DES + 'train/correct/' + image)
for image in images[int(len(images)*0.8):]:
    shutil.copy(SOURCES_DIR + 'correct/' + image, DES + 'val/correct/' + image)

for image in images_wrong[:int(len(images)*0.8)]:
    shutil.copy(SOURCES_DIR + 'wrong/' + image, DES + 'train/wrong/' + image)
for image in images_wrong[int(len(images)*0.8):]:
    shutil.copy(SOURCES_DIR + 'wrong/' + image, DES + 'val/wrong/' + image)

print("done")