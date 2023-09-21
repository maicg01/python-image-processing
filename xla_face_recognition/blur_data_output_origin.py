import cv2
import os
from PIL import Image

path_read = "/home/maicg/Documents/daibieu/output"
path_save = "/home/maicg/Documents/daibieu/output_blur/"

for dir in os.listdir(path_read):
    pathdir = os.path.join(path_read,dir)
    for image in os.listdir(pathdir):
        pathName = os.path.join(pathdir,image)
        img2 = cv2.imread(pathName)
        img_blur = cv2.blur(img2, (3, 3))
        if not os.path.exists(path_save+ dir):
            os.mkdir(path_save+ dir)
        path_save_img = path_save+ dir + '/' + image[:-3] + "jpg"
        path_save_img_blur = path_save+ dir + '/' + image[:-4] + "_blur.jpg"
        print(path_save_img)
        cv2.imwrite(path_save_img, img2)
        cv2.imwrite(path_save_img_blur, img_blur)
