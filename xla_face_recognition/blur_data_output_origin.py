import cv2
import os
from PIL import Image

path_read = "/home/maicg/Desktop/test.mai (copy)"
path_save = "/home/maicg/Desktop/test.mai.nguyen.augu/"

for dir in os.listdir(path_read):
    pathdir = os.path.join(path_read,dir)
    for image in os.listdir(pathdir):
        pathName = os.path.join(pathdir,image)
        img2 = cv2.imread(pathName)
        # img_blur = cv2.blur(img2, (3, 3))
        # img_blur = cv2.blur(img2, (5, 5))
        # img_blur = cv2.blur(img2, (7, 7))
        # img_blur = cv2.blur(img2, (10, 10))
        # img_blur = cv2.blur(img2, (15, 15))
        # img_blur = cv2.blur(img2, (20, 20))
        # img_blur = cv2.blur(img2, (25, 25))
        # img_blur = cv2.blur(img2, (30, 30))
        img_blur = cv2.blur(img2, (35, 35))
        # img_blur = cv2.blur(img2, (40, 40))
        # img_blur = cv2.blur(img2, (45, 45))
        # img_blur = cv2.blur(img2, (50, 50))
        if not os.path.exists(path_save+ dir):
            os.mkdir(path_save+ dir)
        # path_save_img = path_save+ dir + '/' + image[:-3] + "jpg"
        path_save_img_blur = path_save+ dir + '/' + image[:-4] + "_blur91.jpg"
        # print(path_save_img)
        # cv2.imwrite(path_save_img, img2)
        cv2.imwrite(path_save_img_blur, img_blur)
