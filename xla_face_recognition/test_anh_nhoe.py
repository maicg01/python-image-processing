import cv2
import sys
import os
from os.path import exists

def is_image_blurry(image, threshold):

    # Chuyển ảnh sang ảnh đen trắng
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tính toán chỉ số độ nhòe (Blur Index) dựa trên phương pháp Laplacian
    blur_index = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Kiểm tra nếu chỉ số độ nhòe nhỏ hơn ngưỡng
    if blur_index < threshold:
        return True  # Ảnh bị nhòe
    else:
        return False  # Ảnh không bị nhòe
threshold = 200
folder = sys.argv[1]
for dir_image in os.listdir(folder):
    path_image = os.path.join(folder, dir_image)
    face_image = cv2.imread(path_image)

    # Kiểm tra xem ảnh có bị nhòe hay không
    if is_image_blurry(face_image, threshold):
        print("Ảnh bị nhòe")
        if not exists('/home/maicg/Documents/Me/python-image-processing/ADACTYGOC/IDsai/save_quality/1758_nhoe/'):
            os.mkdir('/home/maicg/Documents/Me/python-image-processing/ADACTYGOC/IDsai/save_quality/1758_nhoe/')
        path_save = '/home/maicg/Documents/Me/python-image-processing/ADACTYGOC/IDsai/save_quality/1758_nhoe/' + dir_image
        cv2.imwrite(path_save, face_image)
    else:
        print("Ảnh không bị nhòe")
        if not exists('/home/maicg/Documents/Me/python-image-processing/ADACTYGOC/IDsai/save_quality/1758_yes/'):
            os.mkdir('/home/maicg/Documents/Me/python-image-processing/ADACTYGOC/IDsai/save_quality/1758_yes/')
        path_save = '/home/maicg/Documents/Me/python-image-processing/ADACTYGOC/IDsai/save_quality/1758_yes/' + dir_image
        cv2.imwrite(path_save, face_image)