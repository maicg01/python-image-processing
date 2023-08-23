import argparse
import os
import sys
sys.path.append(os.path.abspath('.'))

import cv2
import numpy as np
import os
from core.util.align_trans import warp_and_crop_face
from core.face.face_vectorize_onnx_quality import FaceVectorizeOnnx

face_vectorize = FaceVectorizeOnnx(use_resnet_50=True)
path_image = '/home/maicg/Downloads/face/images/'
path_txt = '/home/maicg/Downloads/face/labels/'
txt_save = '/home/maicg/Downloads/face/label_save'
img_save = '/home/maicg/Downloads/face/img_save'

def convert_bounding_box_to_draw(image, landmarks, x_center, y_center, width, height): #chuyen toa do tu tuong doi sang tuyet doi
    image_draw = image.copy()
    # Lấy kích thước ảnh gốc
    image_height, image_width, _ = image.shape

    # Chuyển đổi tọa độ bounding box từ tương đối sang tuyệt đối
    x_min = int((x_center - width / 2) * image_width)
    y_min = int((y_center - height / 2) * image_height)
    x_max = int((x_center + width / 2) * image_width)
    y_max = int((y_center + height / 2) * image_height)
    cv2.rectangle(image_draw, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    results_landmarks = []
    for i in range(5):
        point_x = (landmarks[2 * i]) * image_width
        point_y = (landmarks[2 * i + 1]) * image_height
        results_landmarks.append([point_x, point_y])
        # cv2.circle(image_draw, (int(point_x), int(point_y)), 3, (0, 255, 0), -1)
    return image_draw, results_landmarks, x_min, y_min, x_max, y_max

for name_txt in os.listdir(path_txt):
    dir_file_txt = os.path.join(path_txt, name_txt)
    name_txt_save = txt_save + "/" + name_txt
    with open(dir_file_txt, "r") as f:
        line_names = [line.strip() for line in f.readlines()]
        with open(name_txt_save, "w") as file:
            for line in line_names:
                class_index, x_center, y_center, width, height, x1, y1, _, x2, y2, _, x3, y3, _, x4, y4, _, x5, y5, _ = map(float, line.split())
                landmarks = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
                if class_index == 7: 
                    print(path_image + name_txt[:-3] + "jpg")
                    image = cv2.imread(path_image + name_txt[:-3] + "jpg")
                    image_draw, results_landmarks, x_min, y_min, x_max, y_max = convert_bounding_box_to_draw(image, landmarks, x_center, y_center, width, height)
                    face_crop_align = warp_and_crop_face(image, results_landmarks)
                    face_crop_align_resize = cv2.resize(face_crop_align, (112,112))
                    input_vector, input_quality = face_vectorize.vectorize_encoding(face_crop_align_resize,
                                                                                        normalize=True)
                    cv2.putText(image_draw, str(input_quality[0]), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    modified_line = line + " " + str(input_quality[0]) + "\n"
                    cv2.imwrite(img_save + '/' + name_txt[:-3] + "jpg" , image_draw)
                    # cv2.imshow("image", image_draw)
                    # cv2.waitKey(0)

                else:
                    modified_line = line + " 0\n"
                
                file.write(modified_line)

