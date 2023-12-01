import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.abspath('.'))
print("sys.path", sys.path)
from core.face.face_vectorize_onnx import FaceVectorizeOnnx
from core.util.align_trans import warp_and_crop_face


face_vectorize = FaceVectorizeOnnx(use_resnet_50=True)

input_image = cv2.imread(sys.argv[1])
input_image = cv2.copyMakeBorder(input_image, 100, 100, 100, 100, cv2.BORDER_CONSTANT)
bboxes, kpss = face_vectorize.face_detector.detect(input_image, 0.5, input_size=(480, 480))


for i in range(len(bboxes)):
    # Width and height of the image
    h, w = input_image.shape[:2]
    kps = kpss[i]
    face_image_aligned = warp_and_crop_face(input_image, kps, crop_size=(w,h))
    cv2.imshow('Face Align', face_image_aligned)
    cv2.waitKey()
    left_eye = kps[0]
    right_eye = kps[1]
    print(left_eye)

    # Calculating coordinates of a central points of the rectangles
    left_eye_x = int(left_eye[0])
    left_eye_y = int(left_eye[1])
    
    right_eye_x = int(right_eye[0])
    right_eye_y = int(right_eye[1])

    left_eye = (left_eye_x, left_eye_y)
    right_eye = (right_eye_x, right_eye_y)
    
    cv2.circle(input_image, left_eye, 5, (255, 0, 0) , -1)
    cv2.circle(input_image, right_eye, 5, (255, 0, 0) , -1)
    cv2.line(input_image,right_eye, left_eye,(0,200,200),3)
    cv2.imshow('Face Align', input_image)
    cv2.waitKey()

    if left_eye_y > right_eye_y:
        A = (right_eye_x, left_eye_y)
        # Integer -1 indicates that the image will rotate in the clockwise direction
        direction = -1 
    else:
        A = (left_eye_x, right_eye_y)
        # Integer 1 indicates that image will rotate in the counter clockwise  
        # direction
        direction = 1 

    # cv2.circle(input_image, A, 5, (255, 0, 0) , -1)
    
    # cv2.line(input_image,right_eye, left_eye,(0,200,200),3)
    # cv2.line(input_image,left_eye, A,(0,200,200),3)
    # cv2.line(input_image,right_eye, A,(0,200,200),3)
    # cv2.imshow('Face Align', input_image)
    # cv2.waitKey()

    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle=np.arctan(delta_y/delta_x)
    angle = (angle * 180) / np.pi

    # Calculating a center point of the image
    # Integer division "//"" ensures that we receive whole numbers
    center = (w // 2, h // 2)
    # Defining a matrix M and calling
    # cv2.getRotationMatrix2D method
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    # Applying the rotation to our image using the
    # cv2.warpAffine method
    rotated = cv2.warpAffine(input_image, M, (w, h))
    cv2.imshow('Face Align', rotated)
    cv2.waitKey()
    cv2.imwrite("rata.jpg", rotated)