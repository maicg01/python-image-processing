import imp
from mtcnn import MTCNN
import cv2
import os
import threading
import time 

os.chdir('/home/devai01/Me/python-image-processing/')
# os.chdir('/home/devai01/Me/')
cap = video = cv2.VideoCapture('people_check.mp4')
detector = MTCNN()


def draw(image, face):
    for i in range(len(face)):
        person = face[i]
        x, y, width, height = person['box']
        image = cv2.rectangle(image, (x, y), (x+width, y+height), (0, 155, 255), 2)

    return image

        # for key, value in person['keypoints'].items():
        #     cv2.circle(image, value, 2, (0, 155, 255), 2)
    # cv2.waikey(1)
    # os.chdir('/home/devai01/Me/python-image-processing/results')
    # cv2.imwrite("face-detection-MTCNN.jpg", image)

while True:
    result, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640,480))
    faces = detector.detect_faces(img)
    # t2 = threading.Thread
    # t1 = threading.Thread(target=draw,args=(img,faces))
    # t1.start
    img = draw(img, faces)
    cv2.imshow("image", img)
    if cv2.waitKey(1) == ord('q'):
        break

