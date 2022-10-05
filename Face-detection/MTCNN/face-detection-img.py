import imp
from mtcnn import MTCNN
import cv2

img = cv2.cvtColor(cv2.imread("/home/devai01/Me/python-image-processing/people.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()

faces = detector.detect_faces(img)
print(len(faces))

# vẽ keypoints và bounding boxes
def draw(image, face):
    for i in len(face):
        person = face[i]
        x, y, width, height = person['box']
        cv2.rectangle(image, (x, y), (x+width, y+height), (0, 155, 255), 2)

        for key, value in person['keypoints'].items():
            cv2.circle(image, value, 2, (0, 155, 255), 2)

    cv2.imwrite('img.jpg', image)

draw(img,faces)