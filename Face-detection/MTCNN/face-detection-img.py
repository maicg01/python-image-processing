import imp
from mtcnn import MTCNN
import cv2

img = cv2.cvtColor(cv2.imread("E:\python-image-processing\people.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()

faces = detector.detect_faces(img)
# [
#     {
#         'box': [277, 90, 48, 63],
#         'keypoints':
#         {
#             'nose': (303, 131),
#             'mouth_right': (313, 141),
#             'right_eye': (314, 114),
#             'left_eye': (291, 117),
#             'mouth_left': (296, 143)
#         },
#         'confidence': 0.99851983785629272
#     }
# ]

def draw(image, face):
    for i in range(len(face)):
        person = face[i]
        x, y, width, height = person['box']
        cv2.rectangle(image, (x, y), (x+width, y+height), (0, 155, 255), 2)

        # for key, value in person['keypoints'].items():
        #     cv2.circle(image, value, 2, (0, 155, 255), 2)
    
    cv2.imwrite("img.jpg", image)


draw(img, faces)