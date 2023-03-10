import cv2
import os
import math
import matplotlib.pyplot as plt

# kiem tra do tuong phan cua anh
# <60
path_img = "/home/maicg/Documents/Me/test-face-recognition/A_quality/7_2"
# path_img = "/home/maicg/Documents/Me/faceRecogCplus/save_image/hiep.tran"
# for img in os.listdir(path_img):
#     name_img = os.path.join(path_img, img)
#     image = cv2.imread(name_img)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     if cv2.mean(gray)[0] < 60:
#         print("gia tri trung binh mau: ", cv2.mean(gray)[0])
#         plt.imshow(image[:,:,::-1])
#         plt.show()

#     # kiem tra nhieu
#     blur = cv2.medianBlur(gray, 5)
#     if cv2.Laplacian(blur, cv2.CV_64F).var() < 50:
#         print("gia tri trung binh mau: ", cv2.Laplacian(blur, cv2.CV_64F).var())
#         plt.imshow(image[:,:,::-1])
#         plt.show()
#     else:
#         print("qua")

#     # kiem tra su bien dang
#     pi = math.pi
#     edges = cv2.Canny(blur, 100, 200)
#     lines = cv2.HoughLinesP(edges, 1, pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
#     if lines is not None:
#         plt.imshow(image[:,:,::-1])
#         plt.show()
#     else:
#         print("qua")

def check_quality(imgBgr):
    gray = cv2.cvtColor(imgBgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    pi = math.pi
    edges = cv2.Canny(blur, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if cv2.mean(gray)[0] < 50:
        print("gia tri trung binh mau: ", cv2.mean(gray)[0])
        return False
    
    if cv2.Laplacian(blur, cv2.CV_64F).var() < 50:
        print("gia tri nhieu: ", cv2.Laplacian(blur, cv2.CV_64F).var())
        return False
    
    if lines is not None:
        print("not line")
        return False
    
    return True

for img in os.listdir(path_img):
    name_img = os.path.join(path_img, img)
    image = cv2.imread(name_img)

    results = check_quality(imgBgr=image)
    if results == 0:
        plt.imshow(image[:,:,::-1])
        plt.show()
    else:
        print("good image")





