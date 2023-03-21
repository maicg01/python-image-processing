import cv2
import matplotlib.pyplot as plt
import os


# path_img = "/home/maicg/Documents/Me/test-face-recognition/A_quality/7_2"
path_img = "/home/maicg/Documents/Me/python-image-processing/quality_test_face/pic"
for img in os.listdir(path_img):
    name_img = os.path.join(path_img, img)
    img = cv2.imread(name_img)

    # Chuyển đổi ảnh sang ảnh xám
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tính độ tương phản
    contrast = cv2.convertScaleAbs(gray_img, alpha=2, beta=0)
    std_dev = cv2.meanStdDev(contrast)[1][0][0]
    print("std_dev: ", std_dev)

    # Tính độ sáng
    mean_brightness = cv2.mean(gray_img)[0]
    print("mean_brightness: ", mean_brightness)


    # Tính độ nhiễu
    blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    noise = cv2.absdiff(gray_img, blurred_img)
    mean_noise = cv2.mean(noise)[0]
    if mean_noise > 3 or mean_noise < 1.1:
        print("mean_noise: ", mean_noise)
        plt.imshow(img[:,:, ::-1])
        plt.show()


