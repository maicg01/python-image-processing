import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

dir_path = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/data/nkt'

for image in os.listdir(dir_path):
    img_path = os.path.join(dir_path,image)
    img = cv2.imread(img_path)
    img = cv2.resize(img,(64,64))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sum_saturation = np.sum(hsv[:,:,1])
    area = 64*64
    avg_saturation = sum_saturation/area

    if avg_saturation > 80 and avg_saturation < 150:
        print("deo khau trang vao")
    else:
        print("dung la dang deo khau trang")
    print(avg_saturation)

    plt.imshow(hsv[:,:,::-1])
    plt.show()
    



