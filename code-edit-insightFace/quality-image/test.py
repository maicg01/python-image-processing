import numpy as np
import cv2 

x = np.random.randn(5,2,3)
print(x)
ccropped = x.swapaxes(1, 2).swapaxes(0,1)
print("sau khi swap: ", ccropped)
print("shape sau khi swap: ", ccropped.shape)

img = cv2.imread('/home/maicg/Documents/FaceQuality/test_me/frameNEW126_0.6_0.62_0.81.jpg')
print("shape image: ", img.shape)