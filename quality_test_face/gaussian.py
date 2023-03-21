import cv2

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
smooth = cv2.GaussianBlur(gray, (125,125), 0)
division = cv2.divide(gray, smooth, scale=255)
cv2.imwrite('output.jpg',division)