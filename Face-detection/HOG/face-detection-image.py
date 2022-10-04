import dlib
import cv2

image = cv2.imread("/home/devai01/Me/python-image-processing/people.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#nhan cong cu do tim khuon mat HOG
hogFaceDetect = dlib.get_frontal_face_detector()
faces = hogFaceDetect(gray, 1)

#lap thong qua moi khuon mat va ve box
for (i, rect) in enumerate(faces):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    #ve box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#luu hinh anh
cv2.imwrite("Image.jpg", image)