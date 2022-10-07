from re import I
import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('/home/devai01/Me/python-image-processing/file-haarcascade/haarcascade_frontalface_default.xml')

# To capture video from webcam. 
# cap = cv2.VideoCapture(0)
# To use a video file as input 
video = cv2.VideoCapture('/home/devai01/Me/python-image-processing/people_check.mp4')

frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         24, size)



while True:
    # Read the frame
    _, img = video.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        print(h)
    
    # # Display
    # cv2.imshow('img', img)
    result.write(img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
video.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()
   
print("The video was successfully saved")