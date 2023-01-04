import numpy as np
import cv2

camera1 = 'rtsp://ai_dev:123654789@@@192.168.15.10:554/Streaming/Channels/1601'
camera2 = 'rtsp://ai_dev:123654789@@@192.168.15.10:554/Streaming/Channels/1501'
video_capture_0 = cv2.VideoCapture(camera1)
video_capture_1 = cv2.VideoCapture(camera2)


while True:
    dict_img = dict()
    print("===========================dict: ", dict_img)
    # Capture frame-by-frame
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()

    if (ret0):
        # Display the resulting frame
        frame0 = cv2.resize(frame0, (640,640))
        dict_img['cam0'] = frame0
        cv2.namedWindow('cam1')
        cv2.moveWindow('cam1', 0, 0)
        cv2.imshow('cam1', frame0)
        
        
    if (ret1):
        # Display the resulting frame
        frame1 = cv2.resize(frame1, (640,640))
        dict_img['cam1'] = frame1

        cv2.namedWindow('cam2')
        cv2.moveWindow('cam2', 710, 0)
        cv2.imshow('cam2', frame1)

    for key in dict_img:
        print(key)
        print(type(dict_img[key]))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_0.release()
video_capture_1.release()
cv2.destroyAllWindows()