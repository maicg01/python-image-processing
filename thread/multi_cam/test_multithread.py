from threading import Thread
import threading
import time
import cv2

camera0 = 'rtsp://ai_dev:123654789@@@192.168.15.10:554/Streaming/Channels/1501'
camera1 = 'rtsp://ai_dev:123654789@@@192.168.15.10:554/Streaming/Channels/1601'
def cal_square(camera):
    cam0 = cv2.VideoCapture(camera)
    while True:
        ret, frame = cam0.read()

        cv2.imshow(str(camera), frame)
        if cv2.waitKey(1) == ord('q'):
            break
        # print(camera)


try:
	t = time.time()
	t1 = threading.Thread(target=cal_square, args=(camera0,))
	t2 = threading.Thread(target=cal_square, args=(camera1, ))
	t1.start()
	t2.start()
	t1.join()
	t2.join()
	print ("done in ", time.time()- t)
except:
	print ("error")