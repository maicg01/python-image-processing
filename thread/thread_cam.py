import threading
import cv2, time
class Core:
    @staticmethod
    def detection(ip):
        # capture = cv2.VideoCapture('rtsp://'+str(ip))
        capture = cv2.VideoCapture(str(ip))
        while (capture.isOpened()):
            ret, frame = capture.read()
            cv2.imshow('Video',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
class Thread(threading.Thread):
    def __init__(self, threadID,ip):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.ip = ip

    def run(self):
        print("Start threadID" +str(self.threadID))
        Core.detection(self.ip)
        print("Exiting " + str(self.threadID))

threads = []
# thread1 = Thread(1,'192.168.1.4:5554/playlist.m3u')
# thread2 = Thread(2,'192.168.1.4:5554/playlist.m3u')
thread1 = Thread(1,'/home/maicg/Documents/python-image-processing/video_59s.avi')
thread2 = Thread(2,'/home/maicg/Documents/python-image-processing/video_gt.avi')
threads.append(thread1)
threads.append(thread2)
for i in threads:
    i.start()
print("Exit to main thread")