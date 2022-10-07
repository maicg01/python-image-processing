from ast import arg
import imp
from filevideostream import FileVideoStream
from fps import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", '--video', required=True, help="path to input video file")
args = vars(ap.parse_args())

print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

fps = FPS().start()
while fvs.more():
    frame = fvs.read()
    frame = imutils.resize(frame, width=450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])

    cv2.putText(frame, "Queue Size : {}".format(fvs.Q.qsize()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    fps.update()

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    fvs.stop()