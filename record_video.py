import cv2
import numpy as np
import pickle
import os
import cv2
from keras_preprocessing.image import img_to_array

cam_url='http://admin:admin@192.168.43.1:8081'
cap = cv2.VideoCapture(cam_url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out2 = cv2.VideoWriter("data/video/4-fake.avi",fourcc, 30.0, (640, 480),True)

while True:

    ok, frame = cap.read()
    # if the frame was not grabbed, then we have reached the end of the stream
    (h, w) = frame.shape[:2]
    cv2.imshow("ZED-L", frame)
    key = cv2.waitKey(1)

    out2.write(frame)


    if key == ord("q"):
        out2.release()
        cap.release()
        cv2.destroyAllWindows()
        break




