import cv2
import numpy as np
import pickle
import os
import cv2
from keras_preprocessing.image import img_to_array
import tensorflow as tf
from model.livenessNet import LivenessNet

model = LivenessNet(112, 112, 3, 2).build()
model.load_weights("model/model.h5")

print("open camera")
confidence_default = 0.6
protoPath = "model/detect/deploy.prototxt"
modelPath = "model/detect/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
cap = cv2.VideoCapture("http://admin:admin@172.20.10.11:8081")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# ok, frame = cap.read()

while True:
    _, frame = cap.read()

    # if the frame was not grabbed, then we have reached the end of the stream
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # 确保至少找到一张脸
    if len(detections) > 0:
        # 我们假设每个图像只有一张脸，所以找到概率最大的边界框
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # 确保最大概率的检测也意味着我们的最小概率测试（从而帮助过滤掉弱检测）
        if confidence > confidence_default:
            # 计算面部边界框的（x，y）坐标并提取面部ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # print("box is",box)
            (startX, startY, endX, endY) = box.astype("int")

            if startY < 0 or startY >= 480 or endY <= 0 or endY >= 480 or startX <= 0 or startX >= 640 or endX <= 0 or endX >= 640:
                continue
            face_color = frame[startY:endY, startX:endX]
            face_color = cv2.resize(face_color, (112, 112))
            face_color = face_color.astype("float") / 255.0
            face_color = img_to_array(face_color)
            # face.shape(1,32,32,3)
            face_color = np.expand_dims(face_color, axis=0)
            predict = model.predict(face_color)[0]
            j = np.argmax(predict)
            preds = predict[j]
            if j == 0:
                # distance = face_Depth[0, 55, 55, 0] * 255.0 / 0.03
                cv2.putText(frame, "fake", (startX, startY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)

            else:
                cv2.putText(frame, "real", (startX, startY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()



