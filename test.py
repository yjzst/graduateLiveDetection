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
protoPath = "model/detect/deploy.prototxt"
modelPath = "model/detect/res10_300x300_ssd_iter_140000.caffemodel"

image_data = cv2.imread("1.png")

(h, w) = image_data.shape[:2]
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
blob = cv2.dnn.blobFromImage(cv2.resize(image_data, (300, 300)), 1.0,
                         (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
if len(detections) > 0:
# 我们假设每个图像只有一张脸，所以找到概率最大的边界框
    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]

# 确保最大概率的检测也意味着我们的最小概率测试（从而帮助过滤掉弱检测）
    if confidence > 0.6:
        # 计算面部边界框的（x，y）坐标并提取面部ROI
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        # print("box is",box)
        (startX, startY, endX, endY) = box.astype("int")
        face_color = image_data[startY:endY, startX:endX]
        face_color = cv2.resize(face_color, (112, 112))
        face_color = face_color.astype("float") / 255.0
        face_color = img_to_array(face_color)
        # face.shape(1,32,32,3)
        face_color = np.expand_dims(face_color, axis=0)
        predict = model.predict(face_color)[0]
        print(predict)


def liveness():
    while True:
        if os.path.exists("1.png"):
            image_data = cv2.imread("1.png")
            (h, w) = image_data.shape[:2]
            net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
            blob = cv2.dnn.blobFromImage(cv2.resize(image_data, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            if len(detections) > 0:
                # 我们假设每个图像只有一张脸，所以找到概率最大的边界框
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                # 确保最大概率的检测也意味着我们的最小概率测试（从而帮助过滤掉弱检测）
                if confidence > 0.6:
                    # 计算面部边界框的（x，y）坐标并提取面部ROI
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    # print("box is",box)
                    (startX, startY, endX, endY) = box.astype("int")
                    face_color = image_data[startY:endY, startX:endX]
                    face_color = cv2.resize(face_color, (112, 112))
                    face_color = face_color.astype("float") / 255.0
                    face_color = img_to_array(face_color)
                    # face.shape(1,32,32,3)
                    face_color = np.expand_dims(face_color, axis=0)
                    predict = model.predict(face_color)[0]
                    j = np.argmax(predict)
                    if j == 0:
                        print("请不要试图攻击神经网络！！！！")

                    else:
                        print("签到成功！！！！！！")

        else:
            return "重新上传"
