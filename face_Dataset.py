import cv2
import numpy as np
import pickle
import os
import cv2
from keras_preprocessing.image import img_to_array
import tensorflow as tf

print("open camera")
confidence_default = 0.6
protoPath = "model/detect/deploy.prototxt"
modelPath = "model/detect/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
def cropImage(img):
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # 通过网络传递blob并获取检测和预测
    net.setInput(blob)
    detections = net.forward()
    if len(detections) > 0:
        # 我们假设每个图像只有一张脸，所以找到概率最大的边界框
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        # print("i is",i)
        # print("confidence is ",confidence)

        # 确保最大概率的检测也意味着我们的最小概率测试（从而帮助过滤掉弱检测）
        if confidence > confidence_default:
            # 计算面部边界框的（x，y）坐标并提取面部ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # print("box is",box)
            (startX, startY, endX, endY) = box.astype("int")
            face_color = img[startY:endY, startX:endX]
            face_color = cv2.resize(face_color,(128,128))
            # write the frame to disk

            cv2.imwrite("data/student/Yanglinxi/yanglinxi.png", face_color)

if __name__ == '__main__':
    img = cv2.imread("data/dartyimage/P00720-154330.jpg")
    cropImage(img)
