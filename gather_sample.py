import numpy as np
import argparse
import cv2
import os

protoPath = "model/detect/deploy.prototxt"
modelPath = "model/detect/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
input = "data/video/4-fake.avi"
vs = cv2.VideoCapture(input)
read = 0
saved = 382
skip = 5
output_color = "data/face-antispoof-data/fake"

confidence_default = 0.6
# loop over frames from the video file stream
while True:
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    # 增加到目前为止读取的总帧数
    read += 1

    # 检查我们是否应该处理这个框架，因为不是每一帧都需要
    if read % skip != 0:
        continue

    # 抓住框架尺寸并从框架构造一个blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # 通过网络传递blob并获取检测和预测
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

            # if startY < 0 or startY >= 480 or endY <= 0 or endY >= 480 or startX <= 0 or startX >= 640 or endX <= 0 or endX >= 640:
            #     break
            face_color = frame[startY:endY, startX:endX]

            # write the frame to disk
            p = os.path.sep.join([output_color,
                                  "{}.jpg".format(saved)])
            cv2.imwrite(p, face_color)
            saved += 1

            print("[INFO] saved {} to disk".format(p))

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()
