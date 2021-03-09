import threading

import torch
from flask import Flask, request, jsonify
import json
import cv2

from PIL import Image
import numpy as np
from keras_preprocessing.image import img_to_array
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import transforms

from Face_detection import find_simface
from config import config
from model.FacemobileNet import FaceMobileNet
from model.FeatherNet_1in import FeatherNetB_1in
import keras.backend as k
app = Flask(__name__)
app.debug = True
import base64


fe = FeatherNetB_1in()
fe.load_state_dict(torch.load("model/FeatherNet.pth"))
fe = fe.cuda()


arcface = FaceMobileNet(config.embedding_size).cuda()
arcface = nn.DataParallel(arcface)
arcface.load_state_dict(torch.load(config.test_model, map_location=config.device))
arcface.eval()


protoPath = "model/detect/deploy.prototxt"
modelPath = "model/detect/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

sess = k.get_session()
graph = sess.graph
test_trans = transforms.Compose(transforms=[
    transforms.ToTensor(),
    # normalize
])

arc_trans = transforms.Compose(transforms=[
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    # normalize
])
@app.route('/add', methods=['post','get'])
def get_fram():

    if request.method == 'POST':
        # 解析图片数据
        print("连接成功")
        img = base64.b64decode(str(request.form['url']))
        image_data = np.fromstring(img, np.
                                   uint8)
        image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        image_data = cv2.resize(image_data,(640,480))
        cv2.imwrite("1.png",image_data)
        image_data = cv2.imread("1.png")
        (h, w) = image_data.shape[:2]

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
                #
                # if startY < 0 or startY >= 480 or endY <= 0 or endY >= 480 or startX <= 0 or startX >= 640 or endX <= 0 or endX >= 640:
                #         print("签到失败，请重新自拍！")
                #         return "签到失败，请重新自拍！"
                face_color1 = image_data[startY:endY, startX:endX]
                cv2.imwrite("2.png",face_color1)
                face_color = cv2.resize(face_color1, (224, 224))
                face_color = Image.fromarray(cv2.cvtColor(face_color,cv2.COLOR_BGR2RGB))
                face_color = test_trans(face_color)
                # face.shape(1,32,32,3)
                face_color = np.expand_dims(face_color, axis=0)
                face_color = np.reshape(face_color,(1,3,224,224))
                face_color = torch.from_numpy(face_color)
                face_color = Variable(face_color).cuda()
                # with sess.graph.as_default():
                #                 #     with graph.as_default():
                #                 #         predict = model.predict(face_color)
                predict = fe(face_color)
                print(predict)
                out = predict.cpu().detach().squeeze().numpy()
                j = np.argmax(out)
                threshold = out[0]
                if threshold <= 0.8:
                    print("请不要试图攻击神经网络！！！！")

                    return "[INFO] 请不要试图攻击神经网络！"
                else:
                    print("签到成功！！！！！！")
                    img1 = "2.png"
                    check = find_simface(img1, config.test_transform, arcface)
                    if check[0] == 0:
                        print("[INFO] 马康哲，签到成功！")
                        return "[INFO] 马康哲，签到成功！"
                    if check[0] == 1:
                        print("[INFO] 杨杰之，签到成功！")
                        return "[INFO] 杨杰之，签到成功！"
                    if check[0] == 2:
                        print("[INFO] 杨琳希，签到成功！")
                        return "[INFO] 杨琳希，签到成功！"
                    else:
                        print("[INFO] 您不在签到名单中！")
                        return "[INFO] 您不在签到名单中！"




        # cv.imwrite('01.png', image_data)
        #         # cv.imshow('img',image_data)
        #         # cv.waitKey(0)
        return '成功接收图片，但签到失败！！！'


if __name__ == '__main__':
    # model = LivenessNet(112, 112, 3, 2).build()
    # model.load_weights("model/model.h5")
    app.run(host='172.20.10.6', port=5000,debug=True,threaded=True)
    # 这里指定了地址和端口号