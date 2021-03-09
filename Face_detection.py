import sys
sys.path.insert(0, 'E:\pyworkspace')
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import json
import time
from model.FacemobileNet import FaceMobileNet
import config


#cos相似度
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


#把图片拿过来保存到cqnu.txt
# def ImgWrite(img_path):
#     file = "dataset/cqnu.txt"
#     basedir = os.path.abspath(os.path.dirname(__file__))
#     print("当前目录为"+basedir)
#     with open(file,'w')as f:
#         f.write(basedir+'\\'+img_path)


#把图片文件读入数据库写成json,
def writeJson(Jsonfile, transform, net):
    # 图片的路径的所有集合
    file = "data/student/student.csv"
    hh = {}
    checkpoint = []
    with open(file, 'r') as f:
        pairs = f.readlines()
    print(len(pairs))
    for i in range(len(pairs)):
        path = pairs[i].split()[0]
        img_compare = Image.open(path).convert('L')
        img_compare = transform(img_compare)
        imgc = img_compare.unsqueeze(0).cuda()
        feature2 = net(imgc).cpu().detach().numpy().squeeze().tolist()
        # zz.update({path: str(feature2).replace("\n","").replace('  ',' ').replace('[','').replace(']','')})
        hh[path] = feature2
    print(hh)
    json_str = json.dumps(hh)
    with open(Jsonfile, 'w', encoding='utf-8') as f:
        f.write(json_str)
    return json_str



#发现相似的人脸
def find_simface(img, transform, net):
    img = Image.open(img).convert('L')
    print(img)
    img = transform(img)
    img = img.unsqueeze(0).cuda()
    feature = net(img).cpu().detach().numpy().squeeze()
    #cqnu.txt
    file = "dataset/cqnu.txt"
    hh = {}
    checkpoint = []
    start = time.clock()
    with open("data/student/student.json", 'r', encoding='UTF-8') as f:
        #FaceLib.json是一个tensor
        load_dict = json.load(f)

    #拿出来计算图片与数据集里面tensor的相似度
    for key in load_dict:
        feature2 = np.array(load_dict[key])
        similarity = cosin_metric(feature, feature2)
        checkpoint.append(similarity)
        if similarity > 0.9:
            break
    checkarray = np.array(checkpoint)
    argmax = np.argmax(checkarray-1)
    predicted = np.max(checkpoint)

    print(argmax)
    end = time.clock()
    totaltime = end-start
    print("time: {}".format(totaltime))
    image = list(load_dict.keys())[argmax]


    return argmax, predicted, image, totaltime


if __name__ == "__main__":
    #假设读入一张图片，先入库写进了cqnu.txt

    Jsonfile = 'data/student/student.json'
    #需要把图片写进数据库
    img1 ="data/student/Yangjiezhi/yangjiezhi.png"
    basedir = os.path.abspath(os.path.dirname(__file__))
    print("当前目录为" + basedir)
    # ImgWrite(img1)
    # 首先把图片成Json

    model = FaceMobileNet(config.config.embedding_size).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(config.config.test_model, map_location=config.config.device))
    model.eval()
    writeJson(Jsonfile, config.config.test_transform, model)

    check = find_simface(img1, config.config.test_transform, model)
    print(check[0])



