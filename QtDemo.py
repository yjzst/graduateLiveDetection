from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog
import os
import numpy as np
import cv2

from Face_detection import find_simface
from config import config as conf, config
from model.FacemobileNet import FaceMobileNet
import torch.nn as nn
import torch
from PIL import Image


arcface = FaceMobileNet(config.embedding_size).cuda()
arcface = nn.DataParallel(arcface)
arcface.load_state_dict(torch.load(config.test_model, map_location=config.device))
arcface.eval()
# model = FaceMobileNet(conf.embedding_size)
# # model = nn.DataParallel(model)
# model.load_state_dict(torch.load(conf.test_model, map_location=conf.device))
# # model = torch.load(r"checkpoints/model.pkl")
# model.eval()
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(570, 420)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        #打开图片路径
        self.imgPath = QtWidgets.QLineEdit(self.centralwidget)
        self.imgPath.setGeometry(QtCore.QRect(20, 20, 370, 36))
        self.imgPath.setStyleSheet("background-color: rgb(204, 204, 204);color: rgb(74, 74, 74);border-color: rgb(0, 0, 0);font: 10pt \"Sans Serif\";")
        self.imgPath.setObjectName("imgPath")
        self.imgPath.setFrame(False)
        self.imgPath.setReadOnly(True)

        #打开图片按钮
        self.openImg = QtWidgets.QPushButton(self.centralwidget)
        self.openImg.setGeometry(QtCore.QRect(400, 20, 80, 35))
        self.openImg.setText("选择图片")
        self.openImg.setObjectName("openImg")
        self.openImg.clicked.connect(self.openFile)

        #开始识别按钮
        self.distinguish = QtWidgets.QPushButton(self.centralwidget)
        self.distinguish.setGeometry(QtCore.QRect(490, 20, 60, 35))
        self.distinguish.setText("识别")
        self.distinguish.setDisabled(True)
        self.distinguish.setObjectName("distinguish")
        self.distinguish.clicked.connect(self.startDistinguish)

        #输入的图片显示
        self.inputImg = QtWidgets.QLabel(self.centralwidget)
        self.inputImg.setGeometry(QtCore.QRect(20, 70, 260, 200))
        self.inputImg.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.inputImg.setObjectName("inputImg")
        self.inputImg.setAlignment(Qt.AlignCenter)

        #输出的图片显示
        self.outputImg = QtWidgets.QLabel(self.centralwidget)
        self.outputImg.setGeometry(QtCore.QRect(290, 70, 260, 200))
        self.outputImg.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.outputImg.setObjectName("outputImg")
        self.outputImg.setAlignment(Qt.AlignCenter)

        #输出信息展示
        self.information = QtWidgets.QTextEdit(self.centralwidget)
        self.information.setGeometry(QtCore.QRect(20, 280, 531, 131))
        self.information.setText("点击选择图片按钮，单击识别开始识别吧。")
        self.information.setObjectName("information")

        self.welcome = QtWidgets.QLabel(self.centralwidget)
        self.welcome.setGeometry(QtCore.QRect(0, 0, 570, 420))
        self.welcome.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.welcome.setObjectName("welcome")
        self.welcome.setAlignment(Qt.AlignCenter)
        self.welcome.setText("课堂助手人脸识别系统")
        pixmap = QPixmap("welcome.png")
        self.welcome.setPixmap(pixmap)
        #self.welcome.setVisible(False)

        self.welcomeInto = QtWidgets.QPushButton(self.centralwidget)
        self.welcomeInto.setGeometry(QtCore.QRect(430, 350, 80, 35))
        self.welcomeInto.setText("点击进入 >>>")
        self.welcomeInto.setStyleSheet("background-color: rgb(255, 255, 255);border:none")
        self.welcomeInto.setObjectName("welcomeInto")

        self.welcomeInto.clicked.connect(self.hideWelcome)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "课堂助手人脸识别Demo"))
        self.imgPath.setText(_translate("MainWindow", "请选择图片"))
        self.inputImg.setText(_translate("MainWindow", "当前识别的图片"))
        self.outputImg.setText(_translate("MainWindow", "识别结果返回图"))
        #self.setImg(self.inputImg,"cqnu.jpg")

    def setImg(self,setLabel,imgPath):
        pixmap = QPixmap(imgPath)
        if pixmap.height() < pixmap.width():
            scaredPixmap = pixmap.scaled((200/pixmap.width()) * pixmap.height(), 200, aspectRatioMode=Qt.KeepAspectRatio)
        else:
            scaredPixmap = pixmap.scaled(260, (260/pixmap.height()) * pixmap.width(), aspectRatioMode=Qt.KeepAspectRatio)
        setLabel.setPixmap(scaredPixmap)
    def hideWelcome(self):
        self.welcomeInto.setVisible(False)
        self.welcome.setVisible(False)

    #打开文件
    def openFile(self):
        fname = QFileDialog.getOpenFileName(MainWindow,'OpenFile',"c:/","Image files (*.jpg *.gif *.png)")
        print(fname)
        if fname[0] != '':
            self.setImg(self.inputImg,fname[0])
            self.imgFilePath = fname[0]
            self.imgPath.setText(fname[0])
            self.distinguish.setDisabled(False)
    #调用模型识别函数
    def startDistinguish(self):
        self.information.setText("正在识别中。。。")
        # process the target image
        image = self.imgFilePath
        # give the result
        argmax, predicted, image, totaltime = find_simface(image, conf.test_transform, arcface)



        if predicted > 0.8:
            pixmap = QPixmap(image)
            if pixmap.height() < pixmap.width():
                scaredPixmap = pixmap.scaled((200/pixmap.width()) * pixmap.height(), 200, aspectRatioMode=Qt.KeepAspectRatio)
            else:
                scaredPixmap = pixmap.scaled(260, (260/pixmap.height()) * pixmap.width(), aspectRatioMode=Qt.KeepAspectRatio)
            self.outputImg.setPixmap(scaredPixmap)
        else:
            self.outputImg.setText("识别结果暂无图片")
        #文本框内容
        Txt = '姓名：'+image+'\n相似度：' + str(predicted) + '\n' + '用时：' + str(totaltime)
        self.information.setText(Txt)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
