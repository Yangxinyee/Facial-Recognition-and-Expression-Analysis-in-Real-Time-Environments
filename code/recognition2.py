# -*- coding = utf-8 -*-
# @Author : XinyeYang
# @File : FaceCNN.py
# @Software : PyCharm

import sys
import os
from faces_train import Model
import cv2
import torch
import torch.nn as nn
import numpy as np
from statistics import mode
# 人脸数据归一化,将像素值从0-255映射到0-1之间
def preprocess_input(images):
    """ preprocess input by substracting the train mean
    # Arguments: images or image of any shape
    # Returns: images or image with substracted train mean (129)
    """
    images = images/255.0
    return images


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


class FaceCNN(nn.Module):
    # 初始化网络结构
    def __init__(self):
        super(FaceCNN, self).__init__()

        # 第一次卷积、池化
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.BatchNorm2d(num_features=64),  # 归一化
            nn.RReLU(inplace=True),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大值池化
        )

        # 第二次卷积、池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第三次卷积、池化
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 参数初始化
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=7),
        )

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 数据扁平化
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y


#面部识别分类器
classification_model_path = 'C:/Users/24372/Desktop/test1/model/model_cnn.pkl'
# 加载表情识别模型
emotion_classifier = torch.load(classification_model_path)
frame_window = 10
#表情标签
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
emotion_window = []


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)
    #加载模型
    model = Model()
    model.load_model(file_path='C:/Users/24372/Desktop/test1/model/aggregate.face.model.h5')
    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)
    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)
    # 人脸识别分类器本地存储路径
    cascade_path = "C:/Users/24372/Desktop/test1/model/haarcascade_frontalface_default.xml"

    # 循环检测识别人脸
    while True:
        ret, frame = cap.read()  # 读取一帧视频
        if ret is True:
            # 图像灰化，降低计算复杂度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        # 使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)

        # 利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=2, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                # 截取脸部图像提交给模型识别这是谁
                image = frame[y: y + h, x: x + w]
                faceID = model.face_predict(image)
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                #face_id判断
                for i in range(len(os.listdir('./data/'))):
                    if i == faceID:
                        # 文字提示是谁
                        cv2.putText(frame,os.listdir('./data/')[i],
                                    (x + 30, y + 30),  # 坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                    1,  # 字号
                                    (255, 0, 255),  # 颜色
                                    2)  # 字的线宽
                    if faceID == -1:
                        cv2.putText(frame, 'faces_other',
                                    (x + 30, y + 30),  # 坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                    1,  # 字号
                                    (255, 0, 255),  # 颜色
                                    2)  # 字的线宽

                image_ = frame_gray[y: y + h, x: x + w]
                face = cv2.resize(image_, (48, 48))  #
                # 扩充维度，shape变为(1,48,48,1)
                # 将（1，48，48，1）转换成为(1,1,48,48)
                face = np.expand_dims(face, 0)
                face = np.expand_dims(face, 0)
                # 人脸数据归一化，将像素值从0-255映射到0-1之间
                face = preprocess_input(face)
                new_face = torch.from_numpy(face)
                new_new_face = new_face.float().requires_grad_(False)
                # 调用我们训练好的表情识别模型，预测分类
                emotion_arg = np.argmax(emotion_classifier.forward(new_new_face).detach().numpy())
                emotion = emotion_labels[emotion_arg]
                emotion_window.append(emotion)

                if len(emotion_window) >= frame_window:
                    emotion_window.pop(0)
                try:
                    # 获得出现次数最多的分类
                    emotion_mode = mode(emotion_window)
                except:
                    continue
                # 在矩形框上部，输出分类文字
                cv2.putText(frame, emotion_mode, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow("login", frame)
        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
