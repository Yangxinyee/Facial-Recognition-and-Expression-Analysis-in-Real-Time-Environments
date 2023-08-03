# -*- coding = utf-8 -*-
# @Author : XinyeYang
# @File : FaceCNN.py
# @Software : PyCharm

import torch.nn as nn

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


def insert_FaceCNN():
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