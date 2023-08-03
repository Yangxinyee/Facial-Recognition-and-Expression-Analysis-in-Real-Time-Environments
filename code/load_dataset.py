# -*- coding = utf-8 -*-
# @Author : XinyeYang
# @File : FaceCNN.py
# @Software : PyCharm

import sys
import numpy as np
import os
import cv2

IMAGE_SIZE = 64
#将输入的图像大小统一
def resize_image(image,height = IMAGE_SIZE,width = IMAGE_SIZE):
    top,bottom,left,right = 0,0,0,0
    #获取图像大小
    h,w,_ = image.shape
    #对于长宽不一的，取最大值
    longest_edge = max(h,w)
    #计算较短的边需要加多少像素
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    #定义填充颜色
    BLACK = [0,0,0]

    #给图像增加边界，使图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant_image = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)

    return cv2.resize(constant_image,(height,width))
#读取数据
images = []     #数据集
labels = []     #标注集
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = path_name + '\\' + dir_item
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            #判断是人脸照片
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image)

                images.append(image)
                labels.append(path_name)

    return images,labels

#为每一类数据赋予唯一的标签值
def label_id(label,users,user_num):
    for i in range(user_num):
        if label.endswith(users[i]):
            return i

#从指定位置读数据
def load_dataset(path_name):
    users = os.listdir(path_name)
    user_num = len(users)

    images,labels = read_path(path_name)
    images_np = np.array(images)
    #每个图片夹都赋予一个固定唯一的标签
    labels_np = np.array([label_id(label,users,user_num) for label in labels])

    return images_np,labels_np

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))
    else:
        images,labels = load_dataset('./data')
        #print(labels)