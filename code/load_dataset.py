# -*- coding = utf-8 -*-
# @Author : XinyeYang
# @File : FaceCNN.py
# @Software : PyCharm

import sys
import numpy as np
import os
import cv2

IMAGE_SIZE = 64
# The size of the input image is uniform
def resize_image(image,height = IMAGE_SIZE,width = IMAGE_SIZE):
    top,bottom,left,right = 0,0,0,0
    # Get image size
    h,w,_ = image.shape
    # For different lengths, take the maximum value
    longest_edge = max(h,w)
    # Calculate how many pixels you need to add to the shorter edge
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
    # Define fill color
    BLACK = [0,0,0]

    # Add a border to the image so that the length and width of the image are equal, 
    # cv2.BORDER_CONSTANT specifies that the border color is specified by value.
    constant_image = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)

    return cv2.resize(constant_image,(height,width))
# reading data
images = []
labels = []
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = path_name + '\\' + dir_item
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            # It's a face photo
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image)

                images.append(image)
                labels.append(path_name)

    return images,labels

# Assign a unique label value to each class of data
def label_id(label,users,user_num):
    for i in range(user_num):
        if label.endswith(users[i]):
            return i

# Reads data from a specified location
def load_dataset(path_name):
    users = os.listdir(path_name)
    user_num = len(users)

    images,labels = read_path(path_name)
    images_np = np.array(images)
    # Each folder is assigned a fixed and unique label
    labels_np = np.array([label_id(label,users,user_num) for label in labels])

    return images_np,labels_np

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))
    else:
        images,labels = load_dataset('./data')
        #print(labels)
