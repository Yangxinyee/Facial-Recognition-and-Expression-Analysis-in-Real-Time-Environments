# -*- coding = utf-8 -*-
# @Author : XinyeYang
# @File : FaceCNN.py
# @Software : PyCharm
"""-----------------------------------------
1. collect face data set
Obtain 10000 face data sets of my own, 
using dlib to recognize faces, 
although the speed is slower than OpenCV recognition,
the recognition effect is better.
Size: 64*64
-----------------------------------------"""
import cv2
import dlib
import os
import random

faces_add_path = './data/'
size = 64

""" Change image parameters: brightness and contrast """
def img_change(img, light=1, bias=0):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(0, width):
        for j in range(0, height):
            for k in range(3):
                tmp = int(img[j, i, k]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,k] = tmp
    return img

"""Feature extractor :dlib comes with frontal_face_detector"""
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)

num = 1
while True:
    print("Whether to collect new faces（y or n）？")
    if input() == 'y':
        add_user_name = input("Please enter your name:")
        print("Look at the camera.")
        faces_add_path = faces_add_path + add_user_name
        if not os.path.exists(faces_add_path):
            os.makedirs(faces_add_path)
        while (num <= 10000):
            print('Being processed picture %s' % num)
            success, img = cap.read()
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            """ Face detection using Feature Extractor """
            dets = detector(gray_img, 1)
            """-------------------------------------------------------------------------------------------
            The enumerate function is used to traverse the elements in the sequence and their subscripts. 
            i is the face number and d is the element corresponding to i.
            left: The distance between the left side of the face and the left edge of the picture; 
            right: The distance between the right side of the face and the left edge of the image
            top: the distance between the upper part of the face and the upper part of the image; 
            bottom: The distance between the bottom of the face and the top border of the picture
            ------------------------------------------------------------------------------------------------"""
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0

                face = img[x1:y1, x2:y2]
                """ Adjust the contrast and brightness of the picture, 
                the contrast and brightness values are random numbers, 
                so as to increase the diversity of the sample """
                face = img_change(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                face = cv2.resize(face, (size, size))
                cv2.imshow('image', face)
                cv2.imwrite(faces_add_path + '/' + str(num) + '.jpg', face)
                num += 1
            key = cv2.waitKey(30)
            if key == 27:
                break
        else:
            print('Finished!')
            break
    else :
        print("No collection, program over")
        break
