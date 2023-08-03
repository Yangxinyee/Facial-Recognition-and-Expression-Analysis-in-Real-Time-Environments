# -*- coding = utf-8 -*-
# @Author : XinyeYang
# @File : FaceCNN.py
# @Software : PyCharm
"""---------------------------------------------------------------
Second, collect other face data sets
There are Yale face library of Yale University, 
ORL face library of Cambridge University, 
FERET face library of the United States Department of Defense and so on
This system USES face data set download: http://vis-www.cs.umass.edu/lfw/lfw.tgz
First put the downloaded photo set in img_source directory, 
and use dlib to batch identify the face part of the image.
And save to the specified directory faces_other
size: 64*64
----------------------------------------------------------------"""
# -*- codeing: utf-8 -*-
import sys
import cv2
import os
import dlib

source_path = './img_source'
faces_other_path = './data/faces_other'
size = 64
if not os.path.exists(faces_other_path):
    os.makedirs(faces_other_path)

"""Feature extractor :dlib comes with frontal_face_detector"""
detector = dlib.get_frontal_face_detector()

num = 1

for (path, dirnames, filenames) in os.walk(source_path):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % num)
            img_path = path+'/'+filename
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            """Use detector for face detection dets as returned result """
            dets = detector(gray_img, 1)

            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0

                face = img[x1:y1,x2:y2]
                face = cv2.resize(face, (size,size))   # Resize the picture
                cv2.imshow('image',face)
                cv2.imwrite(faces_other_path+'/'+str(num)+'.jpg', face)   #save
                num += 1

            key = cv2.waitKey(30)
            if key == 27:
                sys.exit(0)
