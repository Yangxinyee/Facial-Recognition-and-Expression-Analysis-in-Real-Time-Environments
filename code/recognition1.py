# -*- coding = utf-8 -*-
# @Author : XinyeYang
# @File : FaceCNN.py
# @Software : PyCharm

import sys
import os
from faces_train import Model
import cv2
import dlib
import torch
import torch.nn as nn
import numpy as np
from statistics import mode

def preprocess_input(images):
    """ preprocess input by substracting the train mean
    # Arguments: images or image of any shape
    # Returns: images or image with substracted train mean (129)
    """
    images = images / 255.0
    return images


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # If a string cannot be found, -1 is returned. 
    # If the value is not equal to -1, the string contains the character
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


class FaceCNN(nn.Module):
    # Initialize the network structure
    def __init__(self):
        super(FaceCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(num_features=64), 
            nn.RReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y


classification_model_path = 'D:\\Anaconda\\envs\\Tensorflow\\face\\model\\model_cnn.pkl'
# Load the expression recognition model
emotion_classifier = torch.load(classification_model_path)
frame_window = 10
# Emoji tag
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
emotion_window = []


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)
    # Loading model
    model = Model()
    model.load_model(file_path='./model/aggregate.face.model.h5')
    # The color of the rectangular border that frames the face
    color = (0, 255, 0)
    # Captures a live video stream from a specified camera
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    # Cyclic detection and recognition of faces
    while True:
        # Read camera
        ret, frame = cap.read()  # Read a frame of video
        if ret is True:
            # Image graying reduces computational complexity
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        dets = detector(frame_gray, 1)
        # if not len(dets):
        #     key = cv2.waitKey(10)
        #     if key == 27:
        #         sys.exit(0)
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            """人脸大小64*64"""
            face = frame[x1:y1, x2:y2]
            face = cv2.resize(face, (64, 64))
            faceID = model.face_predict(face)
            cv2.rectangle(frame, (x2- 10, x1 - 10), (y2+ 10, y1 + 10), color, thickness=2)
            # face id judgment
            for i in range(len(os.listdir('./data/'))):
                if i == faceID:
                    # Text prompt identity
                    cv2.putText(frame, os.listdir('./data/')[i],
                                (x2 + 30, x1 + 30),  # coordinate
                                cv2.FONT_HERSHEY_SIMPLEX,  # typeface
                                1,  # word size
                                (255, 0, 255),  # color
                                2)  # The line width of the word
                if faceID == -1:
                    cv2.putText(frame, 'faces_other',
                                (x2 + 30, x1 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 0, 255),
                                2)

            image_ = frame_gray[x1: y1, x2: y2]
            face = cv2.resize(image_, (48, 48))  #
            # Expand the dimension, shape becomes (1,48,48,1)
            # Convert (1,48,48,1) to (1,1,48,48)
            face = np.expand_dims(face, 0)
            face = np.expand_dims(face, 0)
            # Face data is normalized, mapping pixel values from 0-255 to 0-1
            face = preprocess_input(face)
            new_face = torch.from_numpy(face)
            new_new_face = new_face.float().requires_grad_(False)
            # import trained expression recognition model, predict classification
            emotion_arg = np.argmax(emotion_classifier.forward(new_new_face).detach().numpy())
            emotion = emotion_labels[emotion_arg]
            emotion_window.append(emotion)
            if len(emotion_window) >= frame_window:
                emotion_window.pop(0)
            try:
                # Get the categories that appear the most
                emotion_mode = mode(emotion_window)
            except:
                continue
            # At the top of the rectangular box, output the classified text
            cv2.putText(frame, emotion_mode, (x2, x1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow("rec", frame)

        # Wait 10 milliseconds to see if there is a key input
        k = cv2.waitKey(10)
        # If q is entered, the loop exits
        if k & 0xFF == ord('q'):
            break

        # Release the camera and destroy all Windows
    cap.release()
    cv2.destroyAllWindows()
