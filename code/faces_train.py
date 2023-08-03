# -*- coding = utf-8 -*-
# @Author : XinyeYang
# @File : FaceCNN.py
# @Software : PyCharm

import random
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
import numpy as np
from load_dataset import load_dataset, resize_image

IMAGE_SIZE = 64


class Dataset:
    def __init__(self, path_name):
        # training set
        self.train_images = None
        self.train_labels = None

        # Validation set
        self.valid_images = None
        self.valid_labels = None

        # test set
        self.test_images = None
        self.test_labels = None

        # Data set load path
        self.path_name = path_name
        # Type of image
        self.user_num = len(os.listdir(path_name))
        # Current dimension order
        self.input_shape = None

    # Load the data set, divide the data set according to the principle of cross-validation, 
    # and carry out related preprocessing
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=3):
        # data category
        nb_classes = self.user_num
        # Load the data set into memory
        images, labels = load_dataset(self.path_name)

        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,
                                                                                  random_state=random.randint(0, 100))
        
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            # test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            # test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

            # Output the number of training sets, verification sets, and test sets
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            # print(test_images.shape[0], 'test samples')

            """Our model uses categorical_crossentropy as a loss function, 
            so it is necessary to vectorize category labels by one-hot encoding 
            according to the number of categories nb_classes. 
            There are only two categories here, and the label data becomes two-dimensional after transformation"""
            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
            # test_labels = np_utils.to_categorical(test_labels, nb_classes)

            # Floating the pixel data for normalization
            train_images = train_images.astype('float32')
            valid_images = valid_images.astype('float32')
            # test_images = test_images.astype('float32')

            # The value of each pixel of the image is normalized to the range of 0~1
            train_images /= 255
            valid_images /= 255
            # test_images /= 255

            self.train_images = train_images
            self.valid_images = valid_images
            # self.test_images = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            # self.test_labels = test_labels


# CNN network model class
class Model:
    def __init__(self):
        self.model = None

        # modelling

    def build_model(self, dataset, nb_classes=4):

        self.model = Sequential()

        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                                     input_shape=dataset.input_shape))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        # Output model summary
        self.model.summary()

    # training model
    def train(self, dataset, batch_size=20, nb_epoch=10, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)  # Training with an optimizer with SGD+momentum starts with generating an optimizer object
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # Complete the actual model configuration

        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True)
        # Use real-time data augmentation
        else:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # Whether to decentralize input data (mean 0)
                samplewise_center=False,  # Whether to set the mean of each sample of input data to 0
                featurewise_std_normalization=False,  # Data normalization (input data divided by the standard deviation of the data set)
                samplewise_std_normalization=False,  # Whether to divide each sample data by its own standard deviation
                zca_whitening=False,  # Whether to apply ZCA whitening to input data
                rotation_range=20,  # The Angle of random rotation of the image when the data is lifted (range 0 ~ 180)
                width_shift_range=0.2,  # The magnitude of the horizontal shift of the picture when the data is raised 
                                        # (expressed as a percentage of the picture width, floating-point number between 0 and 1)
                height_shift_range=0.2,  # Same as above, except this is vertical
                horizontal_flip=True,  # Whether to perform a random horizontal flip
                vertical_flip=False)  # Whether to perform a random vertical flip

            # The number of the whole training sample set is calculated for eigenvalue normalization, ZCA whitening and other processing
            datagen.fit(dataset.train_images)

            # Use the generator to start training the model
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                  batch_size=batch_size),
                                     samples_per_epoch=dataset.train_images.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(dataset.valid_images, dataset.valid_labels))

    MODEL_PATH = './model/aggregate.face.model1.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    # Recognize faces
    def face_predict(self, image):
        if K.image_data_format() == 'channels_first' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # The size must be the same as the training set and should be IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # Unlike the model training, the prediction is only made for one image
        elif K.image_data_format() == 'channels_first' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        image = image.astype('float32')
        image /= 255

        # Gives the probability that the input belongs to each category
        pred = self.model.predict(image)
        result_probability = np.argmax(pred,axis=1)
        # result_probability = self.model.predict_proba(image)
        # print('result:', result_probability)

        # Give category prediction
        if max(result_probability[0]) >= 0.9:
            result = self.model.predict_classes(image)
            print('result:', result)
            # Returns category prediction results
            return result[0]
        else:
            print('result:none')
            return -1


if __name__ == '__main__':
    user_num = len(os.listdir('./data/'))

    dataset = Dataset('./data/')
    dataset.load()

    model = Model()
    model.build_model(dataset, nb_classes=user_num)
    model.build_model(dataset, nb_classes=user_num)
    model.train(dataset)
    model.save_model(file_path='./model.aggregate.face.model1.h5')

