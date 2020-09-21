
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv3D, BatchNormalization, Activation, MaxPool2D, MaxPool3D, Dropout, Flatten, Dense, LSTM, GRU, BatchNormalization, LayerNormalization
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

frameLen = 5
class visionModel(Model):
    def __init__(self):
        # 不指定inputshape仍然work
        super(visionModel, self).__init__()
        self.c1 = Conv3D(filters=64, kernel_size=(2,5,5), strides=(1, 1, 1), input_shape=(frameLen, 72, 128, 3), padding='same')  # 卷积层
        # self.c1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1)  # 卷积层
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv3D(filters=64, kernel_size=(2,5,5), strides=(1,1,1), padding='same')  # 卷积层
        # self.c2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')  # 卷积层
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')

        self.p1 = MaxPool3D(pool_size=(2, 3, 3), strides=(2, 2, 2), padding='same')  # 池化层
        # self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)

        self.c3 = Conv3D(filters=128, kernel_size=(2,5,5), padding='same')  # 卷积层
        # self.c3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=1)  # 卷积层
        self.b3 = BatchNormalization()
        self.a3 = Activation('relu')

        self.c4 = Conv3D(filters=128, kernel_size=(2,5,5), padding='same')  # 卷积层
        self.b4 = BatchNormalization()
        self.a4 = Activation('relu')

        self.p2 = MaxPool3D(pool_size=(2, 3, 3), strides=(2, 2, 2), padding='same')  # 池化层
        self.d2 = Dropout(0.2)

        self.c5 = Conv3D(filters=256, kernel_size=(2,5,5), padding='same')  # 卷积层
        # self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=1)  # 卷积层
        self.b5 = BatchNormalization()
        self.a5 = Activation('relu')

        # 先搞5层试试


        self.c6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=1)  # 卷积层
        self.b6 = BatchNormalization()
        self.a6 = Activation('relu')

        self.c7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=1)  # 卷积层
        self.b7 = BatchNormalization()
        self.a7 = Activation('relu')

        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d3 = Dropout(0.2)

        self.c8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=1)  # 卷积层
        self.b8 = BatchNormalization()
        self.a8 = Activation('relu')

        self.c9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=1)  # 卷积层
        self.b9 = BatchNormalization()
        self.a9 = Activation('relu')

        self.c10 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=1)  # 卷积层
        self.b10 = BatchNormalization()
        self.a10 = Activation('relu')

        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d4 = Dropout(0.2)

        self.c11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=1)  # 卷积层
        self.b11 = BatchNormalization()
        self.a11 = Activation('relu')

        self.c12 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=1)  # 卷积层
        self.b12 = BatchNormalization()
        self.a12 = Activation('relu')

        self.c13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=1)  # 卷积层
        self.b13 = BatchNormalization()
        self.a13 = Activation('relu')

        self.p5 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d5 = Dropout(0.2)


        self.gru1 = GRU(1024, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')
        self.layerNor1 = LayerNormalization()
        self.drop1 = Dropout(0.2)
        self.gru2 = GRU(512, return_sequences=False, stateful=True, recurrent_initializer='glorot_uniform')

        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(1, activation='linear')

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu')
        self.d1 = Dropout(0.2)
        self.f2 = Dense(512, activation='relu')
        self.d2 = Dropout(0.2)
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.a1(self.b1(self.c1(x)))
        x = self.a2(self.b2(self.c2(x)))
        x = self.d1(self.p1(x))

        x = self.a3(self.b3(self.c3(x)))
        x = self.a4(self.b4(self.c4(x)))
        x = self.d2(self.p2(x))

        x = self.a5(self.b5(self.c5(x)))

        # x = self.a6(self.b6(self.c6(x)))
        # x = self.a7(self.b7(self.c7(x)))
        # x = self.d3(self.p3(x))

        # x = self.a8(self.b8(self.c8(x)))
        # x = self.a9(self.b9(self.c9(x)))
        # x = self.a10(self.b10(self.c10(x)))
        # x = self.d4(self.p4(x))

        # x = self.a11(self.b11(self.c11(x)))
        # x = self.a12(self.b12(self.c12(x)))
        # x = self.a13(self.b13(self.c13(x)))
        # x = self.d5(self.p5(x))

        x = self.gru2(self.drop1(self.layerNor1(self.gru1(x))))
        y = self.dense2(self.dense1(x))
        # y = self.f3(self.d2(self.f2(self.d1(self.f1(self.flatten(x))))))
        return y


# model = visionModel()
# pic0 = trainX[0]
# pic1 = trainX[1]
# pic = tf.stack([pic0, pic1, pic0, pic0, pic1], axis=0)
# pic = tf.reshape(pic, [1, 5, 72, 128, 3])
#
# y = np.array([[1]])
#
# model(pic).shape
# model.compile(optimizer='adam', loss = 'mean_squared_error')
# history = model.fit(x=pic, y=y, batch_size=1, epochs=30)