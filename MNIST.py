
# coding: utf-8

import numpy as np
np.random.seed(12)


import sklearn
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout


(train, train_y), (test, test_y) = mnist.load_data()



train = train.reshape(60000, 784).astype('float32')
test = test.reshape(10000, 784).astype('float32')


train /= 255
test /= 255


number_of_classes = 10
train_y = keras.utils.to_categorical(train_y, number_of_classes)
test_y = keras.utils.to_categorical(test_y, number_of_classes)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


print(model.summary())


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(train, train_y, batch_size=128, epochs=10, verbose=1, validation_data=(test, test_y))

