# MLP using keras
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pylab as plt
from PIL import Image
import random
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
import pickle



num_maps_class=100
width=120
height=100
num_pixels=width*height
input_size=width*height*3
input_shape=(width, height, 3)
num_classes = 4

# point_generate_random(num_points,num_pixel)
with open('C:\\Users\\li.7957\\OneDrive\\Images for training\\region classification images for experiments\\test_classification_regions.pickle', 'rb') as file:
    [x_test, y_test] = pickle.load(file)
with open('C:\\Users\\li.7957\\OneDrive\\Images for training\\region classification images for experiments\\train_classification_regions.pickle', 'rb') as file:
    [x_train, y_train] = pickle.load(file)

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()
lr = 0.01
str1 = "128 - 256 - 512 - 1024 \n"
filename='Results_CNN_region'+'1'+'.txt'
file = open(filename,'a')
for inx in range(1):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',
                    input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(1024, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.SGD(lr=0.01),
                metrics=['accuracy'])

    batch_size = 5
    # num_classes = 10
    epochs = 100

    # model.fit(x_train, y_train,
    #         epochs=200,
    #         batch_size=10,verbose=1)
    
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test),
          callbacks=[history])

    # score = model.evaluate(x_test, y_test, batch_size=10)
    score = model.evaluate(x_test, y_test, verbose=0)
    train_acc = history.acc[epochs - 1]
    test_loss=score[0]
    test_acc=score[1]
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)
    str2 = 'Training accuracy:' + \
        str(train_acc) + ' Test accuracy:' + str(test_acc) + '\n'
    file.write(str1)
    file.write(str2)

    y=model.predict(x_test)
    print(y)
    print(score)
    
file.close() 


