# MLP for world map classification using keras
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from PIL import Image
import random
import time
import pickle


# get the training data
# path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
path_source1 = 'C:\\Users\\li.7957\\OneDrive\\Images for training\\region classification images for experiments\\train\\'
path_source2 = 'C:\\Users\\li.7957\\OneDrive\\Images for training\\region classification images for experiments\\test\\'
num_notmap = 500
num_map = 500
num_train = 700
num_test = 300
str1 = "train size:"+str(num_train)+' test size:'+str(num_test)+'\n'
num_total = num_map+num_notmap

width = 120
height = 100
num_pixels = width*height
input_size = width*height*3
input_shape = (width, height, 3)
num_classes = 4

# point_generate_random(num_points,num_pixel)
with open('C:\\Users\\li.7957\\OneDrive\\Images for training\\region classification images for experiments\\test_classification_regions_MLP.pickle', 'rb') as file:
    [x_test, y_test] = pickle.load(file)
with open('C:\\Users\\li.7957\\OneDrive\\Images for training\\region classification images for experiments\\train_classification_regions_MLP.pickle', 'rb') as file:
    [x_train, y_train] = pickle.load(file)


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()
lr = 0.01
beta_1 = 0.9
beta_2 = 0.999

str1 = "2000 - 1000 - 200 - 100 - 4" + "\n"

for inx in range(1):
    model = Sequential()
    model.add(Dense(2000, input_dim=input_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss = keras.losses.binary_crossentropy,
              optimizer = keras.optimizers.SGD(lr = lr),
              metrics=['accuracy'])
    batch_size = 5
    epochs = 100

    y_train = to_categorical(y_train, num_classes=4)
    y_test = to_categorical(y_test, num_classes=4)

    model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,
                    verbose=2,validation_data=(x_test, y_test),
              callbacks=[history])
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    train_acc = history.acc[epochs - 1]
    test_loss=score[0]
    test_acc=score[1]
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    str2 = 'Training accuracy:' + \
        str(train_acc) + ' Test accuracy:' + str(test_acc) + '\n'
# y = model.predict(x_test)
# print(y)
# print(score)
filename='Results_MLP_Region_Classification'+'1'+'.txt'
file = open(filename,'a')
file.write(str1) 
file.write(str2)
file.close() 
