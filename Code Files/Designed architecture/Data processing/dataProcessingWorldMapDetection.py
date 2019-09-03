import numpy as np
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pylab as plt
from PIL import Image
import random
import pickle

# get the training data
# path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
path_source1 = 'C:\\Users\\jiali\\OneDrive\\Images for training\\map identification_world maps\\train\\'
path_source2 = 'C:\\Users\\jiali\\OneDrive\\Images for training\\map identification_world maps\\test\\'
num_notmap = 500
num_map = 500
num_train = 700
num_test = 300
str1 = "train size:"+str(num_train)+' test size:'+str(num_test)+'\n'
num_total = num_map+num_notmap
# this size is for transfer learning (VGG16)
width = 224
height = 224
# width = 120
# height = 100
num_pixels = width*height
input_size = width*height*3
input_shape = (width, height, 3)
num_classes = 2


# Get the image data and store data into x_? and y_?
def dataCollector(path_source1):
    train_images_0 = os.listdir(path_source1+'0')
    train_images_1 = os.listdir(path_source1+'1')
    num_images = len(train_images_0) + len(train_images_1)
    trainImages = []
    for imgName in train_images_0:
        img = Image.open(path_source1 + '0\\' + imgName)
        img_resized = img.resize((width, height), Image.ANTIALIAS)
        pixel_values = list(img_resized.getdata())
        trainImages.append(pixel_values)

    for imgName in train_images_1:
        img = Image.open(path_source1 + '1\\' + imgName)
        img_resized = img.resize((width, height), Image.ANTIALIAS)
        pixel_values = list(img_resized.getdata())
        trainImages.append(pixel_values)

    trainImages3 = []
    # get the 3 channels of images and add a label
    for i in range(num_images):
        pixel_value_list = []
        for j in range(num_pixels):
            pixels = trainImages[i][j]
            pixel_value_list.append(pixels[0])
            pixel_value_list.append(pixels[1])
            pixel_value_list.append(pixels[2])
        if i <= len(train_images_0):
            # print(len(pixel_value_list))
            trainImages3.append(pixel_value_list+[0]+[i])
        else:
            # print(len(pixel_value_list))
            trainImages3.append(pixel_value_list+[1]+[i])

    len_x = len(trainImages3[0])-2
    inx_y = len_x+1
    inx_image = inx_y+1

    X_batches = []
    y_batches = []
    # for i in range(num_total):
    #     print(len(data_pair_3[i]))
    data_pair_train = np.array(trainImages3)

    # print(len_x)
    X_batches_255 = [trainImages3[i][0:len_x] for i in range(num_images)]
    # for j in range(num_total):
    # print(len(data_pair_3[j])-1)
    # print(data_pair_3[j][len(data_pair_3[j])-1])
    y_batches = [trainImages3[i][len_x] for i in range(num_images)]
    # data get from last step is with the total value of pixel 255

    for i in range(num_images):
        X_1img = [X_batches_255[i][j]/255.0 for j in range(len_x)]
        X_batches.append(X_1img)
    X_batches = np.array(X_batches)
    y_batches = np.array(y_batches)
    x_train = X_batches.reshape(num_images, input_size)
    y_train = y_batches.reshape(num_images, 1)
    x_train = x_train.reshape(x_train.shape[0], width, height, 3)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    return x_train, y_train


x_train, y_train = dataCollector(path_source1)
x_test, y_test = dataCollector(path_source2)
#save train and test data into pickle files
f1 = open('train_identification_world_transferlearning.pickle', 'wb')
f2 = open('test_identification_world_transferlearning.pickle', 'wb')
pickle.dump([x_train, y_train], f1)
pickle.dump([x_test, y_test], f2)
f1.close()
f2.close()