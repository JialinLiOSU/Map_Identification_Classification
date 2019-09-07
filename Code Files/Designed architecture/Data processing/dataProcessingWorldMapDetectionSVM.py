# This file has not been completed
# The pickle files for MLP can be also used for SVM
import random
import numpy as np
from PIL import Image
import time
import pickle
import os

# get the training data
# path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
path = 'C:\\Users\\li.7957\\OneDrive\Images for training\\region classification images for experiments\\'
path_source1 = path + 'train\\'
path_source2 = path + 'test\\'
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
num_classes = 2

# Get the image data and store data into x_? and y_?
def dataCollector(path_source1):
    train_images_1 = os.listdir(path_source1+'1')
    train_images_2 = os.listdir(path_source1+'2')
    train_images_3 = os.listdir(path_source1+'3')
    train_images_4 = os.listdir(path_source1+'4')
    num_images = len(train_images_1) + len(train_images_2) +  len(train_images_3) + len(train_images_4)
    trainImages = []
    for imgName in train_images_1:
        img = Image.open(path_source1 + '1\\' + imgName)
        img_resized = img.resize((width, height), Image.ANTIALIAS)
        pixel_values = list(img_resized.getdata())
        trainImages.append(pixel_values)

    for imgName in train_images_2:
        img = Image.open(path_source1 + '2\\' + imgName)
        img_resized = img.resize((width, height), Image.ANTIALIAS)
        pixel_values = list(img_resized.getdata())
        trainImages.append(pixel_values)

    for imgName in train_images_3:
        img = Image.open(path_source1 + '3\\' + imgName)
        img_resized = img.resize((width, height), Image.ANTIALIAS)
        pixel_values = list(img_resized.getdata())
        trainImages.append(pixel_values)
    
    for imgName in train_images_4:
        img = Image.open(path_source1 + '4\\' + imgName)
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
        if i <= len(train_images_1):
            # print(len(pixel_value_list))
            trainImages3.append(pixel_value_list+[0]+[i])
        elif i <= len(train_images_2)*2:
            trainImages3.append(pixel_value_list+[1]+[i])
        elif i <= len(train_images_3)*3:
            trainImages3.append(pixel_value_list+[2]+[i])
        else:
            # print(len(pixel_value_list))
            trainImages3.append(pixel_value_list+[3]+[i])

    len_x = len(trainImages3[0])-2
    inx_y = len_x+1

    X_batches = []
    y_batches = []

    # print(len_x)
    X_batches_255 = [trainImages3[i][0:len_x] for i in range(num_images)]
    y_batches = [trainImages3[i][len_x] for i in range(num_images)]
    # data get from last step is with the total value of pixel 255

    for i in range(num_images):
        X_1img = [X_batches_255[i][j]/255.0 for j in range(len_x)]
        X_batches.append(X_1img)
        
    x_train_array=X_batches
    y_train=y_batches

    x_train=[{j:x_train_array[i][j] for j in range(input_size)} for i in range(num_images)]
    # x_train = x_train.reshape(x_train.shape[0], width, height, 3)
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    return x_train, y_train

x_train, y_train = dataCollector(path_source1)
x_test, y_test = dataCollector(path_source2)
#save train and test data into pickle files
f1 = open('train_classification_regions_SVM.pickle', 'wb')
f2 = open('test_classification_regions_SVM.pickle', 'wb')
pickle.dump([x_train, y_train], f1)
pickle.dump([x_test, y_test], f2)
f1.close()
f2.close()