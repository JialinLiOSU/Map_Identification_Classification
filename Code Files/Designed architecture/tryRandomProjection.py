# MLP using keras
# In this file, I increased the number of images used to 400 images totally.
# import numpy as np
# import matplotlib.pylab as plt
# from PIL import Image
import random
import time
import os
import pickle

# get the training data
path_root = 'D:\\OneDrive - The Ohio State University\\Images for training\\maps for classification of projections\\'
# path_root = 'C:\\Users\\jiali\\OneDrive\\Images for training\\maps for classification of projections\\'
path_source0 = path_root + 'Other_Projections_Maps\\'
path_source1 = path_root+'Equirectangular_Projection_Maps\\'
path_source2 = path_root+'Mercator_Projection_Maps\\'
path_source3 = path_root+'EqualArea_Projection_Maps\\'
path_source4 = path_root+'Robinson_Projection_Maps\\'

num_maps_class = 250
width = 120
height = 100
num_pixels = width*height
input_size = width*height*3
input_shape = (width, height, 3)

strList = []  # save the strings to be written in files

num_classes = 5

# Get the image data and store data into X_batches and y_batches
data_pair = []
OtherProjection_images = os.listdir(path_source0)
Equirectangular_images = os.listdir(path_source1)
Mercator_images = os.listdir(path_source2)
EqualArea_images = os.listdir(path_source3)
Robinson_images = os.listdir(path_source4)

# Read map images from other projections
count = 0
imgNameList = []
for imgName in OtherProjection_images:
    imgNameList.append(imgName)
    data_pair.append(1)
    count = count + 1
    if count >= num_maps_class:
        break

count = 0
for imgName in Equirectangular_images:
    imgNameList.append(imgName)
    data_pair.append(1)
    count = count + 1
    if count >= num_maps_class:
        break

count = 0
for imgName in Mercator_images:
    imgNameList.append(imgName)
    
    data_pair.append(1)
    count = count + 1
    if count >= num_maps_class:
        break

count = 0
for imgName in EqualArea_images:
    imgNameList.append(imgName)

    data_pair.append(1)
    count = count + 1
    # if len(data_pair)==251:
    #     print(imgName)
    if count >= num_maps_class:
        break

count = 0
for imgName in Robinson_images:
    imgNameList.append(imgName)

    data_pair.append(1)
    count = count + 1
    if count >= num_maps_class:
        break

num_total = num_maps_class*num_classes
# data_pair_temp=[data_pair[i] for i in range(300,400)]
data_pair_3 = []
for i in range(num_total):

    if i < num_maps_class:
        # print(len(pixel_value_list))
        # after pixel values, then class number and index
        data_pair_3.append([1]+[0]+[i])
    elif i >= num_maps_class and i < num_maps_class*2:
        # print(len(pixel_value_list))
        data_pair_3.append([1]+[1]+[i])
    elif i >= num_maps_class*2 and i < num_maps_class*3:
        # print(len(pixel_value_list))
        data_pair_3.append([1]+[2]+[i])
    elif i >= num_maps_class*3 and i < num_maps_class*4:
        # print(len(pixel_value_list))
        data_pair_3.append([1]+[3]+[i])
    elif i>=num_maps_class*4 and i < num_maps_class*5:
        # print(len(pixel_value_list))
        data_pair_3.append([1]+[4]+[i])

dp3_name = zip(data_pair_3,imgNameList)
dp3_name = list(dp3_name)

len_x = len(data_pair_3[0])-2
inx_y = len_x+1
inx_image = inx_y+1
# Shuffle data_pair as input of Neural Network


train_size = int(num_total*0.8)
num_test = num_total-train_size
strTemp = "train size:"+str(train_size)+' test size:'+str(num_test)
strList.append(strTemp)

random.seed(42)
for i in range(20000,50000):
    randomSeed = i
    random.seed(randomSeed)
    random.shuffle(dp3_name)
    data_pair_3, imgNameList = zip(*dp3_name)
    # data_pair = np.array(data_pair_3)
    imgName = imgNameList[train_size]
    if imgName == 'miller_projection_map39.jpg':
        print('that is it!')
        test_img_names = imgNameList[train_size:num_total]
        print('test image names: ')
        print(test_img_names)