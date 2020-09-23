# CSE 5526 Programming Assignment 3 SVM #
# There are two parts in this project
# When you execute part 1, you can comment part 2 and vice versa
from libsvm.python.svm import *
from libsvm.python.svmutil import *
import random
import numpy as np
from PIL import Image
import os
import time

# get the training data
path_root = 'C:\\Users\\li.7957\\OneDrive - The Ohio State University\\Images for training\\region classification images for experiments\\'
# path_root = 'C:\\Users\\jiali\\OneDrive\\Images for training\\maps for classification of projections\\'
path_source0 = path_root + 'Other maps\\'
path_source1 = path_root+'China maps\\'
path_source2 = path_root+'South Korea maps\\'
path_source3 = path_root+'US maps\\'
path_source4 = path_root+'world maps\\'
path_source5 = path_root+'Cartograms\\pop2007_12_china\\'
# img = Image.open('C:\\Users\\jiali\\OneDrive\\Images for training\\maps for classification of projections\\Equirectangular_Projection_Maps\\equirectangular_projection_map1.jpg')
# path_source5='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'

# num_notmap=60
# num_map=80
num_maps_class = 250

width = 120
height = 100
num_pixels = width*height
input_size = width*height*3

num_classes = 5
strList = []  # save the strings to be written in files
incorrectImgNameStrList = []

data_pair = []

# Get the image data and store data into X_batches and y_batches
OtherMap_images = os.listdir(path_source0)
ChinaMap_images = os.listdir(path_source1)
SKoreaMap_images = os.listdir(path_source2)
USMap_images = os.listdir(path_source3)
WorldMap_images = os.listdir(path_source4)

cartoImgList = []
carto_images = os.listdir(path_source5)

count = 0
imgNameList = []
for imgName in OtherMap_images:
    imgNameList.append(imgName)
    fullName = path_source0 + imgName
    img = Image.open(fullName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break

count = 0
# imgNameList = []
for imgName in ChinaMap_images:
    imgNameList.append(imgName)
    fullName = path_source1 + imgName
    img = Image.open(fullName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break

count = 0
for imgName in SKoreaMap_images:
    imgNameList.append(imgName)
    img = Image.open(path_source2 + imgName, 'r')
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break

count = 0
for imgName in USMap_images:
    imgNameList.append(imgName)
    img = Image.open(path_source3 + imgName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    # if len(data_pair)==251:
    #     print(imgName)
    if count >= 250:
        break

count = 0
for imgName in WorldMap_images:
    imgNameList.append(imgName)
    img = Image.open(path_source4 + imgName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break

for cartoImg in carto_images:
    img = Image.open(path_source5 + cartoImg)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    cartoImgList.append(pixel_values)

num_total = num_maps_class*num_classes
train_size = 1000

data_pair_3 = []
for i in range(num_total):
    pixel_value_list = []
    for j in range(num_pixels):
        # print("j:",j)
        pixels = data_pair[i][j]
        pixel_value_list.append(pixels[0])
        pixel_value_list.append(pixels[1])
        pixel_value_list.append(pixels[2])
    if i < num_maps_class:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[0])
    elif i >= num_maps_class and i < num_maps_class*2:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[1])
    elif i >= num_maps_class*2 and i < num_maps_class*3:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[2])
    elif i >= num_maps_class*3 and i < num_maps_class*4:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[3])
    elif i>=num_maps_class*4 and i < num_maps_class*5:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[4])

cartoImgList_3 = []
numCartoImg = len(cartoImgList)
# numRotatedImg = 10
for i in range(numCartoImg):
    pixel_value_list = []
    for j in range(num_pixels):
        # print("j:",j)
        pixels = cartoImgList[i][j]
        pixel_value_list.append(pixels[0])
        pixel_value_list.append(pixels[1])
        pixel_value_list.append(pixels[2])
    cartoImgList_3.append(pixel_value_list+[1])


dp3_name = zip(data_pair_3,imgNameList)
dp3_name = list(dp3_name)

len_x = len(data_pair_3[0])-1
train_size=1000
num_test=num_total-train_size
strTemp = "region: " + "China" + "; number of iterations: " + str(12)
strList.append(strTemp)
# Shuffle data_pair as input of Neural Network
# random.seed(42)

for inx in range(3):
    print('Index of sets is: ', inx)
    strTemp = "sets of experiments" + str(inx)
    strList.append(strTemp)
    X_batches = []
    y_batches = []

    X_carto_batches = []
    y_carto_batches = []

    random.shuffle(dp3_name)
    data_pair_3, imgNameList = zip(*dp3_name)
    data_pair = np.array(data_pair_3)
    cartoImgList = np.array(cartoImgList_3)

    X_batches_255 = [data_pair_3[i][0:len_x] for i in range(num_total)]
    y_batches = [data_pair_3[i][len_x] for i in range(num_total)]
    X_carto_255 = [cartoImgList_3[i][0:len_x] for i in range(numCartoImg)]
    y_carto = [cartoImgList_3[i][len_x] for i in range(numCartoImg)]

    for i in range(num_total):
        X_1img = [X_batches_255[i][j]/255.0 for j in range(len_x)]
        X_batches.append(X_1img)
    
    for i in range(numCartoImg):
        X_carto_1img = [X_carto_255[i][j]/255.0 for j in range(len_x)]
        X_carto_batches.append(X_carto_1img)

    x_train_array = X_batches[0:train_size]
    x_test_array = X_batches[train_size:num_total]
    y_train_array = y_batches[0:train_size]
    y_test_array = y_batches[train_size:num_total]

    x_carto_test_array = X_carto_batches
    y_carto_test_array = y_carto

    y_train = y_train_array
    y_test = y_test_array
    y_carto_test = y_carto_test_array

    x_train = [{j: x_train_array[i][j]
                for j in range(input_size)} for i in range(train_size)]
    x_test = [{j: x_test_array[i][j]
               for j in range(input_size)} for i in range(num_total-train_size)]
    x_carto_test = [{j: x_carto_test_array[i][j]
               for j in range(input_size)} for i in range(numCartoImg)]

    num_train = len(y_train)
    num_test = len(y_test)
    num_carto_test = numCartoImg

    strTemp = "\ntrain size:"+str(train_size)+' test size:'+str(num_test)
    strList.append(strTemp)
    # print('training set:',num_train)
    # print('testing set:',num_test)
    c_list = [2**(i-4) for i in range(0, 13)]
    alpha_list = [2**(i-4) for i in range(0, 13)]
    r_list = [2**(i-4) for i in range(0, 13)]

    prob = svm_problem(y_train, x_train)
    
# Part 3: Classification using polynomial SVM
    strTemp = "\nPolynomial kernel: "
    strList.append(strTemp)
    
    c=0.125
    alpha=0.125
    r=256

    param = svm_parameter('-t 1 -h 0 -g '+str(alpha)+' -c '+str(c)+' -r '+str(r))
    m = svm_train(prob, param)
    # column=index%13
    print('\nTraining acc:')
    p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
    strTemp = ' Training acc:' + str(p_acc[0])
    strList.append(strTemp)
    
    print(' Testing acc:')
    start_test = time.time()
    p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
    end_test = time.time()
    test_time = end_test - start_test
    strTemp = ' Testing acc:' + str(p_acc[0])
    strList.append(strTemp)

    print(' Testing acc:')
    start_test = time.time()
    p_label, p_acc, p_val = svm_predict(y_carto_test, x_carto_test, m)
    end_test = time.time()
    test_time = end_test - start_test
    strTemp = '\nCarto Testing acc:' + str(p_acc[0])
    strList.append(strTemp)
    

filename = 'SVM_region_cartoTest'+'.txt'
file = open(filename, 'a')
file.writelines(strList)
file.writelines(incorrectImgNameStrList)
file.close()

# Part 4: Classification using sigmoid kernel SVM
# acc_c_list = []
# for c in c_list:
#     acc_alpha_list = []
#     for alpha in alpha_list:
#         acc_r_list = []
#         for r in r_list:
#             print('\n value of c is: ', c)
#             print('value of alpha is: ', alpha)
#             print('value of r is: ', r)
#             param = svm_parameter(
#                 '-t 3 -v 5 -h 0 -g '+str(alpha)+' -c '+str(c)+' -r '+str(r))
#             m = svm_train(prob, param)
#             acc_r_list.append(m)
#         acc_alpha_list.append(acc_r_list)
#     acc_c_list.append(acc_alpha_list)

# index = np.argmax(acc_c_list)
# index_c = index//169
# index_alpha = (index - 169*index_c) // 13
# index_r = (index - 169*index_c - 13 * index_alpha)
# c = c_list[index_c]
# alpha = alpha_list[index_alpha]
# r = r_list[index_r]
# print('\n value of c is: ', c)
# print('value of alpha is: ', alpha)
# print('value of r is: ', r)
# param = svm_parameter('-t 3 -h 0 -g '+str(alpha) +
#                       ' -c '+str(c)+' -r '+str(r))
# m = svm_train(prob, param)
# # column=index%13
# print('Training acc:')
# p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
# print('Testing acc:')
# p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
