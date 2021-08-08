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
path_root = 'C:\\Users\\li.7957\\OneDrive - The Ohio State University\\Images for training\\maps for classification of projections\\'
# path_root = 'C:\\Users\\jiali\\OneDrive\\Images for training\\maps for classification of projections\\'
path_source0 = path_root + 'Other_Projections_Maps\\'
path_source1 = path_root+'Equirectangular_Projection_Maps\\'
path_source2 = path_root+'Mercator_Projection_Maps\\'
path_source3 = path_root+'EqualArea_Projection_Maps\\'
path_source4 = path_root+'Robinson_Projection_Maps\\'
path_source5 = path_root+'Horizontal rotated maps Antarctica cea\\270\\'
# img = Image.open('C:\\Users\\jiali\\OneDrive\\Images for training\\maps for classification of projections\\Equirectangular_Projection_Maps\\equirectangular_projection_map1.jpg')
# path_source5='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'

# num_notmap=60
# num_map=80
num_maps_class = 250

width = 120
height = 100
num_pixels = width*height
input_size = width*height*3

data_pair = []
num_classes = 5

# Get the image data and store data into X_batches and y_batches
OtherProjection_images = os.listdir(path_source0)
Equirectangular_images = os.listdir(path_source1)
Mercator_images = os.listdir(path_source2)
EqualArea_images = os.listdir(path_source3)
Robinson_images = os.listdir(path_source4)

rotatedImgList = []
rotated_images = os.listdir(path_source5)

# Read map images from other projections
count = 0
imgNameList = []
for imgName in OtherProjection_images:
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
for imgName in Equirectangular_images:
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
for imgName in Mercator_images:
    imgNameList.append(imgName)
    img = Image.open(path_source2 + imgName, 'r')
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break

count = 0
for imgName in EqualArea_images:
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
for imgName in Robinson_images:
    imgNameList.append(imgName)
    img = Image.open(path_source4 + imgName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break
for rotatedImg in rotated_images:
    img = Image.open(path_source5 + rotatedImg)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    rotatedImgList.append(pixel_values)

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
    elif i >= num_maps_class*4 and i < num_maps_class*5:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[4])

rotatedImgList_3 = []
numRotatedImg = len(rotatedImgList)
# numRotatedImg = 10
for i in range(numRotatedImg):
    pixel_value_list = []
    for j in range(num_pixels):
        # print("j:",j)
        pixels = rotatedImgList[i][j]
        pixel_value_list.append(pixels[0])
        pixel_value_list.append(pixels[1])
        pixel_value_list.append(pixels[2])
    rotatedImgList_3.append(pixel_value_list+[1])

dp3_name = zip(data_pair_3,imgNameList)
dp3_name = list(dp3_name)

len_x = len(data_pair_3[0])-1
# Shuffle data_pair as input of Neural Network
random.seed(42)
strList = []  # save the strings to be written in files

for inx in range(1):
    print('Index of sets is: ', inx)
    strTemp = "sets of experiments" + str(inx)
    strList.append(strTemp)
    X_batches = []
    y_batches = []

    X_rotated_batches = []
    y_rotated_batches = []

    random.shuffle(dp3_name)
    data_pair_3, imgNameList = zip(*dp3_name)
    data_pair = np.array(data_pair_3)
    rotatedImgList = np.array(rotatedImgList_3)

    X_batches_255 = [data_pair_3[i][0:len_x] for i in range(num_total)]
    y_batches = [data_pair_3[i][len_x] for i in range(num_total)]
    X_rotated_255 = [rotatedImgList_3[i][0:len_x] for i in range(numRotatedImg)]
    y_rotated = [rotatedImgList_3[i][len_x] for i in range(numRotatedImg)]


    for i in range(numRotatedImg):
            X_rotated_1img = [X_rotated_255[i][j]/255.0 for j in range(len_x)]
            X_rotated_batches.append(X_rotated_1img)


    x_rotate_test_array = X_rotated_batches
    y_rotate_test_array = y_rotated

    y_rotate_test = y_rotate_test_array

    x_rotate_test = [{j: x_rotate_test_array[i][j]
               for j in range(input_size)} for i in range(numRotatedImg)]
    num_rotate_test = numRotatedImg
    
    path_model = r'C:\Users\li.7957\OneDrive - The Ohio State University\Map classification'
    m = svm_load_model(path_model + '\\' + 'svm_model_projection_2')


    print(' Testing acc:')
    start_test = time.time()
    p_label, p_acc, p_val = svm_predict(y_rotate_test, x_rotate_test, m)
    end_test = time.time()
    test_time = end_test - start_test
    strTemp = '\nShifted Testing acc:' + str(p_acc[0])
    strList.append(strTemp)

filename='SVM_Shifted_projection_7_24_2021'+'.txt'
file = open(filename,'a')
file.writelines(strList)
file.close() 

    

# alpha=64
# c=64
# r=2
# Part 2: Classification using RBF kernel SVM
#     acc_c_list = []
#     for c in c_list:
#         acc_alpha_list = []
#         for alpha in alpha_list:
#             print('value of c is: ', c)
#             print('value of alpha is: ', alpha)
#             param = svm_parameter('-t 2 -v 5 -h 0 -g '+str(alpha)+' -c '+str(c))
#             m = svm_train(prob, param)
#             acc_alpha_list.append(m)
#         acc_c_list.append(acc_alpha_list)

#     index = np.argmax(acc_c_list)
#     index_c = index//13
#     index_alpha = (index - 13*index_c)
#     c = c_list[index_c]
#     alpha = alpha_list[index_alpha]
#     print('\n value of c is: ', c)
#     str2 = '\n value of c is:  ' + str(c)
#     file.write(str2)
#     print('\n value of alpha is: ', alpha)
#     str3 = '\n value of alpha is:  ' + str(alpha)
#     file.write(str3)
#     param = svm_parameter('-t 2 -h 0 -g '+str(alpha)+' -c '+str(c))
#     m = svm_train(prob, param)
#     # column=index%13
#     print('\nTraining acc:')
#     p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
#     str4 = '\nTraining acc:' + str(p_acc[0])
#     print('Testing acc:')
#     p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
#     str5 = '\nTesting acc:' + str(p_acc[0]) + "\n"

#     file.write(str4)
#     file.write(str5)
# file.close()

# Part 3: Classification using polynomial SVM
# acc_c_list=[]
# for c in c_list:
#     acc_alpha_list=[]
#     for alpha in alpha_list:
#         acc_r_list=[]
#         for r in r_list:
#             print('\n value of c is: ',c)
#             print('value of alpha is: ',alpha)
#             print('value of r is: ',r)
#             param = svm_parameter('-t 1 -v 5 -h 0 -g '+str(alpha)+' -c '+str(c)+' -r '+str(r))
#             m = svm_train(prob, param)
#             acc_r_list.append(m)
#         acc_alpha_list.append(acc_r_list)
#     acc_c_list.append(acc_alpha_list)

# index=np.argmax(acc_c_list)
# index_c=index//169
# index_alpha=(index - 169*index_c) // 13
# index_r = (index - 169*index_c - 13 * index_alpha)
# c=c_list[index_c]
# alpha=alpha_list[index_alpha]
# r=r_list[index_r]
# print('\n value of c is: ',c)
# print('value of alpha is: ',alpha)
# print('value of r is: ',r)
# param = svm_parameter('-t 1 -h 0 -g '+str(alpha)+' -c '+str(c)+' -r '+str(r))
# m = svm_train(prob, param)
# # column=index%13
# print('Training acc:')
# p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
# print('Testing acc:')
# p_label, p_acc, p_val = svm_predict(y_test, x_test, m)


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
