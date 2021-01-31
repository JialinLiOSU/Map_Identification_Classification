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
numIter = 10
path_root = 'C:\\Users\\li.7957\\OneDrive - The Ohio State University\\Images for training\\region classification images for experiments\\Cartograms\\equalArea\\iter' \
                + str(numIter) + '\\'
path_model = r'C:\Users\li.7957\OneDrive - The Ohio State University\Map classification'

path_source0 = path_root + 'other\\'
path_source1 = path_root+'china\\'
path_source2 = path_root+'sk\\'
path_source3 = path_root+'us\\'
path_source4 = path_root+'world\\'

num_maps_class = 60
width = 120
height = 100
num_pixels = width*height
input_size = width*height*3


num_classes = 2
strList = []  # save the strings to be written in files
incorrectImgNameStrList = []

data_pair = []

# Get the image data and store data into X_batches and y_batches
OtherMap_images = os.listdir(path_source0)
ChinaMap_images = os.listdir(path_source1)
SKoreaMap_images = os.listdir(path_source2)
USMap_images = os.listdir(path_source3)
WorldMap_images = os.listdir(path_source4)

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
    if count >= num_maps_class:
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
    if count >= num_maps_class:
        break

count = 0
for imgName in SKoreaMap_images:
    imgNameList.append(imgName)
    img = Image.open(path_source2 + imgName, 'r')
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= num_maps_class:
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
    if count >= num_maps_class:
        break

count = 0
for imgName in WorldMap_images:
    imgNameList.append(imgName)
    img = Image.open(path_source4 + imgName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= num_maps_class:
        break

num_total = num_maps_class*5
data_pair_3 = []
for i in range(num_total):
    pixel_value_list = []
    for j in range(num_pixels):
        # print("j:",j)
        pixels = data_pair[i][j]
        try:
            pixel_value_list.append(pixels[0])
            pixel_value_list.append(pixels[1])
            pixel_value_list.append(pixels[2])
        except:
            print("i:",i)
            break
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

dp3_name = zip(data_pair_3,imgNameList)
dp3_name = list(dp3_name)

len_x = len(data_pair_3[0])-1


# strTemp = "train size:"+str(train_size)+' test size:'+str(num_test)
strTemp = "number of iterations:"+str(20)
strList.append(strTemp)

X_batches = []
y_batches = []

random.shuffle(dp3_name)
data_pair_3, imgNameList = zip(*dp3_name)
data_pair = np.array(data_pair_3)
    

X_batches_255 = [data_pair_3[i][0:len_x] for i in range(num_total)]
y_batches = [data_pair_3[i][len_x] for i in range(num_total)]
for i in range(num_total):
    X_1img = [X_batches_255[i][j]/255.0 for j in range(len_x)]
    X_batches.append(X_1img)

x_test_array = X_batches[0:num_total]
y_test_array = y_batches[0:num_total]

y_test = y_test_array
x_test = [{j: x_test_array[i][j]
            for j in range(input_size)} for i in range(num_total)]
num_test = len(y_test)
    
m = svm_load_model(path_model+'\\'+'svm_model_region')

p_label, p_acc, p_val = svm_predict(y_test, x_test, m)

strTemp = ' Testing acc:' + str(p_acc[0])
strList.append(strTemp)

# p_label, p_acc, p_val = svm_predict(y_test, x_test, m)


filename = 'SVMforIdentification2_1_27_cg'+'.txt'
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
