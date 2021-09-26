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
path_root = 'C:\\Users\\li.7957\\OneDrive - The Ohio State University\\Images for training\\quilts' + '\\'
path_model = r'C:\Users\li.7957\OneDrive - The Ohio State University\Map classification'

path_source0 = path_root

num_maps_class = 6
width = 120
height = 100
num_pixels = width*height
input_size = width*height*3


num_classes = 2
strList = []  # save the strings to be written in files
strTemp = '\n Quilts retuls ' 
strList.append(strTemp)
incorrectImgNameStrList = []

data_pair = []

# Get the image data and store data into X_batches and y_batches
OtherMap_images = os.listdir(path_source0)

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

num_total = num_maps_class
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

    data_pair_3.append(pixel_value_list+[0])


dp3_name = zip(data_pair_3,imgNameList)
dp3_name = list(dp3_name)

len_x = len(data_pair_3[0])-1

random.seed(42)

# strTemp = "train size:"+str(train_size)+' test size:'+str(num_test)
strTemp = "number of iterations:"+str(numIter)
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
    
m = svm_load_model(path_model+'\\'+'svm_model_identification')

p_label, p_acc, p_val = svm_predict(y_test, x_test, m)

strTemp = ' Testing acc:' + str(p_acc[0])
strList.append(strTemp)

# p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
# number of predicted label
count_p_label0 = p_label.count(0.0)
count_p_label1 = p_label.count(1.0)
# number of desired label
count_d_label0 = y_test.count(0)
count_d_label1 = y_test.count(1)
# number of real label
count_r_label0 = 0
count_r_label1 = 0

# collect wrongly classified images
incorrectImgNameStrList.append('\n')
for i in range(len(p_label)):
    if p_label[i] == 0 and y_test[i] == 0:
        count_r_label0 = count_r_label0 + 1
    elif p_label[i] == 1 and y_test[i] == 1:
        count_r_label1 = count_r_label1 + 1
    else:
        imgName = imgNameList[i ]
        incorrectImgString = '\n' + imgName + ',' + str(y_test[i]) + ',' + str(p_label[i])
        incorrectImgNameStrList.append(incorrectImgString)
    
    # precise for the four classes
# precise = []
# if count_p_label0 == 0:
#     precise.append(-1)
# else:
#     precise.append(count_r_label0/count_p_label0)

# if count_p_label1 == 0:
#     precise.append(-1)
# else:
#     precise.append(count_r_label1/count_p_label1)

# strTemp = " Precise:"
# strList.append(strTemp)
# strTemp = ' '
# for p in precise:
#     strTemp = strTemp + str(p)+','
# strList.append(strTemp)
    

#     # recall for the four classes
# recall = []
# recall.append(count_r_label0 / count_d_label0)
# recall.append(count_r_label1 / count_d_label1)

# strTemp = " Recall:"
# strList.append(strTemp)
# strTemp = ' '
# for r in recall:
#     strTemp = strTemp + str(r)+','
# strList.append(strTemp)

# # recall for the four classes   
# F1score = []
# if precise[0] == -1 or precise[0] == 0 or recall[0] == 0:
#     F1score.append(-1)
# else:
#     F1score.append(2/((1/precise[0])+(1/recall[0])))
# if precise[1] == -1 or precise[1] == 0 or recall[1] == 0:
#     F1score.append(-1)
# else:
#     F1score.append(2/((1/precise[1])+(1/recall[1])))

# strTemp = " F1 Score:"
# strList.append(strTemp)
# strTemp = ''
# for f1 in F1score:
#     strTemp = strTemp + str(f1)+','
# strList.append(strTemp)

filename = 'SVMforIdentification2_carto_7_19_2021'+'.txt'
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
