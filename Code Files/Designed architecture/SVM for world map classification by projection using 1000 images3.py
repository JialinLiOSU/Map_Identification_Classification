# CSE 5526 Programming Assignment 3 SVM #
# There are two parts in this project
# When you execute part 1, you can comment part 2 and vice versa
from libsvm.python.svm import *
from libsvm.python.svmutil import *
import random
import numpy as np
from PIL import Image
import os

# get the training data
path_root = 'C:\\Users\\li.7957\\OneDrive\\Images for training\\maps for classification of projections\\'
# path_root = 'C:\\Users\\jiali\\OneDrive\\Images for training\\maps for classification of projections\\'
path_source1 = path_root+'Equirectangular_Projection_Maps\\'
path_source2 = path_root+'Mercator_Projection_Maps\\'
path_source3 = path_root+'EqualArea_Projection_Maps\\'
path_source4 = path_root+'Robinson_Projection_Maps\\'
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

# Get the image data and store data into X_batches and y_batches
Equirectangular_images = os.listdir(path_source1)
Mercator_images = os.listdir(path_source2)
EqualArea_images = os.listdir(path_source3)
Robinson_images = os.listdir(path_source4)

count = 0
for imgName in Equirectangular_images:
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
    img = Image.open(path_source2 + imgName, 'r')
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break

count = 0
for imgName in EqualArea_images:
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
    img = Image.open(path_source4 + imgName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break

num_total = num_maps_class*4

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

len_x = len(data_pair_3[0])-1
# Shuffle data_pair as input of Neural Network
# random.seed(42)
filename = 'SVMforProjection4'+'.txt'
file = open(filename, 'a')

for inx in range(1):
    print('Index of sets is: ', inx)
    str1 = 'Index of sets is: ' + str(inx)
    file.write(str1)
    X_batches = []
    y_batches = []

    random.shuffle(data_pair_3)
    data_pair = np.array(data_pair_3)

    X_batches_255 = [data_pair_3[i][0:len_x] for i in range(num_total)]
    y_batches = [data_pair_3[i][len_x] for i in range(num_total)]

    for i in range(num_total):
        X_1img = [X_batches_255[i][j]/255.0 for j in range(len_x)]
        X_batches.append(X_1img)

    train_size = 800

    x_train_array = X_batches[0:train_size]
    x_test_array = X_batches[train_size:num_total]
    y_train_array = y_batches[0:train_size]
    y_test_array = y_batches[train_size:num_total]

    y_train = y_train_array
    y_test = y_test_array
    x_train = [{j: x_train_array[i][j]
                for j in range(input_size)} for i in range(train_size)]
    x_test = [{j: x_test_array[i][j]
               for j in range(input_size)} for i in range(num_total-train_size)]
    num_train = len(y_train)
    num_test = len(y_test)
    # print('training set:',num_train)
    # print('testing set:',num_test)
    c_list = [2**(i-4) for i in range(0, 13)]
    alpha_list = [2**(i-4) for i in range(0, 13)]
    r_list = [2**(i-4) for i in range(0, 13)]

    prob = svm_problem(y_train, x_train)

    # Part 1: Classification using linear SVMs
#     acc_c_list = []
#     for c in c_list:
#         print('value of c is: ', c)
#         param = svm_parameter('-t 0 -v 5 -h 0 -c '+str(c))
#         m = svm_train(prob, param)
#         acc_c_list.append(m)
#     index = np.argmax(acc_c_list)
#     index_c = index
#     # index_alpha=(index - 13*index_c)
#     c = c_list[index_c]
#     # alpha=alpha_list[index_alpha]
#     print('\n value of c is: ', c)
#     str2 = '\n value of c is:  ' + str(c)
#     file.write(str2)

#     # print('value of alpha is: ',alpha)
#     param = svm_parameter('-t 0 -h 0 -c '+str(c))
#     m = svm_train(prob, param)
#     # column=index%13

#     print('\nTraining acc:')
#     p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
#     str3 = '\nTraining acc:' + str(p_acc[0])
#     print('Testing acc:')
#     p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
#     str4 = '\nTesting acc:' + str(p_acc[0]) + "\n"

#     ####     calculate Precise, Recall and F1 score      #####
#     # p_label is the predicted class labels
#     # y_test is the desired class labels

#     # number of predicted label
#     count_p_label0 = p_label.count(0.0)
#     count_p_label1 = p_label.count(1.0)
#     count_p_label2 = p_label.count(2.0)
#     count_p_label3 = p_label.count(3.0)
#     # number of desired label
#     count_d_label0 = y_test.count(0)
#     count_d_label1 = y_test.count(1)
#     count_d_label2 = y_test.count(2)
#     count_d_label3 = y_test.count(3)
#     # number of real label
#     count_r_label0 = 0
#     count_r_label1 = 0
#     count_r_label2 = 0
#     count_r_label3 = 0

#     for i in range(len(p_label)):
#         if p_label[i] == 0 and y_test[i] == 0:
#             count_r_label0 = count_r_label0 + 1
#         elif p_label[i] == 1 and y_test[i] == 1:
#             count_r_label1 = count_r_label1 + 1
#         elif p_label[i] == 2 and y_test[i] == 2:
#             count_r_label2 = count_r_label2 + 1
#         elif p_label[i] == 3 and y_test[i] == 3:
#             count_r_label3 = count_r_label3 + 1
    
#     # precise for the four classes
#     precise = []
#     precise.append(count_r_label0/count_p_label0)
#     precise.append(count_r_label1/count_p_label1)
#     precise.append(count_r_label2/count_p_label2)
#     precise.append(count_r_label3/count_p_label3)
#     file.write("\nPrecise:\n")
#     for p in precise:
#         file.write(str(p)+',')
#     file.write("\n")

#     # recall for the four classes
#     recall = []
#     recall.append(count_r_label0 / count_d_label0)
#     recall.append(count_r_label1 / count_d_label1)
#     recall.append(count_r_label2 / count_d_label2)
#     recall.append(count_r_label3 / count_d_label3)
#     file.write("\nRecall:\n")
#     for r in recall:
#         file.write(str(r)+',')
#     file.write("\n")

#     # recall for the four classes   
#     F1score = []
#     F1score.append(2/((1/precise[0])+(1/recall[0])))
#     F1score.append(2/((1/precise[1])+(1/recall[1])))
#     F1score.append(2/((1/precise[2])+(1/recall[2])))
#     F1score.append(2/((1/precise[3])+(1/recall[3])))
#     file.write("\nF1 Score:\n")
#     for f1 in F1score:
#         file.write(str(f1)+',')
#     file.write("\n")

#     file.write(str3)
#     file.write(str4)
# file.close()

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
#     print('\nTesting acc:')
#     p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
#     str5 = '\nTesting acc:' + str(p_acc[0]) + "\n"

#     ####     calculate Precise, Recall and F1 score      #####
#     # p_label is the predicted class labels
#     # y_test is the desired class labels

#     # number of predicted label
#     count_p_label0 = p_label.count(0.0)
#     count_p_label1 = p_label.count(1.0)
#     count_p_label2 = p_label.count(2.0)
#     count_p_label3 = p_label.count(3.0)
#     # number of desired label
#     count_d_label0 = y_test.count(0)
#     count_d_label1 = y_test.count(1)
#     count_d_label2 = y_test.count(2)
#     count_d_label3 = y_test.count(3)
#     # number of real label
#     count_r_label0 = 0
#     count_r_label1 = 0
#     count_r_label2 = 0
#     count_r_label3 = 0

#     for i in range(len(p_label)):
#         if p_label[i] == 0 and y_test[i] == 0:
#             count_r_label0 = count_r_label0 + 1
#         elif p_label[i] == 1 and y_test[i] == 1:
#             count_r_label1 = count_r_label1 + 1
#         elif p_label[i] == 2 and y_test[i] == 2:
#             count_r_label2 = count_r_label2 + 1
#         elif p_label[i] == 3 and y_test[i] == 3:
#             count_r_label3 = count_r_label3 + 1
    
#     # precise for the four classes
#     precise = []
#     precise.append(count_r_label0/count_p_label0)
#     precise.append(count_r_label1/count_p_label1)
#     precise.append(count_r_label2/count_p_label2)
#     precise.append(count_r_label3/count_p_label3)
#     file.write("\nPrecise:\n")
#     for p in precise:
#         file.write(str(p)+',')
#     file.write("\n")

#     # recall for the four classes
#     recall = []
#     recall.append(count_r_label0 / count_d_label0)
#     recall.append(count_r_label1 / count_d_label1)
#     recall.append(count_r_label2 / count_d_label2)
#     recall.append(count_r_label3 / count_d_label3)
#     file.write("\nRecall:\n")
#     for r in recall:
#         file.write(str(r)+',')
#     file.write("\n")

#     # recall for the four classes   
#     F1score = []
#     F1score.append(2/((1/precise[0])+(1/recall[0])))
#     F1score.append(2/((1/precise[1])+(1/recall[1])))
#     F1score.append(2/((1/precise[2])+(1/recall[2])))
#     F1score.append(2/((1/precise[3])+(1/recall[3])))
#     file.write("\nF1 Score:\n")
#     for f1 in F1score:
#         file.write(str(f1)+',')
#     file.write("\n")

#     file.write(str4)
#     file.write(str5)
# file.close()

# Part 3: Classification using polynomial SVM
#     acc_c_list=[]
#     for c in c_list:
#         acc_alpha_list=[]
#         for alpha in alpha_list:
#             acc_r_list=[]
#             for r in r_list:
#                 print('\n value of c is: ',c)
#                 print('value of alpha is: ',alpha)
#                 print('value of r is: ',r)
#                 param = svm_parameter('-t 1 -v 5 -h 0 -g '+str(alpha)+' -c '+str(c)+' -r '+str(r))
#                 m = svm_train(prob, param)
#                 acc_r_list.append(m)
#             acc_alpha_list.append(acc_r_list)
#         acc_c_list.append(acc_alpha_list)

#     index=np.argmax(acc_c_list)
#     index_c=index//169
#     index_alpha=(index - 169*index_c) // 13
#     index_r = (index - 169*index_c - 13 * index_alpha)
#     c=c_list[index_c]
#     alpha=alpha_list[index_alpha]
#     r=r_list[index_r]
#     print('\n value of c is: ',c)
#     str2 = '\n value of c is:  ' + str(c)
#     file.write(str2)
#     print('value of alpha is: ',alpha)
#     str3 = '\n value of alpha is:  ' + str(alpha)
#     file.write(str3)
#     print('value of r is: ',r)
#     str4 = '\n value of alpha is:  ' + str(r)
#     file.write(str4)
#     param = svm_parameter('-t 1 -h 0 -g '+str(alpha)+' -c '+str(c)+' -r '+str(r))
#     m = svm_train(prob, param)
#     # column=index%13
#     print('Training acc:')
#     p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
#     str5 = '\nTraining acc:' + str(p_acc[0])
#     print('Testing acc:')
#     p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
#     str6 = '\nTesting acc:' + str(p_acc[0]) + "\n"

#     ####     calculate Precise, Recall and F1 score      #####
#     # p_label is the predicted class labels
#     # y_test is the desired class labels

#     # number of predicted label
#     count_p_label0 = p_label.count(0.0)
#     count_p_label1 = p_label.count(1.0)
#     count_p_label2 = p_label.count(2.0)
#     count_p_label3 = p_label.count(3.0)
#     # number of desired label
#     count_d_label0 = y_test.count(0)
#     count_d_label1 = y_test.count(1)
#     count_d_label2 = y_test.count(2)
#     count_d_label3 = y_test.count(3)
#     # number of real label
#     count_r_label0 = 0
#     count_r_label1 = 0
#     count_r_label2 = 0
#     count_r_label3 = 0

#     for i in range(len(p_label)):
#         if p_label[i] == 0 and y_test[i] == 0:
#             count_r_label0 = count_r_label0 + 1
#         elif p_label[i] == 1 and y_test[i] == 1:
#             count_r_label1 = count_r_label1 + 1
#         elif p_label[i] == 2 and y_test[i] == 2:
#             count_r_label2 = count_r_label2 + 1
#         elif p_label[i] == 3 and y_test[i] == 3:
#             count_r_label3 = count_r_label3 + 1
    
#     # precise for the four classes
#     precise = []
#     precise.append(count_r_label0/count_p_label0)
#     precise.append(count_r_label1/count_p_label1)
#     precise.append(count_r_label2/count_p_label2)
#     precise.append(count_r_label3/count_p_label3)
#     file.write("\nPrecise:\n")
#     for p in precise:
#         file.write(str(p)+',')
#     file.write("\n")

#     # recall for the four classes
#     recall = []
#     recall.append(count_r_label0 / count_d_label0)
#     recall.append(count_r_label1 / count_d_label1)
#     recall.append(count_r_label2 / count_d_label2)
#     recall.append(count_r_label3 / count_d_label3)
#     file.write("\nRecall:\n")
#     for r in recall:
#         file.write(str(r)+',')
#     file.write("\n")

#     # recall for the four classes   
#     F1score = []
#     F1score.append(2/((1/precise[0])+(1/recall[0])))
#     F1score.append(2/((1/precise[1])+(1/recall[1])))
#     F1score.append(2/((1/precise[2])+(1/recall[2])))
#     F1score.append(2/((1/precise[3])+(1/recall[3])))
#     file.write("\nF1 Score:\n")
#     for f1 in F1score:
#         file.write(str(f1)+',')
#     file.write("\n")

#     file.write(str5)
#     file.write(str6)
# file.close()


# Part 4: Classification using sigmoid kernel SVM
    acc_c_list = []
    for c in c_list:
        acc_alpha_list = []
        for alpha in alpha_list:
            acc_r_list = []
            for r in r_list:
                print('\n value of c is: ', c)
                print('value of alpha is: ', alpha)
                print('value of r is: ', r)
                param = svm_parameter(
                    '-t 3 -v 5 -h 0 -g '+str(alpha)+' -c '+str(c)+' -r '+str(r))
                m = svm_train(prob, param)
                acc_r_list.append(m)
            acc_alpha_list.append(acc_r_list)
        acc_c_list.append(acc_alpha_list)

    index = np.argmax(acc_c_list)
    index_c = index//169
    index_alpha = (index - 169*index_c) // 13
    index_r = (index - 169*index_c - 13 * index_alpha)
    c = c_list[index_c]
    alpha = alpha_list[index_alpha]
    r = r_list[index_r]
    print('\n value of c is: ',c)
    str2 = '\n value of c is:  ' + str(c)
    file.write(str2)
    print('value of alpha is: ',alpha)
    str3 = '\n value of alpha is:  ' + str(alpha)
    file.write(str3)
    print('value of r is: ',r)
    str4 = '\n value of alpha is:  ' + str(r)
    file.write(str4)
    param = svm_parameter('-t 3 -h 0 -g '+str(alpha) +
                        ' -c '+str(c)+' -r '+str(r))
    m = svm_train(prob, param)
    # column=index%13
    print('Training acc:')
    p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
    str5 = '\nTraining acc:' + str(p_acc[0])
    print('Testing acc:')
    p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
    str6 = '\nTesting acc:' + str(p_acc[0]) + "\n"

    ####     calculate Precise, Recall and F1 score      #####
    # p_label is the predicted class labels
    # y_test is the desired class labels

    # number of predicted label
    count_p_label0 = p_label.count(0.0)
    count_p_label1 = p_label.count(1.0)
    count_p_label2 = p_label.count(2.0)
    count_p_label3 = p_label.count(3.0)
    # number of desired label
    count_d_label0 = y_test.count(0)
    count_d_label1 = y_test.count(1)
    count_d_label2 = y_test.count(2)
    count_d_label3 = y_test.count(3)
    # number of real label
    count_r_label0 = 0
    count_r_label1 = 0
    count_r_label2 = 0
    count_r_label3 = 0

    for i in range(len(p_label)):
        if p_label[i] == 0 and y_test[i] == 0:
            count_r_label0 = count_r_label0 + 1
        elif p_label[i] == 1 and y_test[i] == 1:
            count_r_label1 = count_r_label1 + 1
        elif p_label[i] == 2 and y_test[i] == 2:
            count_r_label2 = count_r_label2 + 1
        elif p_label[i] == 3 and y_test[i] == 3:
            count_r_label3 = count_r_label3 + 1
    
    # precise for the four classes
    precise = []
    precise.append(count_r_label0/count_p_label0)
    precise.append(count_r_label1/count_p_label1)
    precise.append(count_r_label2/count_p_label2)
    precise.append(count_r_label3/count_p_label3)
    file.write("\nPrecise:\n")
    for p in precise:
        file.write(str(p)+',')
    file.write("\n")

    # recall for the four classes
    recall = []
    recall.append(count_r_label0 / count_d_label0)
    recall.append(count_r_label1 / count_d_label1)
    recall.append(count_r_label2 / count_d_label2)
    recall.append(count_r_label3 / count_d_label3)
    file.write("\nRecall:\n")
    for r in recall:
        file.write(str(r)+',')
    file.write("\n")

    # recall for the four classes   
    F1score = []
    F1score.append(2/((1/precise[0])+(1/recall[0])))
    F1score.append(2/((1/precise[1])+(1/recall[1])))
    F1score.append(2/((1/precise[2])+(1/recall[2])))
    F1score.append(2/((1/precise[3])+(1/recall[3])))
    file.write("\nF1 Score:\n")
    for f1 in F1score:
        file.write(str(f1)+',')
    file.write("\n")

    file.write(str5)
    file.write(str6)
file.close()
