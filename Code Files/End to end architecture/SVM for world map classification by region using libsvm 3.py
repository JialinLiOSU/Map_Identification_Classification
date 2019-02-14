# CSE 5526 Programming Assignment 3 SVM #
# There are two parts in this project
# When you execute part 1, you can comment part 2 and vice versa
from libsvm.python.svm import *
from libsvm.python.svmutil import *
import random
import numpy as np
from PIL import Image


# get the training data
path='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\maps for classification of regions\\'
path_source1=path+'world maps\\'
path_source2=path+'China maps\\'
path_source3=path+'South Korea maps\\'
path_source4=path+'US maps\\'

# num_notmap=60
# num_map=80
num_maps_class=40

width=120
height=100
num_pixels=width*height
input_size=width*height*3

num_classes = 4

data_pair=[]

# Get the image data and store data into X_batches and y_batches

for i in range(num_maps_class):
    name_source='map'+str(i+1)+'.jpg'
    img = Image.open(path_source1+name_source)
    img = img.resize((width, height), Image.ANTIALIAS)
    pixel_values=list(img.getdata())
    # print(len(pixel_values))
    data_pair.append(pixel_values)

for i in range(num_maps_class):
    name_source='china_map'+str(i+1)+'.jpg'
    img = Image.open(path_source2+name_source)
    img = img.resize((width, height), Image.ANTIALIAS)
    pixel_values=list(img.getdata())
    data_pair.append(pixel_values)

for i in range(num_maps_class):
    name_source='south_korea_map'+str(i+1)+'.jpg'
    img = Image.open(path_source3+name_source)
    img = img.resize((width, height), Image.ANTIALIAS)
    pixel_values=list(img.getdata())
    # print(len(pixel_values))
    data_pair.append(pixel_values)

for i in range(num_maps_class):
    name_source='us_map'+str(i+1)+'.jpg'
    img = Image.open(path_source4+name_source)
    img = img.resize((width, height), Image.ANTIALIAS)
    pixel_values=list(img.getdata())
    data_pair.append(pixel_values)

num_total=num_maps_class*4

data_pair_3=[]
for i in range(num_total):
    # print("i:",i)
    pixel_value_list=[]
    for j in range(num_pixels):
        # print("j:",j)
        pixels=data_pair[i][j]
        pixel_value_list.append(pixels[0])
        pixel_value_list.append(pixels[1])
        pixel_value_list.append(pixels[2])
    if i<num_maps_class:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[0])
    elif i>=num_maps_class and i < num_maps_class*2:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[1])
    elif i>=num_maps_class*2 and i < num_maps_class*3:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[2])
    elif i>=num_maps_class*3 and i < num_maps_class*4:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[3])

len_x=len(data_pair_3[0])-1
train_size=140

for inx in range(10):
    print('Index of sets is: ',inx)
    X_batches=[]
    y_batches=[]

    random.shuffle(data_pair_3)
    data_pair=np.array(data_pair_3)

    X_batches_255=[data_pair_3[i][0:len_x] for i in range(num_total)]  
    y_batches=[data_pair_3[i][len_x] for i in range(num_total)]

    for i in range(num_total):
        X_1img=[X_batches_255[i][j]/255.0 for j in range(len_x)]
        X_batches.append(X_1img)

    x_train_array=X_batches[0:train_size]
    x_test_array=X_batches[train_size:num_total]
    y_train_array=y_batches[0:train_size]
    y_test_array=y_batches[train_size:num_total]

    y_train=y_train_array
    y_test=y_test_array
    x_train=[{j:x_train_array[i][j] for j in range(input_size)} for i in range(train_size)]
    x_test=[{j:x_test_array[i][j] for j in range(input_size)} for i in range(num_total-train_size)]
    num_train=len(y_train)
    num_test=len(y_test)
    # print('training set:',num_train)
    # print('testing set:',num_test)
    c_list=[2**(i-4) for i in range(0,13)]
    alpha_list=[2**(i-4) for i in range(0,13)]
    r_list=[2**(i-4) for i in range(0,13)]

    # Part 1: Classification using linear SVMs
    prob  = svm_problem(y_train, x_train)
    # acc_c_list=[]
    # for c in c_list:
    #     print('value of c is: ',c)
    #     param = svm_parameter('-t 0 -v 5 -h 0 -c '+str(c))
    #     m = svm_train(prob, param)
    #     acc_c_list.append(m)
    # index=np.argmax(acc_c_list)
    # index_c=index
    # # index_alpha=(index - 13*index_c)
    # c=c_list[index_c]
    # # alpha=alpha_list[index_alpha]
    # print('\n value of c is: ',c)
    # # print('value of alpha is: ',alpha)
    # param = svm_parameter('-t 0 -h 0 -c '+str(c))
    # m = svm_train(prob, param)
    # # column=index%13
    # print('\nTraining acc:')
    # p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
    # print('Testing acc:')
    # p_label, p_acc, p_val = svm_predict(y_test, x_test, m)

    # alpha=64
    # c=64
    # r=2
# Part 2: Classification using RBF kernel SVM
    # acc_c_list=[]
    # for c in c_list:
    #     acc_alpha_list=[]
    #     for alpha in alpha_list:
    #         print('value of c is: ',c)
    #         print('value of alpha is: ',alpha)
    #         param = svm_parameter('-t 2 -v 5 -h 0 -g '+str(alpha)+' -c '+str(c))
    #         m = svm_train(prob, param)
    #         acc_alpha_list.append(m)
    #     acc_c_list.append(acc_alpha_list)

    # index=np.argmax(acc_c_list)
    # index_c=index//13
    # index_alpha=(index - 13*index_c)
    # c=c_list[index_c]
    # alpha=alpha_list[index_alpha]
    # print('\n value of c is: ',c)
    # print('value of alpha is: ',alpha)
    # param = svm_parameter('-t 2 -h 0 -g '+str(alpha)+' -c '+str(c))
    # m = svm_train(prob, param)
    # # column=index%13
    # print('\nTraining acc:')
    # p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
    # print('Testing acc:')
    # p_label, p_acc, p_val = svm_predict(y_test, x_test, m)


# Part 3: Classification using polynomial SVM
    acc_c_list=[]
    for c in c_list:
        acc_alpha_list=[]
        for alpha in alpha_list:
            acc_r_list=[]
            for r in r_list:
                print('\n value of c is: ',c)
                print('value of alpha is: ',alpha)
                print('value of r is: ',r)
                param = svm_parameter('-t 1 -v 5 -h 0 -g '+str(alpha)+' -c '+str(c)+' -r '+str(r))
                m = svm_train(prob, param)
                acc_r_list.append(m)
            acc_alpha_list.append(acc_r_list)
        acc_c_list.append(acc_alpha_list)

    index=np.argmax(acc_c_list)
    index_c=index//169
    index_alpha=(index - 169*index_c) // 13
    index_r = (index - 169*index_c - 13 * index_alpha)
    c=c_list[index_c]
    alpha=alpha_list[index_alpha]
    r=r_list[index_r]
    print('\n value of c is: ',c)
    print('value of alpha is: ',alpha)
    print('value of r is: ',r)
    param = svm_parameter('-t 1 -h 0 -g '+str(alpha)+' -c '+str(c)+' -r '+str(r))
    m = svm_train(prob, param)
    # column=index%13
    print('Training acc:')
    p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
    print('Testing acc:')
    p_label, p_acc, p_val = svm_predict(y_test, x_test, m)


# Part 4: Classification using sigmoid kernel SVM
    # acc_c_list=[]
    # for c in c_list:
    #     acc_alpha_list=[]
    #     for alpha in alpha_list:
    #         acc_r_list=[]
    #         for r in r_list:
    #             print('\n value of c is: ',c)
    #             print('value of alpha is: ',alpha)
    #             print('value of r is: ',r)
    #             param = svm_parameter('-t 3 -v 5 -h 0 -g '+str(alpha)+' -c '+str(c)+' -r '+str(r))
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
    # param = svm_parameter('-t 3 -h 0 -g '+str(alpha)+' -c '+str(c)+' -r '+str(r))
    # m = svm_train(prob, param)
    # # column=index%13
    # print('Training acc:')
    # p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
    # print('Testing acc:')
    # p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
