# CSE 5526 Programming Assignment 3 SVM #
# There are two parts in this project
# When you execute part 1, you can comment part 2 and vice versa
from libsvm.python.svm import *
from libsvm.python.svmutil import *
import random
import numpy as np
from PIL import Image
import time
import pickle

# get the training data
# path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
path_source1 = 'C:\\Users\\li.7957\\OneDrive\\Images for training\\map identification_world maps\\train\\'
path_source2 = 'C:\\Users\\li.7957\\OneDrive\\Images for training\\map identification_world maps\\test\\'
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

# point_generate_random(num_points,num_pixel)
with open('C:\\Users\\li.7957\\OneDrive\\Images for training\\map identification_world maps\\test_identification_world_SVM.pickle', 'rb') as file:
    [x_test, y_test] = pickle.load(file)
with open('C:\\Users\\li.7957\\OneDrive\\Images for training\\map identification_world maps\\train_identification_world_SVM.pickle', 'rb') as file:
    [x_train, y_train] = pickle.load(file)


# lists of different parameters
c_list=[2**(i-4) for i in range(0,13)]
alpha_list=[2**(i-4) for i in range(0,13)]
r_list=[2**(i-4) for i in range(0,13)]

str1 = "Classification using RBF kernel SVM" + "\n"

for inx in range(1):
    prob  = svm_problem(y_train, x_train)
    opt_train_acc=0
    opt_test_acc=0
    # for c in c_list:
    #     print('value of c is: ',c)
    #     param = svm_parameter('-t 0 -h 0 -c '+str(c))
    #     m = svm_train(prob, param)
    #     print('Training acc:')
    #     p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
    #     train_acc=p_acc[0]
    #     # test time
    #     print('Testing acc:')
    #     test_start=time.time()
    #     p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
    #     test_end=time.time() # end time for testing
    #     test_time=test_end-test_start
    #     test_acc=p_acc[0]
    #     if train_acc>opt_train_acc:
    #         opt_train_acc=train_acc
    #     if test_acc>opt_test_acc:
    #         opt_test_acc=test_acc
    #     pass
    # pass

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
    # str2=" value of c is: "+str(c)+' value of alpha is: '+str(alpha)+'\n'
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
    str2=" value of c is: "+str(c)+' value of alpha is: '+str(alpha)+ ' value of r is: ' + str(r) +'\n'
    param = svm_parameter('-t 1 -h 0 -g '+str(alpha)+' -c '+str(c)+' -r '+str(r))
    m = svm_train(prob, param)
    # column=index%13
    print('Training acc:')
    p_label, p_acc_1, p_val = svm_predict(y_train, x_train, m)
    print('Testing acc:')
    p_label, p_acc_2, p_val = svm_predict(y_test, x_test, m)

str3="train_acc: "+str(p_acc_1)+' test_acc: '+str(p_acc_2)+'\n'
filename='Results_SVM_Identification'+'1'+'.txt'
file = open(filename,'a')
file.write(str1)
file.write(str2)
file.write(str3)
file.close() 


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
