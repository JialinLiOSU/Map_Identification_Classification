# Examine effect of training size on prediction accuracy #
# using linear SVMs

from libsvm.python.svm import *
from libsvm.python.svmutil import *
import random
import numpy as np
from PIL import Image
import time
import pickle

# get the training data
path_source1 = 'C:\\Users\\li.7957\\OneDrive\\Images for training\\region classification images for experiments\\train\\'
path_source2 = 'C:\\Users\\li.7957\\OneDrive\\Images for training\\region classification images for experiments\\test\\'
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
num_classes = 4

# point_generate_random(num_points,num_pixel)
with open('C:\\Users\\li.7957\\OneDrive\\Images for training\\region classification images for experiments\\test_classification_regions_SVM.pickle', 'rb') as file:
    [x_test, y_test] = pickle.load(file)
with open('C:\\Users\\li.7957\\OneDrive\\Images for training\\region classification images for experiments\\train_classification_regions_SVM.pickle', 'rb') as file:
    [x_train, y_train] = pickle.load(file)

# lists of different parameters
c_list=[2**(i-4) for i in range(0,13)]
alpha_list=[2**(i-4) for i in range(0,13)]
r_list=[2**(i-4) for i in range(0,13)]

str1 = "Classification using linear SVMs" + "\n"
# Part 1: Classification using linear SVMs
prob  = svm_problem(y_train, x_train)
opt_train_acc=0
opt_test_acc=0

for c in c_list:
    print('value of c is: ',c)
    param = svm_parameter('-t 0 -h 0 -c '+str(c))
    # train time
    train2_start=time.time() # start time for training
    m = svm_train(prob, param)
    print('Training acc:')
    p_label, p_acc, p_val = svm_predict(y_train, x_train, m)

    train_acc=p_acc[0]
    # test time
    print('Testing acc:')
    test_start=time.time()
    p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
    test_acc=p_acc[0]
    if train_acc>opt_train_acc:
        opt_train_acc=train_acc
    if test_acc>opt_test_acc:
        opt_test_acc=test_acc
    pass

pass

str2="train_acc: "+str(opt_train_acc)+' test_acc: '+str(opt_test_acc)+'\n'

# str3="train_acc_ave: "+str(train_acc_ave)+' test_acc_ave: '+str(test_acc_ave)+'\n'

filename='Results_SVM_Region'+'1'+'.txt'
file = open(filename,'a')
file.write(str1) 
file.write(str2)
file.close() 
