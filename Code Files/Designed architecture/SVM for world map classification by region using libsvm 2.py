# CSE 5526 Programming Assignment 3 SVM #
# There are two parts in this project
# When you execute part 1, you can comment part 2 and vice versa
from libsvm.python.svm import *
from libsvm.python.svmutil import *
import random
import numpy as np
from PIL import Image
import time

# get the training data
path='C:\\Users\\li.7957\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\maps for classification of regions\\'
path_source1=path+'world maps\\'
path_source2=path+'China maps\\'
path_source3=path+'South Korea maps\\'
path_source4=path+'US maps\\'


width=120
height=100
num_pixels=width*height
input_size=width*height*3
input_shape=(width, height, 3)

num_classes = 4

num_list=[240,280,320,360,400]
for num in num_list:

    num_total=num
    num_test=40
    num_train=num_total-num_test
    num_map_region=int(num_total/4)

    str1="train size:"+str(num_train)+' test size:'+str(num_test)+'\n'
    print(str1)
    data_pair=[]

    # Get the image data and store data into X_batches and y_batches

    for i in range(num_map_region):
        name_source='map'+str(i+1)+'.jpg'
        img = Image.open(path_source1+name_source)
        img = img.resize((width, height), Image.ANTIALIAS)
        pixel_values=list(img.getdata())
        # print(len(pixel_values))
        data_pair.append(pixel_values)

    for i in range(num_map_region):
        name_source='china_map'+str(i+1)+'.jpg'
        img = Image.open(path_source2+name_source)
        img = img.resize((width, height), Image.ANTIALIAS)
        pixel_values=list(img.getdata())
        data_pair.append(pixel_values)

    for i in range(num_map_region):
        name_source='south_korea_map'+str(i+1)+'.jpg'
        img = Image.open(path_source3+name_source)
        img = img.resize((width, height), Image.ANTIALIAS)
        pixel_values=list(img.getdata())
        # print(len(pixel_values))
        data_pair.append(pixel_values)

    for i in range(num_map_region):
        name_source='us_map'+str(i+1)+'.jpg'
        img = Image.open(path_source4+name_source)
        img = img.resize((width, height), Image.ANTIALIAS)
        pixel_values=list(img.getdata())
        data_pair.append(pixel_values)



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
        if i<num_map_region:
            # print(len(pixel_value_list))
            data_pair_3.append(pixel_value_list+[0]+[i])
        elif i>=num_map_region and i < num_map_region*2:
            # print(len(pixel_value_list))
            data_pair_3.append(pixel_value_list+[1]+[i])
        elif i>=num_map_region*2 and i < num_map_region*3:
            # print(len(pixel_value_list))
            data_pair_3.append(pixel_value_list+[2]+[i])
        elif i>=num_map_region*3 and i < num_map_region*4:
            # print(len(pixel_value_list))
            data_pair_3.append(pixel_value_list+[3]+[i])

# len_x=len(data_pair_3[0])-1
    len_x=len(data_pair_3[0])-2
    inx_y=len_x+1
    inx_image=inx_y+1
    # Shuffle data_pair as input of Neural Network
    # random.seed(42)
    test_acc_list=[]
    train_acc_list=[]
    train_time_list=[]
    test_time_list=[]
    train_time_c_ave_list=[]
    test_time_c_ave_list=[]

    for inx in range(10):
        X_batches=[]
        y_batches=[]
        print("sets of experiments",inx)

        random.shuffle(data_pair_3)
        data_pair=np.array(data_pair_3)

        X_batches_255=[data_pair_3[i][0:len_x] for i in range(num_total)]  
        y_batches=[data_pair_3[i][len_x] for i in range(num_total)]

        for i in range(num_total):
            X_1img=[X_batches_255[i][j]/255.0 for j in range(len_x)]
            X_batches.append(X_1img)

        x_train_array=X_batches[0:num_train]
        x_test_array=X_batches[num_train:num_total]
        y_train_array=y_batches[0:num_train]
        y_test_array=y_batches[num_train:num_total]

        y_train=y_train_array
        y_test=y_test_array
        x_train=[{j:x_train_array[i][j] for j in range(input_size)} for i in range(num_train)]
        x_test=[{j:x_test_array[i][j] for j in range(input_size)} for i in range(num_total-num_train)]
        # print('training set:',num_train)
        # print('testing set:',num_test)
        c_list=[2**(i-4) for i in range(0,13)]
        alpha_list=[2**(i-4) for i in range(0,13)]
        r_list=[2**(i-4) for i in range(0,13)]


        train1_start=time.time()
        prob  = svm_problem(y_train, x_train)
        train1_end=time.time()
        train1=train1_end-train1_start
    # Part 2: Classification using RBF kernel SVM
        acc_c_list=[]
        for c in c_list:
            acc_alpha_list=[]
            for alpha in alpha_list:
                print('value of c is: ',c)
                print('value of alpha is: ',alpha)
                param = svm_parameter('-t 2 -v 5 -h 0 -g '+str(alpha)+' -c '+str(c))
                m = svm_train(prob, param)
                acc_alpha_list.append(m)
            acc_c_list.append(acc_alpha_list)

        index=np.argmax(acc_c_list)
        index_c=index//13
        index_alpha=(index - 13*index_c)
        c=c_list[index_c]
        alpha=alpha_list[index_alpha]
        print('\n value of c is: ',c)
        print('value of alpha is: ',alpha)
        param = svm_parameter('-t 2 -h 0 -g '+str(alpha)+' -c '+str(c))
        train2_start=time.time() # start time for training
        m = svm_train(prob, param)
        # column=index%13
        print('\nTraining acc:')
        p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
        train2_end=time.time() # end time for training
        train2=train2_end-train2_start
        train_time=train1+train2
        train_acc=p_acc[0]
        print('Testing acc:')
        test_start=time.time()
        p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
        test_end=time.time() # end time for testing
        test_time=test_end-test_start
        test_acc=p_acc[0]
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_time_list.append(train_time)
        test_time_list.append(test_time)

    train_time_ave=sum(train_time_list)/len(train_time_list)
    test_time_ave=sum(test_time_list)/len(test_time_list)
    train_acc_ave=sum(train_acc_list)/len(train_acc_list)
    test_acc_ave=sum(test_acc_list)/len(test_acc_list)

    str2="train_time_ave: "+str(train_time_ave)+' test_time_ave: '+str(test_time_ave)+'\n'
    str3="train_acc_ave: "+str(train_acc_ave)+' test_acc_ave: '+str(test_acc_ave)+'\n'

    filename='Results_SVM_Region'+'2'+'.txt'
    file = open(filename,'a')
    file.write(str1) 
    file.write(str2)
    file.write(str3)
    file.close() 
