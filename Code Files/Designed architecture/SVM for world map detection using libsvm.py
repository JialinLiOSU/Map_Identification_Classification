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
# path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
path='C:\\Users\\\jiali\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\maps for classification of regions\\'
path_source_nonmap='C:\\Users\\\jiali\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMaps\\'

path_source_world=path+'world maps\\'
path_source_China=path+'China maps\\'
path_source_Korea=path+'South Korea maps\\'
path_source_US=path+'US maps\\'

width=120
height=100
num_pixels=width*height
input_size=width*height*3
input_shape=(width, height, 3)

num_list=[60,100,140,180,220]
for num in num_list:
    num_notmap=num
    num_map=num_notmap
    num_total=num_map+num_notmap
    num_test=40
    num_train=num_total-num_test
    num_map_region=int(num_map/4)

    str1="train size:"+str(num_train)+' test size:'+str(num_test)+'\n'
    print(str1)
    data_pair=[]

    # Get the image data and store data into X_batches and y_batches
    for i in range(num_map_region):
        name_source='map'+str(i+1)+'.jpg'
        img = Image.open(path_source_world+name_source)
        img_resized = img.resize((width, height), Image.ANTIALIAS)
        pixel_values=list(img_resized.getdata())
        data_pair.append(pixel_values)

    for i in range(num_map_region):
        name_source='china_map'+str(i+1)+'.jpg'
        img = Image.open(path_source_China+name_source)
        img_resized = img.resize((width, height), Image.ANTIALIAS)
        pixel_values=list(img_resized.getdata())
        data_pair.append(pixel_values)

    for i in range(num_map_region):
        name_source='south_korea_map'+str(i+1)+'.jpg'
        img = Image.open(path_source_Korea+name_source)
        img_resized = img.resize((width, height), Image.ANTIALIAS)
        pixel_values=list(img_resized.getdata())
        data_pair.append(pixel_values)

    for i in range(num_map_region):
        name_source='us_map'+str(i+1)+'.jpg'
        img = Image.open(path_source_US+name_source)
        img_resized = img.resize((width, height), Image.ANTIALIAS)
        pixel_values=list(img_resized.getdata())
        data_pair.append(pixel_values)

    for i in range(num_notmap):
        name_source='NotMap'+str(i+1)+'.jpeg'
        img = Image.open(path_source_nonmap+name_source)
        img_resized = img.resize((width, height), Image.ANTIALIAS)
        pixel_values=list(img_resized.getdata())
        data_pair.append(pixel_values)

    data_pair_3=[]
    for i in range(num_total):
        pixel_value_list=[]
        for j in range(num_pixels):
            pixels=data_pair[i][j]
            pixel_value_list.append(pixels[0])
            pixel_value_list.append(pixels[1])
            pixel_value_list.append(pixels[2])
        if i<=num_map:
            # print(len(pixel_value_list))
            # data_pair_3.append(pixel_value_list+[1])
            data_pair_3.append(pixel_value_list+[1]+[i])
        else:
            # print(len(pixel_value_list))
            # data_pair_3.append(pixel_value_list+[0])
            data_pair_3.append(pixel_value_list+[0]+[i])

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

        # #The indice of images to be tested, to check which images they are
        # index_image_list=[]
        # for i in range(num_total-num_test,num_total):
        #     index_image_list.append(data_pair_3[i][inx_image-1]+1)
        # print('The indice of images to be test')
        # print(index_image_list)

        X_batches_255=[data_pair_3[i][0:len_x] for i in range(num_total)]  
        y_batches=[data_pair_3[i][len_x] for i in range(num_total)]

        for i in range(num_total):
            X_1img=[X_batches_255[i][j]/255.0 for j in range(len_x)]
            X_batches.append(X_1img)

        # train_size=120

        x_train_array=X_batches[0:num_train]
        x_test_array=X_batches[num_train:num_total]
        y_train_array=y_batches[0:num_train]
        y_test_array=y_batches[num_train:num_total]

        y_train=y_train_array
        y_test=y_test_array
        x_train=[{j:x_train_array[i][j] for j in range(input_size)} for i in range(num_train)]
        x_test=[{j:x_test_array[i][j] for j in range(input_size)} for i in range(num_total-num_train)]

        # lists of different parameters
        c_list=[2**(i-4) for i in range(0,13)]
        alpha_list=[2**(i-4) for i in range(0,13)]
        r_list=[2**(i-4) for i in range(0,13)]

        # Part 1: Classification using linear SVMs
        train1_start=time.time()
        prob  = svm_problem(y_train, x_train)
        train1_end=time.time()
        train1=train1_end-train1_start
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
            train2_end=time.time() # end time for training
            train2=train2_end-train2_start
            train_time=train1+train2
            train_acc=p_acc[0]
            # test time
            print('Testing acc:')
            test_start=time.time()
            p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
            test_end=time.time() # end time for testing
            test_time=test_end-test_start
            test_acc=p_acc[0]
            if train_acc>opt_train_acc:
                opt_train_acc=train_acc
            if test_acc>opt_test_acc:
                opt_test_acc=test_acc
            train_time_list.append(train_time)
            test_time_list.append(test_time)
            pass
        train_time_c_ave=sum(train_time_list)/len(train_time_list)
        test_time_c_ave=sum(test_time_list)/len(test_time_list)
        train_time_c_ave_list.append(train_time_c_ave)
        test_time_c_ave_list.append(test_time_c_ave)
        train_acc_list.append(opt_train_acc)
        test_acc_list.append(opt_test_acc)
        pass
    train_time_ave=sum(train_time_c_ave_list)/len(train_time_c_ave_list)
    test_time_ave=sum(test_time_c_ave_list)/len(test_time_c_ave_list)
    train_acc_ave=sum(train_acc_list)/len(train_acc_list)
    test_acc_ave=sum(test_acc_list)/len(test_acc_list)

    str2="train_time_ave: "+str(train_time_ave)+' test_time_ave: '+str(test_time_ave)+'\n'
    str3="train_acc_ave: "+str(train_acc_ave)+' test_acc_ave: '+str(test_acc_ave)+'\n'

    filename='Results_SVM_Identification'+'1'+'.txt'
    file = open(filename,'a')
    file.write(str1) 
    file.write(str2)
    file.write(str3)
    file.close() 
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
