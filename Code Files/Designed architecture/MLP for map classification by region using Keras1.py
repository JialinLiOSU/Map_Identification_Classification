# MLP for world map classification using keras
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from PIL import Image
import random
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
import time

# get the training data
# path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
path='C:\\Users\\li.7957\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\maps for classification of regions\\'
path_source1=path+'world maps\\'
path_source2=path+'China maps\\'
path_source3=path+'South Korea maps\\'
path_source4=path+'US maps\\'
# path_source5='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'

width=120
height=100
num_pixels=width*height
input_size=width*height*3
input_shape=(width, height, 3)

num_classes = 4

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

num_list=[240,280,320,360,400] # total number of images used

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
        pixel_value_list=[]
        for j in range(num_pixels):
            # print("j:",j)
            pixels=data_pair[i][j]
            try:
                pixel_value_list.append(pixels[0])
                pixel_value_list.append(pixels[1])
                pixel_value_list.append(pixels[2])
            except:
                print("i:",i)
                break
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

    len_x=len(data_pair_3[0])-2
    inx_y=len_x+1
    inx_image=inx_y+1
    # Shuffle data_pair as input of Neural Network
    # random.seed(42)
    test_loss_list=[]
    test_acc_list=[]
    train_time_list=[]
    test_time_list=[]

    for inx in range(10):
        model = Sequential()
        model.add(Dense(500, input_dim=input_size, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy'])

        X_batches=[]
        y_batches=[]
        print("sets of experiments",inx)
        random.shuffle(data_pair_3)
        data_pair=np.array(data_pair_3)

        index_image_list=[]
        for i in range(num_total-num_test,num_total):
            index_image_list.append(data_pair_3[i][inx_image-1]+1)

        # print(len_x)
        X_batches_255=[data_pair_3[i][0:len_x] for i in range(num_total)]  
        y_batches=[data_pair_3[i][len_x] for i in range(num_total)]
        # data get from last step is with the total value of pixel 255 

        for i in range(num_total):
            X_1img=[X_batches_255[i][j]/255.0 for j in range(len_x)]
            X_batches.append(X_1img)
        X_batches=np.array(X_batches)
        y_batches=np.array(y_batches)

        x_train=X_batches[0:num_train].reshape(num_train,input_size)
        x_test=X_batches[num_train:num_total].reshape(num_test,input_size)
        y_train=y_batches[0:num_train].reshape(num_train,1)
        y_test=y_batches[num_train:num_total].reshape(num_test,1)

        batch_size = 10
        epochs = 100

        y_train = to_categorical(y_train, num_classes=4)
        y_test = to_categorical(y_test, num_classes=4)

        start=time.time() # start time for training
        model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=2)

        end_train=time.time() # end time for training
         # score = model.evaluate(x_test, y_test, batch_size=10)
        score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
        end_test=time.time() # end time for testing
        train_time=end_train-start
        test_time=end_test-end_train

        test_loss=score[0]
        test_acc=score[1]
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        train_time_list.append(train_time)
        test_time_list.append(test_time)

    train_time_ave=sum(train_time_list)/len(train_time_list)
    test_time_ave=sum(test_time_list)/len(test_time_list)
    test_loss_ave=sum(test_loss_list)/len(test_loss_list)
    test_acc_ave=sum(test_acc_list)/len(test_acc_list)

    str2="train_time_ave: "+str(train_time_ave)+' test_time_ave: '+str(test_time_ave)+'\n'
    str3="test_loss_ave: "+str(test_loss_ave)+' test_acc_ave: '+str(test_acc_ave)+'\n'

    filename='Results_MLP_Region'+'1'+'.txt'
    file = open(filename,'a')
    file.write(str1) 
    file.write(str2)
    file.write(str3)
    file.close() 


