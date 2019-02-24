# MLP using keras
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pylab as plt
from PIL import Image
import random
import time


# get the training data
# path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
path='C:\\Users\\li.7957\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\maps for classification of regions\\'
path_source_nonmap='C:\\Users\\li.7957\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMaps\\'

path_source_world=path+'world maps\\'
path_source_China=path+'China maps\\'
path_source_Korea=path+'South Korea maps\\'
path_source_US=path+'US maps\\'

width=120
height=100
num_pixels=width*height
input_size=width*height*3
input_shape=(width, height, 3)

num_classes = 2

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

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
            data_pair_3.append(pixel_value_list+[1]+[i])
        else:
            # print(len(pixel_value_list))
            data_pair_3.append(pixel_value_list+[0]+[i])

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
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                        activation='relu',
                        input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.binary_crossentropy,
                        optimizer=keras.optimizers.SGD(lr=0.01),
                        metrics=['accuracy'])
        X_batches=[]
        y_batches=[]
        print("sets of experiments",inx)

        random.shuffle(data_pair_3)
        data_pair=np.array(data_pair_3)

        index_image_list=[]
        for i in range(num_total-num_test,num_total):
            index_image_list.append(data_pair_3[i][inx_image-1]+1)



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
        # print('y_test:',y_test.reshape(1,num_test))

        x_train = x_train.reshape(x_train.shape[0], width, height, 3)
        x_test = x_test.reshape(x_test.shape[0], width, height, 3)

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        batch_size = 5

        epochs = 100

        start=time.time() # start time for training
        model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test, y_test),
            callbacks=[history])

        end_train=time.time() # end time for training
        # score = model.evaluate(x_test, y_test, batch_size=10)
        score = model.evaluate(x_test, y_test, verbose=0)
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


        # plt.plot(range(1, 101), history.acc)
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.show()

        # y=model.predict(x_test)
        # print(y)
        # print(score)
    train_time_ave=sum(train_time_list)/len(train_time_list)
    test_time_ave=sum(test_time_list)/len(test_time_list)
    test_loss_ave=sum(test_loss_list)/len(test_loss_list)
    test_acc_ave=sum(test_acc_list)/len(test_acc_list)

    str2="train_time_ave: "+str(train_time_ave)+' test_time_ave: '+str(test_time_ave)+'\n'
    str3="test_loss_ave: "+str(test_loss_ave)+' test_acc_ave: '+str(test_acc_ave)+'\n'

    filename='Results_CNN_Identification'+'2'+'.txt'
    file = open(filename,'a')
    file.write(str1) 
    file.write(str2)
    file.write(str3)
    file.close() 
