# MLP using keras
# In this file, I increased the number of images used to 400 images totally.
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pylab as plt
from PIL import Image
import random
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
import time


# get the training data
# path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'

path='C:\\Users\\li.7957\\OneDrive\\Images for training\\maps for classification of projections\\'

path_source1=path+'Equirectangular_Projection_Maps\\\\'
path_source2=path+'Mercator_Projection_Maps\\'
path_source3=path+'Miller_Projection_Maps\\'
path_source4=path+'Robinson_Projection_Maps\\'

num_maps_class=100
width=120
height=100
num_pixels=width*height
input_size=width*height*3
input_shape=(width, height, 3)
# model = Sequential()
# model.add(Dense(500, input_dim=input_size, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

num_classes = 4


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()



# num_width=300
# num_height=250
# num_pixels=num_width*num_height

data_pair=[]

# Get the image data and store data into X_batches and y_batches

for i in range(num_maps_class):
    name_source='equirectangular_projection_map'+str(i+1)+'.jpg'
    img = Image.open(path_source1+name_source)
    img = img.resize((width, height), Image.ANTIALIAS)
    pixel_values=list(img.getdata())
    # print(len(pixel_values))
    data_pair.append(pixel_values)

for i in range(num_maps_class):
    name_source='mercator_projection_map'+str(i+1)+'.jpg'
    img = Image.open(path_source2+name_source)
    img = img.resize((width, height), Image.ANTIALIAS)
    pixel_values=list(img.getdata())
    data_pair.append(pixel_values)

for i in range(num_maps_class):
    name_source='miller_projection_map'+str(i+1)+'.jpg'
    img = Image.open(path_source3+name_source)
    img = img.resize((width, height), Image.ANTIALIAS)
    pixel_values=list(img.getdata())
    # print(len(pixel_values))
    data_pair.append(pixel_values)

for i in range(num_maps_class):
    name_source='robinson_projection_map'+str(i+1)+'.jpg'
    img = Image.open(path_source4+name_source)
    img = img.resize((width, height), Image.ANTIALIAS)
    pixel_values=list(img.getdata())
    data_pair.append(pixel_values)

num_total=num_maps_class*4
# data_pair_temp=[data_pair[i] for i in range(300,400)]
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
        data_pair_3.append(pixel_value_list+[0]+[i])# after pixel values, then class number and index
    elif i>=num_maps_class and i < num_maps_class*2:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[1]+[i])
    elif i>=num_maps_class*2 and i < num_maps_class*3:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[2]+[i])
    elif i>=num_maps_class*3 and i < num_maps_class*4:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[3]+[i])

len_x=len(data_pair_3[0])-2
inx_y=len_x+1
inx_image=inx_y+1
# Shuffle data_pair as input of Neural Network
# random.seed(42)

train_size=360
num_test=num_total-train_size
str1="train size:"+str(train_size)+' test size:'+str(num_test)+'\n'
test_loss_list=[]
test_acc_list=[]

filename='Results_CNN_Project'+'_10_percent'+'.txt'
file = open(filename,'a')
for inx in range(1):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',
                    input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.SGD(lr=0.01),
                metrics=['accuracy'])
    X_batches=[]
    y_batches=[]
    print("sets of experiments",inx)
    random.shuffle(data_pair_3)
    # for i in range(num_total):
    #     print(len(data_pair_3[i]))
    data_pair=np.array(data_pair_3)
    # print(data_pair[0].shape)
    # print(data_pair[0][75000])
    num_test_image=num_total-train_size
    index_image_list=[]
    for i in range(train_size,num_total):
        index_image_list.append(data_pair_3[i][inx_image-1]+1)
    print('The indice of images to be test')
    print(index_image_list)
    file.write(str(index_image_list)+'\n') 

    # print(len_x)
    X_batches_255=[data_pair_3[i][0:len_x] for i in range(num_total)]  
    # for j in range(num_total):
        # print(len(data_pair_3[j])-1)
        # print(data_pair_3[j][len(data_pair_3[j])-1])
    y_batches=[data_pair_3[i][len_x] for i in range(num_total)]
    # data get from last step is with the total value of pixel 255 

    for i in range(num_total):
        X_1img=[X_batches_255[i][j]/255.0 for j in range(len_x)]
        X_batches.append(X_1img)
    X_batches=np.array(X_batches)
    y_batches=np.array(y_batches)

    x_train=X_batches[0:train_size].reshape(train_size,input_size)
    x_test=X_batches[train_size:num_total].reshape(num_total-train_size,input_size)
    y_train=y_batches[0:train_size].reshape(train_size,1)
    y_test=y_batches[train_size:num_total].reshape(num_total-train_size,1)

    print('y_test:',y_test.reshape(1,num_total-train_size))
    file.write(str(y_test.reshape(1,num_total-train_size)) +'\n')

    x_train = x_train.reshape(x_train.shape[0], width, height, 3)
    x_test = x_test.reshape(x_test.shape[0], width, height, 3)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    batch_size = 10
    # num_classes = 10
    epochs = 100

    # model.fit(x_train, y_train,
    #         epochs=200,
    #         batch_size=10,verbose=1)
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
    print("train_time:"+ str(train_time)+"\n")
    print("test_time:"+ str(test_time) + "\n")

    
    test_loss=score[0]
    test_acc=score[1]
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)
    file.write('Test loss:'+str(test_loss) +'Test accuracy:'+str(test_acc)+'\n')
    y=model.predict(x_test)
    print(y)
    print(score)
    file.write(str(y)+'\n')
        
    y=model.predict(x_test)
    print(y)
    print(score)


# file.write(str1) 
# file.write(str2)
file.close() 

    # # score = model.evaluate(x_test, y_test, batch_size=10)
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # # plt.plot(range(1, 101), history.acc)
    # # plt.xlabel('Epochs')
    # # plt.ylabel('Accuracy')
    # # plt.show()




