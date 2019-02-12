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


# get the training data
# path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
path_source1='C:\\Users\\li.7957\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMaps\\'
path_source2='C:\\Users\\li.7957\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\world maps\\'
num_notmap=60
num_map=80

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



# num_width=300
# num_height=250
# num_pixels=num_width*num_height

data_pair=[]

# Get the image data and store data into X_batches and y_batches

for i in range(num_map):
    name_source='map'+str(i+1)+'.jpg'
    img = Image.open(path_source2+name_source)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values=list(img_resized.getdata())
    # print(len(pixel_values))
    data_pair.append(pixel_values)

for i in range(num_notmap):
    name_source='NotMap'+str(i+1)+'.jpeg'
    img = Image.open(path_source1+name_source)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values=list(img_resized.getdata())
    data_pair.append(pixel_values)

num_total=num_map+num_notmap

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

for inx in range(1):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(10, 10), strides=(1, 1),
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
    # for i in range(num_total):
    #     print(len(data_pair_3[i]))
    data_pair=np.array(data_pair_3)

    num_test_image=20
    index_image_list=[]
    for i in range(num_total-20,num_total):
        index_image_list.append(data_pair_3[i][inx_image-1])
    print('The indice of images to be test')
    print(index_image_list)
    # print(data_pair[0].shape)
    # print(data_pair[0][75000])

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

    x_train=X_batches[0:120].reshape(120,input_size)
    x_test=X_batches[120:140].reshape(20,input_size)
    y_train=y_batches[0:120].reshape(120,1)
    y_test=y_batches[120:140].reshape(20,1)
    print('y_test:',y_test.reshape(1,20))

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
    
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test),
          callbacks=[history])

    # score = model.evaluate(x_test, y_test, batch_size=10)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # plt.plot(range(1, 101), history.acc)
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.show()

    # y=model.predict(x_test)
    # print(y.reshape(1,20))
    # print(score)


