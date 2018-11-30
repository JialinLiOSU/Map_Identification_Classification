# MLP using keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from PIL import Image
import random


# get the training data
# path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
path_source1='C:\\Users\\li.7957\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMaps (same size)\\'
path_source2='C:\\Users\\li.7957\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\world maps (same size)\\'
num_notmap=60
num_map=80

width=120
height=100
num_pixels=width*height
input_size=width*height*3

model = Sequential()
model.add(Dense(500, input_dim=input_size, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# num_width=300
# num_height=250
# num_pixels=num_width*num_height

data_pair=[]

# Get the image data and store data into X_batches and y_batches

for i in range(num_map):
    name_source='map'+str(i+1)+'.png'
    img = Image.open(path_source2+name_source)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values=list(img_resized.getdata())
    # print(len(pixel_values))
    data_pair.append(pixel_values)

for i in range(num_notmap):
    name_source='NotMap'+str(i+1)+'.png'
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
        data_pair_3.append(pixel_value_list+[1])
    else:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[0])

len_x=len(data_pair_3[0])-1
# Shuffle data_pair as input of Neural Network
# random.seed(42)

for inx in range(10):
    X_batches=[]
    y_batches=[]
    print("sets of experiments",inx)
    random.shuffle(data_pair_3)
    # for i in range(num_total):
    #     print(len(data_pair_3[i]))
    data_pair=np.array(data_pair_3)
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

    model.fit(x_train, y_train,
            epochs=200,
            batch_size=10,verbose=2)
    score = model.evaluate(x_test, y_test, batch_size=10)
    y=model.predict(x_test)
    print(y.reshape(1,20))
    print(score)


