# MLP using keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from PIL import Image
import random

# get the training data
# path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\simple images (same size)\\'
path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\world maps (same size)\\'
num_notmap=80
num_map=80

width=60
height=50
num_pixels=width*height
input_size=width*height*3
# num_width=300
# num_height=250
# num_pixels=num_width*num_height
X_batches=[]
y_batches=[]
data_pair=[]

# Get the image data and store data into X_batches and y_batches
index=1
for i in range(num_notmap):
    try:
        name_source='image'+str(index)+'.png'
        img = Image.open(path_source1+name_source)
    except:
        name_source='image'+str(index+1)+'.png'
        img = Image.open(path_source1+name_source)
    
    # img = img.resize((width, height), Image.ANTIALIAS)
    pixel_values=list(img.getdata())
    data_pair.append(pixel_values)
    index=index+1

for i in range(num_map):
    name_source='map'+str(i+1)+'.png'
    img = Image.open(path_source2+name_source)
    # img = img.resize((width, height), Image.ANTIALIAS)
    pixel_values=list(img.getdata())
    print(len(pixel_values))
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
        print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[1])
    else:
        print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[0])

len_x=len(data_pair_3[0])-1
# Shuffle data_pair as input of Neural Network
# random.seed(42)
random.shuffle(data_pair_3)
for i in range(num_total):
    print(len(data_pair_3[i]))
data_pair=np.array(data_pair_3)
# print(data_pair[0].shape)
# print(data_pair[0][75000])

print(len_x)
X_batches_255=[data_pair_3[i][0:len_x] for i in range(num_total)]  
for j in range(num_total):
    print(len(data_pair_3[j])-1)
    print(data_pair_3[j][len(data_pair_3[j])-1])
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
print('y_test:',y_test)

##x_train_1 = np.random.random((1000, 20))
##y_train_1 = np.random.randint(2, size=(1000, 1))
##x_test_1 = np.random.random((100, 20))
##y_test_1 = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(300, input_dim=input_size, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=50,
          batch_size=10)
score = model.evaluate(x_test, y_test, batch_size=10)
y=model.predict(x_test)
print(y)
print(score)


