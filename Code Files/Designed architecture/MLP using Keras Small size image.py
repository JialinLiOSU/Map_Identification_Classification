# MLP using keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from PIL import Image
import random

# get the training data
path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey_7500\\'
path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey_7500\\'
# path_source1='C:\\Users\\li.7957\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\li.7957\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
num_notmap=220
num_map=60
num_width=100
num_height=75
num_pixels=num_width*num_height
X_batches=[]
y_batches=[]
data_pair=[]

# Get the image data and store data into X_batches and y_batches
for i in range(num_notmap):
    name_source='NotMap'+str(i+1)+'.png'
    img = Image.open(path_source1+name_source)
    pixel_values=list(img.getdata())
    # X_batches.append(pixel_values)
    # y_batches.append(0)
    data_pair.append(pixel_values+[0])

for i in range(num_map):
    name_source='Map'+str(i+1)+'.png'
    img = Image.open(path_source2+name_source)
    pixel_values=list(img.getdata())
    # X_batches.append(pixel_values)
    # y_batches.append(1)
    data_pair.append(pixel_values+[1])

# Shuffle data_pair as input of Neural Network
random.seed(42)
random.shuffle(data_pair)
data_pair=np.array(data_pair)
len_x=len(data_pair[0])-1
X_batches_255=[data_pair[i][0:len_x] for i in range(data_pair.shape[0])]  
y_batches=[data_pair[i][len_x] for i in range(data_pair.shape[0])]
# data get from last step is with the total value of pixel 255 

for i in range(data_pair.shape[0]):
    X_1img=[X_batches_255[i][j][0]/255.0 for j in range(len_x)]
    X_batches.append(X_1img)
X_batches=np.array(X_batches)
y_batches=np.array(y_batches)

total_size=280
train_size=260
test_size=total_size-train_size
x_train=X_batches[0:train_size].reshape(train_size,num_pixels)
x_test=X_batches[train_size:total_size].reshape(test_size,num_pixels)
y_train=y_batches[0:train_size].reshape(train_size,1)
y_test=y_batches[train_size:total_size].reshape(test_size,1)
print('y_test:',y_test)

##x_train_1 = np.random.random((1000, 20))
##y_train_1 = np.random.randint(2, size=(1000, 1))
##x_test_1 = np.random.random((100, 20))
##y_test_1 = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(3000, input_dim=num_pixels, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=1)
score = model.evaluate(x_test, y_test, batch_size=10)
y=model.predict(x_test)
print(y)
print(score)
