# MLP for world map classification using keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from PIL import Image
import random
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD



# get the training data
# path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
path='C:\\Users\\li.7957\\Desktop\\ML-Final-Project\\JialinLi\\VGG16 Architecture\\maps for classification of projections\\'

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

model = Sequential()
model.add(Dense(250, input_dim=input_size, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

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
train_size=320
num_test=num_total-train_size
str1="train size:"+str(train_size)+' test size:'+str(num_test)+'\n'
test_loss_list=[]
test_acc_list=[]

filename='Results_MLP_Project'+'4'+'.txt'
file = open(filename,'a')
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

    y_train = to_categorical(y_train, num_classes=4)
    y_test = to_categorical(y_test, num_classes=4)

    model.fit(x_train, y_train,
            epochs=100,
            batch_size=10,verbose=2)
    score = model.evaluate(x_test, y_test, batch_size=10)
    test_loss=score[0]
    test_acc=score[1]
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)
    file.write('Test loss:'+str(test_loss) +'Test accuracy:'+str(test_acc)+'\n')

    y=model.predict(x_test)
    print(y)
    print(score)
    file.write(str(y)+'\n')
file.close() 

