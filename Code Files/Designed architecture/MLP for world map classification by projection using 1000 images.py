# MLP for world map classification using keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from PIL import Image
import random
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
import time
import os

# get the training data
path_root = 'C:\\Users\\li.7957\\OneDrive\\Images for training\\maps for classification of projections\\'
# path_root = 'C:\\Users\\jiali\\OneDrive\\Images for training\\maps for classification of projections\\'
path_source1 = path_root+'Equirectangular_Projection_Maps\\'
path_source2 = path_root+'Mercator_Projection_Maps\\'
path_source3 = path_root+'EqualArea_Projection_Maps\\'
path_source4 = path_root+'Robinson_Projection_Maps\\'

num_maps_class=250
width=120
height=100
num_pixels=width*height
input_size=width*height*3
input_shape=(width, height, 3)

strList = [] # save the strings to be written in files

model = Sequential()
model.add(Dense(200, input_dim=input_size, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

strTemp = '200-100-4\n'
strList.append(strTemp)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

strTemp = 'SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)\n'
strList.append(strTemp)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# num_width=300
# num_height=250
# num_pixels=num_width*num_height

data_pair=[]

# Get the image data and store data into X_batches and y_batches
Equirectangular_images = os.listdir(path_source1)
Mercator_images = os.listdir(path_source2)
EqualArea_images = os.listdir(path_source3)
Robinson_images = os.listdir(path_source4)

count = 0
for imgName in Equirectangular_images:
    fullName = path_source1 + imgName
    img = Image.open(fullName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break

count = 0
for imgName in Mercator_images:
    img = Image.open(path_source2 + imgName, 'r')
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break

count = 0
for imgName in EqualArea_images:
    img = Image.open(path_source3 + imgName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    # if len(data_pair)==251:
    #     print(imgName)
    if count >= 250:
        break

count = 0
for imgName in Robinson_images:
    img = Image.open(path_source4 + imgName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break

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
train_size=800
num_test=num_total-train_size
str1="train size:"+str(train_size)+' test size:'+str(num_test)+'\n'
test_loss_list=[]
test_acc_list=[]

for inx in range(10):
    X_batches=[]
    y_batches=[]
    print("sets of experiments",inx)
    strTemp = "sets of experiments"+ str(inx) + '\n'
    strList.append(strTemp)

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
    
    # file.write(str(index_image_list)+'\n') 

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
    # file.write(str(y_test.reshape(1,num_total-train_size)) +'\n')

    y_train_cat = to_categorical(y_train, num_classes=4)
    y_test_cat = to_categorical(y_test, num_classes=4)

    start=time.time() # start time for training
    model.fit(x_train, y_train_cat,
            epochs=100,
            batch_size=10,verbose=2)

    end_train=time.time() # end time for training
    score = model.evaluate(x_test, y_test_cat, batch_size=10)
    end_test=time.time() # end time for testing
    train_time=end_train-start
    test_time=end_test-end_train
    print("train_time:"+ str(train_time)+"\n")
    print("test_time:"+ str(test_time) + "\n")
    strTemp = "train_time:"+ str(train_time)+"\n"
    strList.append(strTemp)
    strTemp = "test_time:"+ str(test_time) + "\n"
    strList.append(strTemp)
    
    test_loss=score[0]
    test_acc=score[1]
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)
    # file.write('Test loss:'+str(test_loss) +' Test accuracy:'+str(test_acc)+'\n')
    strTemp = 'Test loss:'+str(test_loss) +' Test accuracy:'+str(test_acc)+'\n'
    strList.append(strTemp)

    y=model.predict(x_test)
    p_label = np.argmax(y, axis=-1)
    print(p_label)
    print(score)

    # convert from a list of np.array to a list of int
    y_test = [y.tolist()[0] for y in (y_test)]

    # number of predicted label
    count_p_label0 = p_label.count(0.0)
    count_p_label1 = p_label.count(1.0)
    count_p_label2 = p_label.count(2.0)
    count_p_label3 = p_label.count(3.0)
    # number of desired label
    count_d_label0 = y_test.count(0)
    count_d_label1 = y_test.count(1)
    count_d_label2 = y_test.count(2)
    count_d_label3 = y_test.count(3)
    # number of real label
    count_r_label0 = 0
    count_r_label1 = 0
    count_r_label2 = 0
    count_r_label3 = 0

    for i in range(len(p_label)):
        if p_label[i] == 0 and y_test[i] == 0:
            count_r_label0 = count_r_label0 + 1
        elif p_label[i] == 1 and y_test[i] == 1:
            count_r_label1 = count_r_label1 + 1
        elif p_label[i] == 2 and y_test[i] == 2:
            count_r_label2 = count_r_label2 + 1
        elif p_label[i] == 3 and y_test[i] == 3:
            count_r_label3 = count_r_label3 + 1
    
    # precise for the four classes
    precise = []
    precise.append(count_r_label0/count_p_label0)
    precise.append(count_r_label1/count_p_label1)
    precise.append(count_r_label2/count_p_label2)
    precise.append(count_r_label3/count_p_label3)
    # file.write("\nPrecise:\n")
    strTemp = "\nPrecise:\n"
    strList.append(strTemp)
    strTemp = ''
    for p in precise:
        strTemp = strTemp + str(p)+','
    strList.append(strTemp)

    # recall for the four classes
    recall = []
    recall.append(count_r_label0 / count_d_label0)
    recall.append(count_r_label1 / count_d_label1)
    recall.append(count_r_label2 / count_d_label2)
    recall.append(count_r_label3 / count_d_label3)
    # file.write("\nRecall:\n")
    strTemp = "\nRecall:\n"
    strList.append(strTemp)
    strTemp = ''
    for r in recall:
        strTemp = strTemp + str(r)+','
    strList.append(strTemp)

    # recall for the four classes   
    F1score = []
    F1score.append(2/((1/precise[0])+(1/recall[0])))
    F1score.append(2/((1/precise[1])+(1/recall[1])))
    F1score.append(2/((1/precise[2])+(1/recall[2])))
    F1score.append(2/((1/precise[3])+(1/recall[3])))
    # file.write("\nF1 Score:\n")
    strTemp = "\nF1 Score:\n"
    strList.append(strTemp)
    strTemp = ''
    for f1 in F1score:
        strTemp = strTemp + str(f1)+','
    strList.append(strTemp)

filename='MLPforProjection'+'1'+'.txt'
file = open(filename,'a')
for s in strList:
    file.write(s)
file.close() 

