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
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
import time
import os
import pickle


# get the training data
path_root = 'C:\\Users\\li.7957\\OneDrive - The Ohio State University\\Images for training\\region classification images for experiments\\generated images\\'
# path_root = 'C:\\Users\\jiali\\OneDrive\\Images for training\\maps for classification of projections\\'
path_source0 = path_root + 'other\\'
path_source1 = path_root + 'china\\'
path_source2 = path_root + 'south korea\\'
path_source3 = path_root + 'us\\'
path_source4 = path_root + 'world\\'

num_maps_class=250
width=120
height=100
num_pixels=width*height
input_size=width*height*3
input_shape=(width, height, 3)

strList = []  # save the strings to be written in files

num_classes = 5


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
OtherMap_images = os.listdir(path_source0)
ChinaMap_images = os.listdir(path_source1)
SKoreaMap_images = os.listdir(path_source2)
USMap_images = os.listdir(path_source3)
WorldMap_images = os.listdir(path_source4)

# Read map images from other projections
count = 0
imgNameList = []
for imgName in OtherMap_images:
    imgNameList.append(imgName)
    fullName = path_source0 + imgName
    img = Image.open(fullName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= num_maps_class:
        break

count = 0
for imgName in ChinaMap_images:
    imgNameList.append(imgName)
    fullName = path_source1 + imgName
    img = Image.open(fullName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= num_maps_class:
        break

count = 0
for imgName in SKoreaMap_images:
    imgNameList.append(imgName)
    img = Image.open(path_source2 + imgName, 'r')
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= num_maps_class:
        break

count = 0
for imgName in USMap_images:
    imgNameList.append(imgName)
    img = Image.open(path_source3 + imgName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    # if len(data_pair)==251:
    #     print(imgName)
    if count >= num_maps_class:
        break

count = 0
for imgName in WorldMap_images:
    imgNameList.append(imgName)
    img = Image.open(path_source4 + imgName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= num_maps_class:
        break

num_total=num_maps_class*num_classes

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
    if i<num_maps_class:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[0]+[i])
    elif i>=num_maps_class and i < num_maps_class*2:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[1]+[i])
    elif i>=num_maps_class*2 and i < num_maps_class*3:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[2]+[i])
    elif i>=num_maps_class*3 and i < num_maps_class*4:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[3]+[i])
    elif i>=num_maps_class*4 and i < num_maps_class*5:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[4]+[i])

dp3_name = zip(data_pair_3,imgNameList)
dp3_name = list(dp3_name)

len_x=len(data_pair_3[0])-2
inx_y=len_x+1
inx_image=inx_y+1
# Shuffle data_pair as input of Neural Network
# random.seed(42)

train_size= int(num_total*0.8)
num_test=num_total-train_size
strTemp = "Region classification using generated map images"
strList.append(strTemp)
str1="train size:"+str(train_size)+' test size:'+str(num_test)+'\n'
strTemp = "train size:"+str(train_size)+' test size:'+str(num_test)
strList.append(strTemp)

test_loss_list=[]
test_acc_list=[]

# layerSettings = [[16,32], [16, 64], [32, 64],[16,128],[32,128],[64,128],[64,256]]
# layerSettings = [[16,32,64], [16, 64,256], [32, 64,128],[32,128,512],[64,128,256]]
layerSettings = [[16,64,128,256],[64,128,256,512],[32,64,128,256],[128,256,512,1024],[16,32,64,128]]
for ls in layerSettings:
    strList = []  # save the strings to be written in files
    incorrectImgNameStrList = []

    # strTemp = "\n"+str(ls[0]) + "-"+str(ls[1])
    strTemp = "\n"+str(ls[0]) + "-"+str(ls[1]) + "-"+str(ls[2]) + "-"+str(ls[3]) 
    strList.append(strTemp)
    
    for inx in range(1):
        print("sets of experiments", inx)
        strTemp = "\nSets of experiments" + str(inx)
        strList.append(strTemp)

        model = Sequential()
        model.add(Conv2D(ls[0], kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(ls[1], (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(ls[2], (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(ls[3], (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.SGD(lr=0.01),
                    metrics=['accuracy'])
        
        # write the network config into file
        strTemp = " optimizer=keras.optimizers.SGD(lr=0.01)"
        strList.append(strTemp)

        X_batches=[]
        y_batches=[]

        random.shuffle(dp3_name)
        data_pair_3, imgNameList = zip(*dp3_name)
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

        x_train = X_batches[0:train_size].reshape(train_size,input_size)
        x_test = X_batches[train_size:num_total].reshape(num_total-train_size,input_size)
        y_train = y_batches[0:train_size].reshape(train_size,1)
        y_test = y_batches[train_size:num_total].reshape(num_total-train_size,1)

        # import pickle
        # f1 = open('train_classification_region1250_cnn_generated.pickle', 'wb')
        # # f2 = open('test_classification_region1250_cnn.pickle', 'wb')
        # pickle.dump([x_train, y_train], f1)
        # # pickle.dump([x_test, y_test], f2)
        # f1.close()
        # # f2.close()

        # save collected training and testing data for transfer learning and other testing
        # import pickle

        with open(path_root +'test_classification_region1250_cnn.pickle', 'rb') as file:
            [x_test, y_test] = pickle.load(file)


        print('y_test:',y_test.reshape(1,num_total-train_size))
        # file.write(str(y_test.reshape(1,num_total-train_size)) +'\n')

        x_train = x_train.reshape(x_train.shape[0], width, height, 3)
        x_test = x_test.reshape(x_test.shape[0], width, height, 3)

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        # preprocess data for transfer learning
        # f1 = open('train_classification_region1250.pickle', 'wb')
        # f2 = open('test_classification_region1250.pickle', 'wb')
        # f3 = open('imgNameList_after_shuffle_region1250.pickle', 'wb')
        # pickle.dump([x_train, y_train], f1)
        # pickle.dump([x_test, y_test], f2)
        # pickle.dump(imgNameList,f3)
        # f1.close()
        # f2.close()
        # f3.close()

        batch_size = 20
        epochs = 100

        strTemp = ' epochs=100, batch_size=20 '
        strList.append(strTemp)

        start = time.time()  # start time for training
        model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test, y_test),
            callbacks=[history])
        end_train = time.time()  # end time for training
        # score = model.evaluate(x_test, y_test, batch_size=10)
        score = model.evaluate(x_test, y_test, verbose=2)
        end_test = time.time()  # end time for testing
        train_time = end_train-start
        test_time = end_test-end_train
        print("train_time:" + str(train_time)+"\n")
        print("test_time:" + str(test_time) + "\n")
        strTemp = " train_time:" + str(train_time)
        strList.append(strTemp)
        strTemp = " test_time:" + str(test_time)
        strList.append(strTemp)

        test_loss=score[0]
        test_acc=score[1]
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)
        strTemp = ' Test loss:'+str(test_loss) + \
            ' Test accuracy:'+str(test_acc)
        strList.append(strTemp)

        y = model.predict(x_test)
        p_label = np.argmax(y, axis=-1)
        print(p_label)
        print(score)

        # convert from a list of np.array to a list of int
        # y_test = [y.tolist()[0] for y in (y_test)]
        y_test = np.argmax(y_test, axis=-1)
        y_test = y_test.tolist()
        p_label = p_label.tolist()

        # number of predicted label
        count_p_label0 = p_label.count(0)
        count_p_label1 = p_label.count(1)
        count_p_label2 = p_label.count(2)
        count_p_label3 = p_label.count(3)
        count_p_label4 = p_label.count(4)
        # number of desired label
        count_d_label0 = y_test.count(0)
        count_d_label1 = y_test.count(1)
        count_d_label2 = y_test.count(2)
        count_d_label3 = y_test.count(3)
        count_d_label4 = y_test.count(4)
        # number of real label
        count_r_label0 = 0
        count_r_label1 = 0
        count_r_label2 = 0
        count_r_label3 = 0
        count_r_label4 = 0

        # collect wrongly classified images
        incorrectImgNameStrList.append('\n')  
        for i in range(len(p_label)):
            if p_label[i] == 0 and y_test[i] == 0:
                count_r_label0 = count_r_label0 + 1
            elif p_label[i] == 1 and y_test[i] == 1:
                count_r_label1 = count_r_label1 + 1
            elif p_label[i] == 2 and y_test[i] == 2:
                count_r_label2 = count_r_label2 + 1
            elif p_label[i] == 3 and y_test[i] == 3:
                count_r_label3 = count_r_label3 + 1
            elif p_label[i] == 4 and y_test[i] == 4:
                count_r_label4 = count_r_label4 + 1
            else:
                imgName = imgNameList[i + train_size]
                incorrectImgString = '\n' + imgName + ',' + str(y_test[i]) + ',' + str(p_label[i])
                incorrectImgNameStrList.append(incorrectImgString)

        # precise for the four classes
        precise = []
        if count_p_label0 == 0:
            precise.append(-1)
        else:
            precise.append(count_r_label0/count_p_label0)

        if count_p_label1 == 0:
            precise.append(-1)
        else:
            precise.append(count_r_label1/count_p_label1)

        if count_p_label2 == 0:
            precise.append(-1)
        else:
            precise.append(count_r_label2/count_p_label2)

        if count_p_label3 == 0:
            precise.append(-1)
        else:
            precise.append(count_r_label3/count_p_label3)

        if count_p_label4 == 0:
            precise.append(-1)
        else:
            precise.append(count_r_label4/count_p_label4)

        # file.write("\nPrecise:\n")
        strTemp = " Precise:"
        strList.append(strTemp)
        strTemp = ' '
        for p in precise:
            strTemp = strTemp + str(p)+','
        strList.append(strTemp)

        # recall for the four classes
        recall = []
        if count_d_label0 == 0:
            recall.append(-1)
        else:
            recall.append(count_r_label0 / count_d_label0)

        if count_d_label1 == 0:
            recall.append(-1)
        else:
            recall.append(count_r_label1 / count_d_label1)

        if count_d_label2 == 0:
            recall.append(-1)
        else:
            recall.append(count_r_label2 / count_d_label2)

        if count_d_label3 == 0:
            recall.append(-1)
        else:
            recall.append(count_r_label3 / count_d_label3)
        
        if count_d_label4 == 0:
            recall.append(-1)
        else:
            recall.append(count_r_label4 / count_d_label4)

        # file.writ e("\nRecall:\n")
        strTemp = " Recall:"
        strList.append(strTemp)
        strTemp = ' '
        for r in recall:
            strTemp = strTemp + str(r)+','
        strList.append(strTemp)

        # recall for the four classes
        F1score = []
        if precise[0] == -1 or precise[0] == 0 or recall[0] == 0:
            F1score.append(-1)
        else:
            F1score.append(2/((1/precise[0])+(1/recall[0])))
        if precise[1] == -1 or precise[1] == 0 or recall[1] == 0:
            F1score.append(-1)
        else:
            F1score.append(2/((1/precise[1])+(1/recall[1])))
        if precise[2] == -1 or precise[2] == 0 or recall[2] == 0:
            F1score.append(-1)
        else:
            F1score.append(2/((1/precise[2])+(1/recall[2])))
        if precise[3] == -1 or precise[3] == 0 or recall[3] == 0:
            F1score.append(-1)
        else:
            F1score.append(2/((1/precise[3])+(1/recall[3])))

        if precise[4] == -1 or precise[4] == 0 or recall[4] == 0:
            F1score.append(-1)
        else:
            F1score.append(2/((1/precise[4])+(1/recall[4])))

        strTemp = " F1 Score:"
        strList.append(strTemp)
        strTemp = ' '
        for f1 in F1score:
            strTemp = strTemp + str(f1)+','
        strList.append(strTemp)

    filename = 'CNNforRegion_9_29_generated'+'.txt'
    file = open(filename, 'a')
    file.writelines(strList)
    file.writelines(incorrectImgNameStrList)
    file.close()

        




