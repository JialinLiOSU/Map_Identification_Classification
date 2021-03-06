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
path_source0 = path_root + 'Other_Projections_Maps\\'
path_source1 = path_root+'Equirectangular_Projection_Maps\\'
path_source2 = path_root+'Mercator_Projection_Maps\\'
path_source3 = path_root+'EqualArea_Projection_Maps\\'
path_source4 = path_root+'Robinson_Projection_Maps\\'
path_source5 = path_root+'Cartograms\\pop2007_0_china\\'

num_maps_class=250
width=120
height=100
num_pixels=width*height
input_size=width*height*3
input_shape=(width, height, 3)

strList = [] # save the strings to be written in files
num_classes = 5

# num_width=300
# num_height=250
# num_pixels=num_width*num_height

data_pair=[]

# Get the image data and store data into X_batches and y_batches
OtherProjection_images = os.listdir(path_source0)
Equirectangular_images = os.listdir(path_source1)
Mercator_images = os.listdir(path_source2)
EqualArea_images = os.listdir(path_source3)
Robinson_images = os.listdir(path_source4)

cartoImgList = []
carto_images = os.listdir(path_source5)

# Read map images from other projections
count = 0
imgNameList = []
for imgName in OtherProjection_images:
    imgNameList.append(imgName)
    fullName = path_source0 + imgName
    img = Image.open(fullName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break

# Read map images from Equirectangular projections
count = 0
for imgName in Equirectangular_images:
    imgNameList.append(imgName)
    fullName = path_source1 + imgName
    img = Image.open(fullName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break

# Read map images from Mercator projections
count = 0
for imgName in Mercator_images:
    imgNameList.append(imgName)
    img = Image.open(path_source2 + imgName, 'r')
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break

# Read map images from Lambort Cylindrical EqualArea projections
count = 0
for imgName in EqualArea_images:
    imgNameList.append(imgName)
    img = Image.open(path_source3 + imgName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    # if len(data_pair)==251:
    #     print(imgName)
    if count >= 250:
        break

# Read map images from Robinson projections
count = 0
for imgName in Robinson_images:
    imgNameList.append(imgName)
    img = Image.open(path_source4 + imgName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    data_pair.append(pixel_values)
    count = count + 1
    if count >= 250:
        break

# Read cartogram images
for cartoImg in carto_images:
    img = Image.open(path_source5 + cartoImg)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    cartoImgList.append(pixel_values)


num_total=num_maps_class * num_classes

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
    elif i>=num_maps_class*4 and i < num_maps_class*5:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[4]+[i])

cartoImgList_3 = []
numCartoImg = len(cartoImgList)
for i in range(numCartoImg):
    pixel_value_list = []
    for j in range(num_pixels):
        # print("j:",j)
        pixels = cartoImgList[i][j]
        pixel_value_list.append(pixels[0])
        pixel_value_list.append(pixels[1])
        pixel_value_list.append(pixels[2])
    cartoImgList_3.append(pixel_value_list+[1]+[i])

dp3_name = zip(data_pair_3,imgNameList)
dp3_name = list(dp3_name)

len_x=len(data_pair_3[0])-2
inx_y=len_x+1
inx_image=inx_y+1
# Shuffle data_pair as input of Neural Network
# random.seed(42)
train_size=1000
num_test=num_total-train_size
strTemp = "number of iterations:"+str(10)
strList.append(strTemp)
# str1="train size:"+str(train_size)+' test size:'+str(num_test)+'\n'
test_loss_list=[]
test_acc_list=[]

# layerSettings = [[1000,500,200,100]]
# layerSettings = [[100],[150],[200],[300],[350],[400],[450],[500]]
layerSettings = [[450]]
for ls in layerSettings:
    strList = []  # save the strings to be written in files
    incorrectImgNameStrList = []   
    
    strTemp = "\n"+str(ls[0]) + "-5"
    # strTemp = "\n"+str(ls[0]) + "-"+str(ls[1]) + "-"+str(ls[2]) + "-"+str(ls[3]) 
    strList.append(strTemp)

    for inx in range(3):
        print("sets of experiments",inx)
        strTemp = "\nsets of experiments"+ str(inx)
        strList.append(strTemp)

        model = Sequential()
        model.add(Dense(ls[0], input_dim=input_size, activation='relu'))
        model.add(Dropout(0.5))
        # model.add(Dense(ls[1], activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(ls[2], activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(ls[3], activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

        strTemp = ' SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)\n'
        strList.append(strTemp)

        model.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy'])

        X_batches=[]
        y_batches=[]
        X_carto_batches = []
        y_carto_batches = []

        random.shuffle(dp3_name)
        data_pair_3, imgNameList = zip(*dp3_name)
        data_pair = np.array(data_pair_3)
        cartoImgList = np.array(cartoImgList_3)

        num_test_image=num_total-train_size
        index_image_list=[]
        for i in range(train_size,num_total):
            index_image_list.append(data_pair_3[i][inx_image-1]+1)
        print('The indice of images to be test')
        print(index_image_list)
        
        # file.write(str(index_image_list)+'\n') 

        # print(len_x)
        X_batches_255=[data_pair_3[i][0:len_x] for i in range(num_total)]  
        y_batches=[data_pair_3[i][len_x] for i in range(num_total)]
        X_carto_255 = [cartoImgList_3[i][0:len_x] for i in range(numCartoImg)]
        y_carto = [cartoImgList_3[i][len_x] for i in range(numCartoImg)]
        # data get from last step is with the total value of pixel 255 

        for i in range(num_total):
            X_1img=[X_batches_255[i][j]/255.0 for j in range(len_x)]
            X_batches.append(X_1img)
        X_batches=np.array(X_batches)
        y_batches=np.array(y_batches)

        for i in range(numCartoImg):
            X_carto_1img = [X_carto_255[i][j]/255.0 for j in range(len_x)]
            X_carto_batches.append(X_carto_1img)
        X_carto_batches = np.array(X_carto_batches)
        y_carto_batches = np.array(y_carto)

        x_train=X_batches[0:train_size].reshape(train_size,input_size)
        x_test=X_batches[train_size:num_total].reshape(num_total-train_size,input_size)
        x_carto_test = X_carto_batches.reshape(numCartoImg, input_size)

        y_train=y_batches[0:train_size].reshape(train_size,1)
        y_test=y_batches[train_size:num_total].reshape(num_total-train_size,1)
        y_carto_test = y_carto_batches.reshape(numCartoImg, 1)

        print('y_test:',y_test.reshape(1,num_total-train_size))
        # file.write(str(y_test.reshape(1,num_total-train_size)) +'\n')

        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_test_cat = to_categorical(y_test, num_classes=num_classes)
        y_carto_test = to_categorical(y_carto_test, num_classes = num_classes )


        start=time.time() # start time for training
        model.fit(x_train, y_train_cat,
                epochs=100,
                batch_size=20,verbose=2)

        strTemp = 'epochs=100, batch_size=20 '
        strList.append(strTemp)

        end_train=time.time() # end time for training
        score = model.evaluate(x_test, y_test_cat, batch_size=20)
        end_test=time.time() # end time for testing
        train_time=end_train-start
        test_time=end_test-end_train
        print("train_time:"+ str(train_time)+"\n")
        print("test_time:"+ str(test_time) + "\n")
        strTemp = " train_time:"+ str(train_time)
        strList.append(strTemp)
        strTemp = " test_time:"+ str(test_time)
        strList.append(strTemp)
        
        test_loss=score[0]
        test_acc=score[1]
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)
        # file.write('Test loss:'+str(test_loss) +' Test accuracy:'+str(test_acc)+'\n')
        strTemp = ' Test loss:'+str(test_loss) +' Test accuracy:'+str(test_acc)
        strList.append(strTemp)

        y=model.predict(x_test)
        p_label = np.argmax(y, axis=-1)

        score = model.evaluate(x_carto_test, y_carto_test, batch_size=20)
        test_loss=score[0]
        test_acc=score[1]
        print('Carto Test loss:', test_loss)
        print('Carto Test accuracy:', test_acc)
        # file.write('Test loss:'+str(test_loss) +' Test accuracy:'+str(test_acc)+'\n')
        strTemp = ' Carto Test loss:'+str(test_loss) +' Carto Test accuracy:'+str(test_acc)
        strList.append(strTemp)

        y_carto_predicted = model.predict(x_carto_test)
        p_carto_label = np.argmax(y_carto_predicted, axis=-1)
        p_carto_label = p_carto_label.tolist() # predicted for comparison
        print(p_label)
        print(score)

    filename='MLP_carto'+'.txt'
    file = open(filename,'a')
    file.writelines(strList)
    file.writelines(incorrectImgNameStrList)
    file.close() 

