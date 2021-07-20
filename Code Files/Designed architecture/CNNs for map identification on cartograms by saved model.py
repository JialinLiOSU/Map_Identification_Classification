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
numIter = 1
path_root = 'C:\\Users\\li.7957\\OneDrive - The Ohio State University\\Images for training\\region classification images for experiments\\Cartograms\\equalArea\\iter' \
                + str(numIter) + '\\'
path_model = r'C:\Users\li.7957\OneDrive - The Ohio State University\Map classification'

path_source0 = path_root + 'other\\'
path_source1 = path_root+'china\\'
path_source2 = path_root+'sk\\'
path_source3 = path_root+'us\\'
path_source4 = path_root+'worldAntarctica\\'

num_maps_class=60
width=120
height=100
num_pixels=width*height
input_size=width*height*3
input_shape=(width, height, 3)

strList = []  # save the strings to be written in files

strTemp = ' Distortion Level:'+str(numIter) 
strList.append(strTemp)

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


num_total=num_maps_class*5

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
        data_pair_3.append(pixel_value_list+[1]+[i])
    elif i>=num_maps_class and i < num_maps_class*2:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[1]+[i])
    elif i>=num_maps_class*2 and i < num_maps_class*3:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[1]+[i])
    elif i>=num_maps_class*3 and i < num_maps_class*4:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[1]+[i])
    elif i>=num_maps_class*4 and i < num_maps_class*5:
        # print(len(pixel_value_list))
        data_pair_3.append(pixel_value_list+[1]+[i])

dp3_name = zip(data_pair_3,imgNameList)
dp3_name = list(dp3_name)

len_x=len(data_pair_3[0])-2
inx_y=len_x+1
inx_image=inx_y+1
# Shuffle data_pair as input of Neural Network
random.seed(42)


test_loss_list = []
test_acc_list = []

incorrectImgNameStrList = []

path = r'C:\Users\jiali\Desktop\Map_Identification_Classification\Code Files\Designed architecture'
model = keras.models.load_model(path_model+'\\'+'cnn_model_identify2')
X_batches=[]
y_batches=[]

random.shuffle(dp3_name)
data_pair_3, imgNameList = zip(*dp3_name)
data_pair=np.array(data_pair_3)

index_image_list=[]
for i in range(0,num_total):
    index_image_list.append(data_pair_3[i][inx_image-1]+1)
print('The indice of images to be test')
print(index_image_list)

X_batches_255=[data_pair_3[i][0:len_x] for i in range(num_total)]  
y_batches=[data_pair_3[i][len_x] for i in range(num_total)]

for i in range(num_total):
    X_1img=[X_batches_255[i][j]/255.0 for j in range(len_x)]
    X_batches.append(X_1img)
X_batches=np.array(X_batches)
y_batches=np.array(y_batches)


x_test = X_batches[0:num_total].reshape(num_total,input_size)
y_test = y_batches[0:num_total].reshape(num_total,1)

print('y_test:',y_test.reshape(1,num_total))
    
x_test = x_test.reshape(x_test.shape[0], width, height, 3)
y_test = keras.utils.to_categorical(y_test, num_classes)

# preprocess data for transfer learning

# f2 = open('carto_identification_test_' + str(numIter) + '.pickle', 'wb')
# f3 = open('imgNameList_carto_identification_' + str(numIter) +'.pickle', 'wb')
# pickle.dump([x_test, y_test], f2)
# pickle.dump(imgNameList,f3)
# f2.close()
# f3.close()

score = model.evaluate(x_test, y_test, verbose=2)

test_loss = score[0]
test_acc = score[1]
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
# number of desired label
count_d_label0 = y_test.count(0)
count_d_label1 = y_test.count(1)
# number of real label
count_r_label0 = 0
count_r_label1 = 0

# collect wrongly classified images
incorrectImgNameStrList.append('\n')  
for i in range(len(p_label)):
    if p_label[i] == 0 and y_test[i] == 0:
        count_r_label0 = count_r_label0 + 1
    elif p_label[i] == 1 and y_test[i] == 1:
        count_r_label1 = count_r_label1 + 1
    else:
        imgName = imgNameList[i ]
        incorrectImgString = '\n' + imgName + ',' + str(y_test[i]) + ',' + str(p_label[i])
        incorrectImgNameStrList.append(incorrectImgString)

# precise for the two classes
precise = []
if count_p_label0 == 0:
    precise.append(-1)
else:
    precise.append(count_r_label0/count_p_label0)

if count_p_label1 == 0:
    precise.append(-1)
else:
    precise.append(count_r_label1/count_p_label1)

# file.write("\nPrecise:\n")
strTemp = " Precise:"
strList.append(strTemp)
strTemp = ' '
for p in precise:
    strTemp = strTemp + str(p)+','
strList.append(strTemp)

# recall for the two classes
recall = []
if count_d_label0 == 0:
    recall.append(-1)
else:
    recall.append(count_r_label0 / count_d_label0)

if count_d_label1 == 0:
    recall.append(-1)
else:
    recall.append(count_r_label1 / count_d_label1)

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

strTemp = " F1 Score:"
strList.append(strTemp)
strTemp = ' '
for f1 in F1score:
    strTemp = strTemp + str(f1)+','
strList.append(strTemp)

filename = 'CNNforIdentification_carto_7_19_2021'+'.txt'
file = open(filename, 'a')
file.writelines(strList)
file.writelines(incorrectImgNameStrList)
file.close()
