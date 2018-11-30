# MLP using keras
import numpy as np
import keras
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# get the training data
# path_source1='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\NotMapsGrey\\'
# path_source2='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey\\'
path='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\maps for classification of regions\\'
path_source1=path+'world maps\\'
path_source2=path+'China maps\\'
path_source3=path+'South Korea maps\\'
path_source4=path+'US maps\\'

num_maps_class=100
num_maps_per_class=80
width=60
height=50
num_pixels=width*height
input_size=width*height*3
input_shape=(width, height, 3)

num_classes = 4

data_pair=[]
image_matrix = np.empty((num_maps_per_class*4, height, width, 3), dtype=np.float32)

count=0
for i in range(num_maps_class):
    name_source='map'+str(i+1)+'.jpg'
    img = Image.open(path_source1+name_source)
    img = img.resize((width, height), Image.ANTIALIAS)
    img.load()
    data = np.asarray(img, dtype="float32")
    try:
        image_matrix[count] = data
        count=count+1
    except:
        print(i)
    if count>num_maps_per_class:
        break

for i in range(num_maps_class):
    name_source='china_map'+str(i+1)+'.jpg'
    img = Image.open(path_source2+name_source)
    img = img.resize((width, height), Image.ANTIALIAS)
    img.load()
    data = np.asarray(img, dtype="float32")
    try:
        image_matrix[count] = data
        count=count+1
    except:
        print(i)
    if count>num_maps_per_class*2:
        break

for i in range(num_maps_class):
    name_source='south_korea_map'+str(i+1)+'.jpg'
    img = Image.open(path_source3+name_source)
    img = img.resize((width, height), Image.ANTIALIAS)
    img.load()
    data = np.asarray(img, dtype="float32")
    try:
        image_matrix[count] = data
        count=count+1
    except:
        print(i)
    if count>num_maps_per_class*3:
        break

for i in range(num_maps_class):
    name_source='us_map'+str(i+1)+'.jpg'
    img = Image.open(path_source4+name_source)
    img = img.resize((width, height), Image.ANTIALIAS)
    img.load()
    data = np.asarray(img, dtype="float32")
    try:
        image_matrix[count] = data
        count=count+1
    except:
        print(i)
    if count>num_maps_per_class*4:
        break

image_tensor=torch.from_numpy(image_matrix)
list_class1=[0 for i in range(num_maps_per_class)]
list_class2=[1 for i in range(num_maps_per_class)]
list_class3=[2 for i in range(num_maps_per_class)]
list_class4=[3 for i in range(num_maps_per_class)]
list_label=list_class1+list_class2+list_class3+list_class4
label_array=np.array(list_label).reshape(num_maps_per_class*4,1)
label_tensor=torch.from_numpy(label_array)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(num_maps_per_class*4):
        # get the inputs
        # inputs, labels = data
        inputs=image_tensor[i]
        labels=label_tensor[i]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net(images)

########################################################################
# The outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

