import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from skimage.util import random_noise
from skimage import img_as_ubyte


# imagesToAugment = os.listdir(patrh_source1)
path = r'C:\Users\jiali\OneDrive - The Ohio State University\Images for training\maps for classification of projections\collected maps\EqualArea_Projection_Maps'
imagesToRename = os.listdir(path)
os.chdir(path) 
count = 0
for imgName in imagesToRename:
    
    
    img = cv2.imread(imgName)

    cv2.imwrite('equalarea maps ' + str(count) + '.png', img)
    count = count + 1

# img= cv2.imread(r"C:\Users\li.7957\OneDrive\Images for training\maps for classification of projections\EqualArea_Projection_Maps\cea.png")
# img= cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #we are converting image to BGR because matplotlib displays image BGR format
# #if you are using cv2 for displaying image, no need to convert image to BGR
# img= random_noise(img)

# plt.axis('off')
# plt.imshow(img)
# # plt.show()


# #lets check size and dimensions of image
# # height, width, dims= img.shape
# # print(height, width, dims) #print dimensions of original image


# # blur_image= cv2.GaussianBlur(img, (11,11),0)


# # transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

# path = 'C:\\Users\\li.7957\\Desktop\\Map_Identification_Classification\\world map generation'
# os.chdir(path) 
# filename = 'augmentedCeaProjection0.jpg'
# cv2.imwrite(filename, img) 
# plt.savefig(path+filename)
