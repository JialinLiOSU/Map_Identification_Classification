import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from skimage.util import random_noise

img= cv2.imread(r"C:\Users\jiali\OneDrive\Images for training\maps for classification of projections\EqualArea_Projection_Maps\cea.png")
img= cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #we are converting image to BGR because matplotlib displays image BGR format
#if you are using cv2 for displaying image, no need to convert image to BGR

plt.imshow(img)
plt.show()

#lets check size and dimensions of image
height, width, dims= img.shape
print(height, width, dims) #print dimensions of original image

noisy_image= random_noise(img)

plt.subplot(1,2,1)
plt.title('original image')
plt.imshow(img)
plt.subplot(1,2,2)
plt.title('Image after adding noise')
plt.imshow(noisy_image)

blur_image= cv2.GaussianBlur(img, (11,11),0)
plt.subplot(1,2,1)
plt.title('original image')
plt.imshow(img)
plt.subplot(1,2,2)
plt.title('Blurry image')
plt.imshow(blur_image)
