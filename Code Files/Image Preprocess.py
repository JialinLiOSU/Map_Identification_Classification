### Image preprocess before NN training ###
from PIL import Image
import numpy as np
# path_source='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\Maps\\'
path_source='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\simple images\\'
num_map=80
# num_notmap=222
# resize all of the images 
basewidth = 60
hsize=50
# path_target='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\MapsGrey_7500\\'
path_target='C:\\Users\\Administrator\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\simple images (same size)\\'
for i in range(num_map):
    name_source='image'+str(i+1)+'.png'
    img = Image.open(path_source+name_source)
    # img = Image.open(path_source+name_source).LA
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    name_target='image'+str(i+1)+'.png'
    img.save(path_target+name_target)

# img.show()
# print(img)

