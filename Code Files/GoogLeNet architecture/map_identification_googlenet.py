# use GoogLeNet CNN architecture by Keras for map identification
from scipy.misc import imread, imresize

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from googlenet_custom_layers import PoolHelper,LRN
from googlenet import create_googlenet

# the key problem here is make the image format for training