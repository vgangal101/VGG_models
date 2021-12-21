import tensorflow as tf 
from tensorflow import keras



def VGG16D(img_shape,num_classes):
    """
    Implements VGG16D architecture with batch normalization after Conv layers and before activation. 
    """
    
    model = keras.Sequential()
    