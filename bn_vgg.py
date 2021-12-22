import tensorflow as tf 
from tensorflow import keras



def bn_VGG16D(img_shape,num_classes):
    """
    Implements VGG16D architecture with batch normalization. 
    """
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=img_shape))

    model.add(keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))
    
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=1,
                    kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))
    
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))
    
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))
    
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))


    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))
    
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))


    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.BatchNormalization())
    
    
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.BatchNormalization())
    
    
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))
    
    model.add(keras.layers.BatchNormalization())
    
    

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes))

    return model

    