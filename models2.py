import tensorflow as tf
from tensorflow import keras


def VGG11_A(num_classes,img_shape):
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=img_shape))

    model.add(keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))


    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes))

    return model


def VGG13_B(num_classes,img_shape):
    model = keras.Sequential()
    #model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.InputLayer(input_shape=img_shape))

    model.add(keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=1,
                    kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))


    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes))

    return model


def VGG16_C(num_classes,img_shape):
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=img_shape))

    model.add(keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=1,
                    kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(256,(1,1),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))


    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(1,1),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))


    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(1,1),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))


    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes))


    return model

def VGG16_D(num_classes,img_shape):
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=img_shape))

    model.add(keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=1,
                    kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))


    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))


    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))


    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes))

    return model


def VGG19_E(self,num_classes,img_shape):
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=img_shape))

    model.add(keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=1,
                    kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))


    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
                kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=1,
            kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.L2(l2=5e-4)))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes))

    return model
