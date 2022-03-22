import tensorflow as tf
from tensorflow import keras


def VGG16(num_classes,img_shape):

    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=img_shape))

    # conv block 1
    model.add(keras.layers.Conv2D(64,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
    model.add(keras.layers.Dropout(0.25))

    #conv block 2
    model.add(keras.layers.Conv2D(128,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(128,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
    model.add(keras.layers.Dropout(0.25))

    # conv block 3
    model.add(keras.layers.Conv2D(256,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(256,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(256,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
    model.add(keras.layers.Dropout(0.25))


    # conv block 4
    model.add(keras.layers.Conv2D(512,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(512,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(512,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
    model.add(keras.layers.Dropout(0.25))

    # conv block 5
    model.add(keras.layers.Conv2D(512,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(512,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(512,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes))

    return model


def VGG19(num_classes,img_shape):

    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=img_shape))

    # conv block 1
    model.add(keras.layers.Conv2D(64,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
    model.add(keras.layers.Dropout(0.25))


    # conv block 2

    model.add(keras.layers.Conv2D(128,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(128,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
    model.add(keras.layers.Dropout(0.25))


    # conv block 3
    model.add(keras.layers.Conv2D(256,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(256,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(256,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(256,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
    model.add(keras.layers.Dropout(0.25))


    # conv layer 4
    model.add(keras.layers.Conv2D(512,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(512,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(512,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(512,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
    model.add(keras.layers.Dropout(0.25))


    #conv layer 5
    model.add(keras.layers.Conv2D(512,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(512,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(512,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(512,(3,3),padding='same',strides=1,
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.L2(l2=5e-4),bias_regularizer=keras.regularizers.L2(l2=5e-4)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
    model.add(keras.layers.Dropout(0.25))


    # Dense layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes))

    return model
