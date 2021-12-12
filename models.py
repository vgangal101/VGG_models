import tensorflow as tf
from tensorflow import keras

class VGG11_A(tf.keras.Model):
    def __init__(self,num_classes):
        self.model = build_model(num_classes)

    def build_model(self):
        model = keras.Sequential()
        #model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))
        model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))
        model.add(keras.layers.Conv2D(256,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(256,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))
        model.add(keras.layers.Dense(4096,activation='relu'))
        model.add(keras.layers.Dense(4096,activation='relu'))
        model.add(keras.layers.Dense(num_classes))

        return model

    def call(self,x):
        out = self.model(x)
        return out


class VGG13_B(tf.keras.Model):
    def __init__(self):
        self.model = build_model(num_classes)

    def build_model(self,num_classes):
        model = keras.Sequential()
        #model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(256,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(256,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Dense(4096,activation='relu'))
        model.add(keras.layers.Dense(4096,activation='relu'))
        model.add(keras.layers.Dense(num_classes))

        return model

    def call(self,x):
        out = self.model(x)
        return out

class VGG16_B(tf.keras.Model):
    def __init__(self):
        self.model = build_model(num_classes)

    def build_model(self,num_classes):
        model = keras.Sequential()
        #model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(256,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(256,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(256,(1,1),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(1,1),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(1,1),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Dense(4096, activation='relu'))
        model.add(keras.layers.Dense(4096, activation='relu'))
        model.add(keras.layers.Dense(num_classes)

        return model

    def call(self,x):
        out = self.model(x)
        return out

class VGG16_D:
    def __init__(self):
        self.model = build_model()

    def build_model(self,num_classes):
        model = keras.Sequential()

        #model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(256,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(256,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(256,(1,1),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(1,1),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Dense(4096, activation='relu'))
        model.add(keras.layers.Dense(4096, activation='relu'))
        model.add(keras.layers.Dense(num_classes)

        return model

    def call(self,x):
        out = self.model(x)
        return out

class VGG19_D:
    def __init__(self):
        self.model = build_model()

    def build_model(self,num_classes):
        model = keras.Sequential()

        #model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(256,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(256,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(256,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(256,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))

        model.add(keras.layers.Dense(4096, activation='relu'))
        model.add(keras.layers.Dense(4096, activation='relu'))
        model.add(keras.layers.Dense(num_classes)

        return model

    def call(self,x):
        out = self.model(x)
        return out
