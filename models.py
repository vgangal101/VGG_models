import tensorflow as tf
from tensorflow import keras

class VGG11_A(tf.keras.Model):
    def __init__(self,num_classes,img_shape):
        super().__init__()
        self.model = self.build_model(num_classes,img_shape)

    def build_model(self,num_classes,img_shape):
        model = keras.Sequential()
        #model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.InputLayer(input_shape=img_shape))
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
        super().__init__()
        self.model = build_model(self,num_classes,img_shape)

    def build_model(self,num_classes,img_shape):
        model = keras.Sequential()
        #model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.InputLayer(input_shape=img_shape))

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

class VGG16_C(tf.keras.Model):
    def __init__(self,num_classes,img_shape):
        super().__init__()
        self.model = self.build_model(num_classes,img_shape)

    def build_model(self,num_classes,img_shape):
        model = keras.Sequential()
        #model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.InputLayer(input_shape=img_shape))


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
        model.add(keras.layers.Dense(num_classes))

        return model

    def call(self,x):
        out = self.model(x)
        return out

class VGG16_D(tf.keras.Model):
    def __init__(self,num_classes,img_shape):
        super().__init__()
        self.model = self.build_model(num_classes,img_shape)

    def build_model(self,num_classes,img_shape):
        model = keras.Sequential()

        #model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.InputLayer(input_shape=img_shape))


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
        model.add(keras.layers.Dense(num_classes))

        return model

    def call(self,x):
        out = self.model(x)
        return out

class VGG19_E(tf.keras.Model):
    def __init__(self,num_classes,img_shape):
        super().__init__()
        self.model = self.build_model(num_classes,img_shape)

    def build_model(self,num_classes,img_shape):
        model = keras.Sequential()

        #model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.InputLayer(input_shape=img_shape))

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
        model.add(keras.layers.Dense(num_classes))

        return model

    def call(self,x):
        out = self.model(x)
        return out
