import tensorflow as tf
from tensorflow import keras

def normalize_image(image,label):
    return tf.cast(image,tf.float32) / 255., label


def imgnt_mean_substract1(image,label):
    #print(type(image))
    #print(image.shape)
    mean = [103.939, 116.779, 123.68]
    image[..., 0] -= mean[0]
    image[..., 1] -= mean[1]
    image[..., 2] -= mean[2]
    return image, label

def imgnt_mean_substract2(image,label):
    image /= 127.5
    image -= 1.
    return image,label


def imgnt_preproc(train_ds,test_ds):
    train_ds = train_ds.map(imgnt_mean_substract2)
    test_ds = test_ds.map(imgnt_mean_substract2)
    return train_ds, test_ds

def cifar10_preproc(train_ds,test_ds):
    train_ds = train_ds.map(normalize_image)
    test_ds = test_ds.map(normalize_image)
    return train_ds, test_ds


def cifar100_preproc(train_ds,test_ds):
    train_ds = train_ds.map(normalize_image)
    test_ds = test_ds.map(normalize_image)
    return train_ds, test_ds
