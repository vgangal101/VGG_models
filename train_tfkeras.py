import tensorflow as tf
import argparse
import math
import models
import bn_vgg
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import matplotlib


# Todos: 
# 1. Write data augmentation code and evaluate performance 
#  -- a. should get around 93% accuracy roughly on both bn_vgg16 and bn_vgg19 
# THIS IS NOW RESOLVED , getting about 80% acc both places -- CAN RETURN TO THIS AFTER IMAGENET PREPARATION CODE IS WRITTEN


# 2. Write code to process imagenet on single gpu DONE 
#  -- b. should stop training and save a checkpoint when val acc is 50% IN PROGRESS 
#      -- b1 need code for imagenet data aug 




# 3. Write code to utlize data parallel training on gpus in to train imagenet VGG16 and VGG19 , original paper implementation



#https://towardsdatascience.com/neural-network-with-tensorflow-how-to-stop-training-using-callback-5c8d575c18a9
class at_acc_thresh(tf.keras.callbacks.Callback):
     
    
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('acc') > ACCURACY





def get_args():
    parser = argparse.ArgumentParser(description='training configurations')
    parser.add_argument('--model',type=str,help='choices are vgg11,vgg13,vgg16c,vgg16d,vgg19') # either vgg11,13,16,19 , now contains batch normalized options as well 
    parser.add_argument('--dataset',type=str,help='cifar10,cifar100,imagenet')
    parser.add_argument('--batch_size',type=int,default=256)
    # have the requirement that if the code is imagenet , then specify a path to dataset
    parser.add_argument('--imgnt_data_path',type=str,default='/data/petabyte/IMAGENET/Imagenet2012',help='only provide if imagenet is specified')
    parser.add_argument('--num_epochs',type=int,default=100,help='provide number of epochs to run')
    parser.add_argument('--lr',type=float,default=1e-2,help='learning rate to use')
    parser.add_argument('--lr_schedule',type=str,default='constant',help='choice of learning rate scheduler')
    parser.add_argument('--img_size',type=tuple, default=(224,224,3),help='imagenet crop size')
    parser.add_argument('--data_aug',type=bool,default=False,help='use data augmentation or not')
    parser.add_argument('--early_stopping', type=bool, default=False, help='use early stopping')
    parser.add_argument('--train_to_%_accuracy',type=float,default=0.5,help='using early stopping to train to certain percentage')
    parser.add_argument('--save_checkpoints',type=bool,default=False,help='whether to save checkpoints or not')
    args = parser.parse_args()
    return args


def data_augmentation(ds):
    """
    Performs 
    """
    AUTOTUNE = tf.data.AUTOTUNE
    
    augment = tf.keras.Sequential()
    augment.add(tf.keras.layers.RandomRotation(15/360))
    augment.add(tf.keras.layers.RandomWidth(0.1))
    augment.add(tf.keras.layers.RandomHeight(0.1))
    augment.add(tf.keras.layers.RandomFlip())
    
    
    ds = ds.map(lambda x, y:(augment(x),y),num_parallel_calls=AUTOTUNE)
    
    sample = ds.take(1)
    
    for d,l in sample: 
        print(d.shape)
    
    print(sample.shape)
    #ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds

def normalize_image(image,label):
    return tf.cast(image,tf.float32) / 255., label

def preprocess_dataset(args,train_dataset,test_dataset):
    """
    train_dataset : tf.data.Dataset
    test_dataset : tf.data.Dataset

    should return normalized image data + any data augmentation as needed. 
    
    THIS PORTION HERE REQUIRES ATTENTION
    """

    if args.dataset == 'imagenet':
        train_dataset = train_dataset.map(normalize_image)
        test_dataset = test_dataset.map(normalize_image)
    else:     
        train_dataset = train_dataset.map(normalize_image).batch(args.batch_size)
        test_dataset = test_dataset.map(normalize_image).batch(args.batch_size)
    
    return train_dataset, test_dataset


def get_imagenet_dataset(args):
    path_dir = args.imgnt_data_path
    
    if 'train' in path_dir or 'val' in path_dir : 
        raise ValueError('Specify the root directory not the train directory for the imagenet dataset')
    
    path_train = path_dir + '/train'
    path_val = path_dir + '/val'
    
    # specify image size ?????? -- lets set it to 224,224,3
    
    IMG_SIZE = None
    
    if args.img_size:
        IMG_SIZE = args.img_size[:2]
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(path_train,image_size=IMG_SIZE,batch_size=args.batch_size)
    val_dataset = tf.keras.utils.image_dataset_from_directory(path_val,image_size=IMG_SIZE, batch_size=args.batch_size)
    
    return train_dataset,val_dataset
    

def get_dataset(args):
    # should return a TF dataset object for train, test
    train_dataset = None
    test_dataset = None
    
    dataset_name = args.dataset
    
    if dataset_name.lower() == 'cifar10':
        (x_train,y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        return train_dataset, test_dataset
    elif dataset_name.lower() == 'cifar100':
        (x_train,y_train), (x_test,y_test) = tf.keras.datasets.cifar100.load_data()
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        return train_dataset,test_dataset
    elif dataset_name.lower() == 'imagenet':
        return get_imagenet_dataset(args) 

def plot_training(history,args):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.figure()
    plt.title("Epoch vs Accuracy")
    plt.plot(accuracy,label='training accuracy')
    plt.plot(val_accuracy,label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    
    viz_file = 'accuracy_graph_' + args.dataset.lower() + '_' + args.model.lower() + '_bs' + str(args.batch_size) + '_epochs' + str(args.num_epochs) + '.png'  
    plt.savefig(viz_file)        
    plt.show()
    
    
    plt.figure()
    plt.title("Epoch vs Loss")
    plt.plot(history.history['loss'],label='training loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    
    viz_file2 = 'loss_graph_' + args.dataset.lower() + '_' + args.model.lower() + '_bs' + str(args.batch_size) + '_epochs' + str(args.num_epochs) + '.png'  
    plt.savefig(viz_file2)        
    plt.show()        
        

                  
def main():
    
    print(matplotlib.get_backend())
    
    args = get_args()
    num_classes = None
    img_shape = None
    dataset_name = args.dataset
    
    if dataset_name == None:
        raise ValueError('No dataset specified. Exiting')
    elif dataset_name.lower() == 'cifar10':
        img_shape = (32,32,3)
        num_classes = 10
    elif dataset_name.lower() == 'cifar100':
        img_shape = (32,32,3)
        num_classes = 100
    elif dataset_name.lower() == 'imagenet':
        img_shape = (224,224,3)
        num_classes = 1000
    else:
        raise ValueError('Invalid dataset specified, dataset specified=', args.dataset)
       
    model = None
    if args.model.lower() == 'vgg11':
        model = models.VGG11_A(num_classes,img_shape)
    elif args.model.lower() == 'vgg13':
        model = models.VGG13_B(num_classes,img_shape)
    elif args.model.lower() == 'vgg16c':
        model = models.VGG16_C(num_classes,img_shape)
    elif args.model.lower() == 'vgg16d':
        model = models.VGG16_D(num_classes,img_shape)
    elif args.model.lower() == 'vgg19':
        model = models.VGG19_E(num_classes,img_shape)
    elif args.model.lower() == 'bn_vgg16':
        model = bn_vgg.bn_VGG16D(num_classes,img_shape)
    elif args.model.lower() == 'bn_vgg19':
        model = bn_vgg.bn_VGG19E(num_classes,img_shape)
    else:
        raise ValueError('Invalid value for the model name' + 'got model name' + args.model)

        
    callbacks = []
    optimizer = None
    
    if args.lr_schedule == 'constant':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    elif args.lr_schedule == 'time':
        decay = args.lr / args.num_epochs
        
        def lr_time_based_decay(epoch,lr):
            return lr * 1 / (1 + decay * epoch)
        lr_callback = LearningRateScheduler(lr_time_based_decay,verbose=1)
        
        callbacks.append(lr_callback)
    
    elif args.lr_schedule == 'step_decay':
        
        initial_learning_rate = args.lr
        
        def lr_step_decay(epoch,lr):
            drop_rate = 0.5
            epochs_drop = 10.0 
            return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))
    
        lr_callback = LearningRateScheduler(lr_step_decay,verbose=1)
        callbacks.append(lr_callback)
        
    elif args.lr_schedule == 'exp_decay':
        initial_learning_rate = args.lr
            
        def lr_exp_decay(epoch,lr):
            k = 0.1 
            return initial_learning_rate * math.exp(-k*epoch)
        
        lr_callback = LearningRateScheduler(lr_exp_decay,verbose=1)
        callbacks.append(lr_callback)
    
    else: 
        raise ValueError('invalid value for learning rate scheduler got: ', args.lr_scheduler)
        
    
    
    # early stopping 
    
    if args.early_stopping: 
        ea = tf.keras.callbacks.EarlyStopping(monitor='val_acc',patience=10,verbose=2)
    
        
        
    print("preparing data")
    train_dataset, test_dataset = get_dataset(args)    
    
    # potentially include data augmentation as a part of preprocessing ? 
    train_dataset, test_dataset = preprocess_dataset(args,train_dataset,test_dataset)

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=args.lr,momentum=0.9),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    print("starting training")
    history = model.fit(train_dataset,epochs=args.num_epochs,validation_data=test_dataset,callbacks=callbacks)
    #print('history.history.keys()=',history.history.keys())
    print('training complete')
    
    print('plotting...')
    plot_training(history,args)
    print('plotting complete')
    
    test_loss, test_acc = model.evaluate(test_dataset)
    print("test_loss=",test_loss)
    print("test_acc",test_acc)

    train_eval_log_file = open('./train_eval_file_' + args.model + '_' +  args.dataset + '_' + str(args.batch_size) + '.log', 'w')
    
    train_eval_log_file.write("test results\n")

    train_eval_log_file.write('test_loss' + '=' + str(test_loss) + '\n')
    train_eval_log_file.write('test_acc' + '=' + str(test_acc) + '\n')

    save_to_dir = args.model.lower() + '_' + args.dataset.lower() + '_' +  str(args.batch_size)
    model.save(save_to_dir)
    print("training and eval complete")





if __name__ == '__main__':
    main()