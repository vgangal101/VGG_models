import tensorflow as tf
import argparse
import math
import models
import bn_vgg
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Todos: 
# 1. Write data augmentation code and evaluate performance 
#  -- a. should get around 93% accuracy roughly on both bn_vgg16 and bn_vgg19 
# 2. Write code to process imagenet on single gpu
#  -- b. should stop training and save a checkpoint when val acc is 50%
# 3. Write code to utlize data parallel training on gpus in to train imagenet VGG16 and VGG19 , batch normalized ? 

def get_args():
    parser = argparse.ArgumentParser(description='training configurations')
    parser.add_argument('--model',type=str,help='choices are vgg11,vgg13,vgg16,vgg16,vgg19') # either vgg11,13,16,19 , now contains batch normalized options as well 
    parser.add_argument('--dataset',type=str,help='cifar10,cifar100,imagenet')
    parser.add_argument('--batch_size',type=int,default=256)
    # have the requirement that if the code is imagenet , then specify a path to dataset
    parser.add_argument('--data_path',type=str,help='only provide if imagenet is specified')
    parser.add_argument('--num_epochs',type=int,default=100,help='provide number of epochs to run')
    parser.add_argument('--lr',type=float,default=1e-2,help='learning rate to use')
    parser.add_argument('--lr_schedule',type=str,default='constant',help='choice of learning rate scheduler')
    parser.add_argument('--data_aug',type=bool,default=False,help='use data augmentation or not')
    args = parser.parse_args()
    return args


def normalize_image(image,label):
    return tf.cast(image,tf.float32) / 255., label

def preprocess_dataset(args,train_dataset,test_dataset):
    """
    train_dataset : tf.data.Dataset
    test_dataset : tf.data.Dataset

    should return normalized image data + any data augmentation as needed. 
    """
    
    if args.data_aug:
        train_dataset = data_augmentation(train_dataset)
   
    train_dataset = train_dataset.map(normalize_image).batch(args.batch_size)
    test_dataset = test_dataset.map(normalize_image).batch(args.batch_size)
    
    return train_dataset, test_dataset

def get_dataset(dataset_name):
    # should return a TF dataset object for train, test
    train_dataset = None
    test_dataset = None
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
        raise NotImplementedError

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
        raise NotImplementedError
        #num_classes = 1000
    else:
        raise ValueError('Invalid dataset specified, dataset specified= ', args.dataset)

    model = None
    if args.model.lower() == 'bn_vgg16':
        model = bn_vgg.bn_VGG16D(num_classes,img_shape)
    elif args.model.lower() == 'bn_vgg19':
        model = bn_vgg.bn_VGG19E(num_classes,img_shape)
    else:
        raise ValueError('Invalid value for the model name' + 'got model name= ' + args.model)

        
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
            epochs_drop = 20 
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
        
    print("preparing data")
    
    (x_train,y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
    
    datagen = ImageDataGenerator(
                featurewise_center = False, # set input mean to 0 over the dataset 
                samplewise_center = False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=False)
    
    
    datagen.fit(x_train)
    
    
    # compiling model stats 
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=args.lr,momentum=0.9),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    print("starting training")
    
    history = model.fit(datagen.flow(x_train,y_train,batch_size=args.batch_size),
                        epochs=args.num_epochs,
                        validation_data=(x_test,y_test),
                        callbacks=callbacks,verbose=2)
    
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

    save_to_dir = args.model.lower() + '_' + args.dataset.lower() + str(ags.batch_size)
    model.save(save_to_dir)
    print("training and eval complete")





if __name__ == '__main__':
    main()
