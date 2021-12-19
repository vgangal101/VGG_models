import tensorflow as tf
import argparse
import models
from tensorflow import keras
import matplotlib.pyplot as plt

# 1. first prepare code on the cifar10 dataset
#    make sure you are able to train efectively, build graph training functionality 
# 2. imagenet is second step ,

def get_args():
    parser = argparse.ArgumentParser(description='training configurations')
    parser.add_argument('--model',type=str,help='choices are vgg11,vgg13,vgg16,vgg19') # either vgg11,13,16,19
    parser.add_argument('--dataset',type=str,help='cifar10,cifar100,imagenet') # either 'cifar10' or 'imagenet'
    parser.add_argument('--batch_size',type=int,default=64)
    # have the requirement that if the code is imagenet , then specify a path to dataset
    parser.add_argument('--data_path',type=str,help='only provide if imagenet is specified')
    parser.add_argument('--num_epochs',type=int,default=50,help='provide number of epochs to run')
    args = parser.parse_args()
    return args



def normalize_image(image,label):
    return tf.cast(image,tf.float32) / 255., label

def preprocess_dataset(args,train_dataset,test_dataset):
    """
    train_dataset : tf.data.Dataset
    test_dataset : tf.data.Dataset

    should return normalized image data
    """

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
    else:
        raise ValueError('Invalid value for the model name' + 'got model name' + args.models)

        
        

    print("preparing data")
    train_dataset, test_dataset = get_dataset(args.dataset)
    train_dataset, test_dataset = preprocess_dataset(args,train_dataset,test_dataset)

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-2,momentum=0.9),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    print("starting training")
    history = model.fit(train_dataset,epochs=args.num_epochs,validation_data=test_dataset)
    print('history.history.keys()=',history.history.keys())
    print('training complete')
    
    print('plotting...')
    plot_training(history,args)
    print('plotting complete')
    
    test_loss, test_acc = model.evaluate(test_dataset)
    print("test_loss=",test_loss)
    print("test_acc",test_acc)

    train_eval_log_file = open('./train_eval_file_' + args.model + '_' +  args.dataset + '_' + str(args.batch_size) + '.log', 'w')
    
    #train_eval_log_file.write('training metrics\n')
    #train_eval_log_file.write('\n')
    
    #for key in history.history:    
    #   train_eval_log_file.write(str(key) + '=' + history.history[key] + '\n')

    
    #train_eval_log_file.write('\n')
    
    #train_eval_log_file.write('eval results\n')
 
    train_eval_log_file.write("test results\n")

    train_eval_log_file.write('test_loss' + '=' + str(test_loss) + '\n')
    train_eval_log_file.write('test_acc' + '=' + str(test_acc) + '\n')

    save_to_dir = args.model.lower() + '_' + args.dataset.lower()
    model.save(save_to_dir)
    print("training and eval complete")





if __name__ == '__main__':
    main()
