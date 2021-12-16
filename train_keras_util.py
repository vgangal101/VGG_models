import tensorflow as tf
import argparse
import models2
from tensorflow import keras


# 1. first prepare code on the cifar10 dataset
# 2. imagenet is second step ,

def get_args():
    parser = argparse.ArgumentParser(description='training configurations')
    parser.add_argument('--model',type=str,help='choices are vgg11,vgg13,vgg16,vgg19') # either vgg11,13,16,19
    parser.add_argument('--dataset',type=str,help='cifar10,cifar100,imagenet') # either 'cifar10' or 'imagenet'
    parser.add_argument('--batch_size',type=int,default=256)
    # have the requirement that if the code is imagenet , then specify a path to dataset
    parser.add_argument('--data_path',type=str,help='only provide if imagenet is specified')
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

def main():
    args = get_args()
    num_classes = None
    img_shape = None
    dataset_name = args.dataset
    if dataset_name.lower() == 'cifar10':
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
        model = models2.VGG11_A(num_classes,img_shape)
    elif args.model.lower() == 'vgg13':
        model = models2.VGG13_B(num_classes,img_shape)
    elif args.model.lower() == 'vgg16':
        model = models2.VGG16_D(num_classes,img_shape)
    elif args.model.lower() == 'vgg19':
        model = models2.VGG19_E(num_classes,img_shape)
    else:
        raise ValueError('Invalid value for the model name' + 'got model name' + args.models)


    print("preparing data")
    train_dataset, test_dataset = get_dataset(args.dataset)
    train_dataset, test_dataset = preprocess_dataset(args,train_dataset,test_dataset)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-9),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    print("starting training")
    history = model.fit(train_dataset,epochs=50)

    results = model.evaluate(test_dataset)
    print(results)

    train_eval_log_file = open('./train_eval_file_' + args.model + '_' +  args.dataset + '_' + args.batch_size + '.log', 'w')
    
    train_eval_log_file.write('training metrics\n')
    train_eval_log_file.write('\n')
    
    for key in history:    
        train_eval_log_file.write(str(key) + '=' + history(key) + '\n')

    
    train_eval_log_file.write('\n')
    
    train_eval_log_file.write('eval results\n')

    for key in results: 
        train_eval_log_file.write(str(key) + '=' + results(key) + '\n')


    save_to_dir = args.model.lower() + '_' + args.dataset.lower()
    model.save(save_to_dir)
    print("training and eval complete")





if __name__ == '__main__':
    main()
