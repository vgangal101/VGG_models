import tensorflow as tf 
from tensorflow import keras 
from train_utils.preprocessing import imgnt_mean_substract2
import argparse
import atexit


def get_args():
    parser = argparse.ArgumentParser(description='training configurations')
    parser.add_argument('--checkpoint_dir',type=str,default='./checkpoints',help='where to save checkpoints')
    parser.add_argument('--num_gpus',type=int,default=1,help='number of gpus to use (on node)')
    parser.add_argument('--img_size',type=tuple, default=(224,224,3),help='imagenet crop size')
    parser.add_argument('--imgnt_data_path',type=str,default='/data/petabyte/IMAGENET/Imagenet2012',help='only provide if imagenet is specified')
    parser.add_argument('--batch_size',type=int,default=256)
    args = parser.parse_args()
    return args


def preprocess_imgnt_val(test_ds):
    test_ds = test_ds.map(imgnt_mean_substract2)
    return test_ds


def get_imgnt_validation_set(args):
    print('loading imagenet dataset')
    path_dir = args.imgnt_data_path
    
    path_val = path_dir + '/val'
    IMG_SIZE = args.img_size[:2]

    val_dataset = tf.keras.utils.image_dataset_from_directory(path_val,image_size=IMG_SIZE, batch_size=args.batch_size)

    return val_dataset


def main():
    args = get_args()
    val_dataset = get_imgnt_validation_set(args)
    val_ds = preprocess_imgnt_val(val_dataset)
   

    print('setting up distribution strategy')
    gpus = tf.config.list_logical_devices('GPU')
    print('number of gpus used = ',args.num_gpus)
    strategy = tf.distribute.MirroredStrategy(gpus[:args.num_gpus])

    with strategy.scope():
        recovered_model = tf.keras.models.load_model(args.checkpoint_dir)
        test_loss, test_acc = recovered_model.evaluate(val_ds)
    
    print('test_loss=',test_loss)
    print('test_acc=',test_acc)
    atexit.register(strategy._extended._collective_ops._pool.close)

    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    





