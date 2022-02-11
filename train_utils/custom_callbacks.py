import tensorflow as tf

class stop_acc_thresh(tf.keras.callbacks.Callback):
    """
    callback to stop training when a certain validation accuracy is reached
    """
    def __init__(self,acc):
        super(stop_acc_thresh,self).init__()
        self.acc_thresh = acc

    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('val_accuracy') > self.acc_thresh):
            print("\n Reached %2.2f accuracy" %(self.acc_thresh*100))
            self.model.stop_training = True
        print('val acc = %2.2f' %(logs.get('val_acc')))
