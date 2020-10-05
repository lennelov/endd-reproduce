import keras
from datetime import datetime
from packaging import version
def train_priornet_toy_dataset(x_train,logits_train,model,batch = 100,n_epochs = 40):
        '''
        trains the given model on the given data.
        Inputs:
        x_train, toy_dataset (x,y)-value
        logits_train,logits for each training point
        model, the model to train
        batch = 100, batch_size
        n_epochs = 40,  number of epochs for training
        '''
        logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        model.fit(
            x = x_train,
            y = logits_train,
            batch_size = batch,epochs = n_epochs,
                        callbacks = tensorboard_callback)
        return model
