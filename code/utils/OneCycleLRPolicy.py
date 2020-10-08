import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class OneCycleLRPolicy(tf.keras.callbacks.Callback):
    """This callback implements a variant of the 1-cycle learning rate policy
    as introduced by the paper https://arxiv.org/pdf/1708.07120.pdf. 

    This particular variant is the same as used in https://arxiv.org/pdf/1802.10501.pdf,
    where the learning rate is linearly increased from the inital to the maximum 
    over half the cycle length, and then linearly decreased to the initial over the
    half. It is then linearly decreased to min_lr over remaining epochs. 


    # Arguments
        init_lr      : the initial learning rate
        max_lr       : the maxumum learning rate
        min_lr       : the minimum learning rate
        cycle_length : the length of the cycle, in epcohs, must be even
        epochs       : the total number of epochs, must be larger than cycle_length

    """

    def __init__(self, init_lr, max_lr, min_lr, cycle_length, epochs):
        assert (cycle_length % 2 == 0)
        assert (epochs > cycle_length)
        self.schedule = np.hstack(
            (np.linspace(init_lr, max_lr,
                         cycle_length // 2), np.linspace(max_lr, init_lr, cycle_length // 2),
             np.linspace(init_lr, min_lr, epochs - cycle_length)))

    def on_epoch_begin(self, epoch, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.schedule[epoch])

    def plot(self):
        plt.plot(self.schedule, '.-')
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title("One-cycle learning rate policy")
        plt.show()


#olp = OneCyclePolicy(0.001, 0.01, 0.000001, 30, 45)
#print(olp.schedule)
#print(olp.schedule.size)
#olp.plot()
