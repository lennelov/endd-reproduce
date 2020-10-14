
import tensorflow as tf
from tensorflow.math import exp,reduce_sum,lgamma,reduce_mean,log
from tensorflow.nn import softmax
class DirichletEnDDLoss(tf.keras.losses.Loss):
    """
    Negative Log-likelihood of the model on the transfer dataset"""
    def __init__(self, epsilon=1e-8,teacher_epsilon = 1e-3):
        super().__init__()
        self.smooth_val = epsilon
        self.tp_scaling = 1-teacher_epsilon
    def call(self, ensemble_logits,logits,temp = 2.5):
        '''
        teacher_logits are the outputs from our ensemble (batch x ensembles x classes)
        logits are the predicted outputs from our model (batch x classes)
        '''
        logits = tf.cast(logits,dtype = tf.float64)
        ensemble_logits = tf.cast(ensemble_logits,dtype = tf.float64)
        alphas = exp(logits / temp)

        precision = reduce_sum(alphas, axis=1) #sum over classes

        ensemble_probs = softmax(ensemble_logits / temp, axis=2) #softmax over classes
        # Smooth for num. stability:
        probs_mean = 1 / (ensemble_probs.shape[2]) #divide by nr of classes
        # Subtract mean, scale down, add mean back)
        teacher_probs = self.tp_scaling * (ensemble_probs - probs_mean) + probs_mean

        log_ensemble_probs_geo_mean = reduce_mean(log(ensemble_probs + self.smooth_val), axis=1) #mean over ensembles

        target_independent_term = reduce_sum(lgamma(alphas + self.smooth_val), axis=1) - lgamma(precision + self.smooth_val) #sum over lgammma of classes - lgamma(precision)

        target_dependent_term = - reduce_sum((alphas - 1.) * log_ensemble_probs_geo_mean, axis=1) # -sum over classes

        cost = target_dependent_term + target_independent_term

        return reduce_mean(cost) * (temp ** 2) #mean of all batches
