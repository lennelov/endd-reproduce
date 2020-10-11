import tensorflow as tf
from tensorflow.math import lgamma
from tensorflow.math import digamma
'''
  This function computes the Reverse KL divergence between two dirichlet distributions based on their alphas.
    Inputs: 
          alpha_true (Batch_size x n_classes), the alphas corresponding to the true distribution. (1,1,1) for OOD and (100,1,1) / (1,100,1) / (1,1,100) for ID. 
          alpha_pred (Batch_size x n_classes), the predicted alphas from our network given input X. 
          epsilon = 1e-8, smoothing factor
    Output:
          KL-divergences (Batch-Size x 1), Reverse KL divergence of the two dirichlet distributions RKL(true_dirichlet||pred_dirichlet)
'''


class DirichletKL(tf.keras.losses.Loss):

    def __init__(self, epsilon=1e-10):
        super().__init__()
        self.epsilon = epsilon

    def call(self, alpha_true, alpha_pred):
        epsilon = self.epsilon
        KL = lgamma(tf.math.reduce_sum(alpha_pred)) - tf.math.reduce_sum(
            lgamma(alpha_pred + epsilon)) - lgamma(
                tf.math.reduce_sum(alpha_true)) + tf.math.reduce_sum(
                    lgamma(alpha_true + epsilon)) + tf.math.reduce_sum(
                        (alpha_pred - alpha_true) *
                        (digamma(alpha_pred + epsilon) - digamma(tf.math.reduce_sum(alpha_pred))))
        return KL
