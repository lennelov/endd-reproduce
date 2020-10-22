import tensorflow as tf
from tensorflow.math import exp, reduce_sum, lgamma, reduce_mean, log, digamma
from tensorflow.nn import softmax


class DirichletEnDDLoss(tf.keras.losses.Loss):
    """
    Negative Log-likelihood of the model on the transfer dataset"""

    def __init__(self, epsilon=1e-8, teacher_epsilon=1e-3, init_temp=2.5):
        super().__init__()
        self.smooth_val = epsilon
        self.tp_scaling = 1 - teacher_epsilon
        self.init_temp = init_temp
        self.temp = None

    def call(self, ensemble_logits, logits):
        '''
        teacher_logits are the outputs from our ensemble (batch x ensembles x classes)
        logits are the predicted outputs from our model (batch x classes)
        '''
        if self.temp is None:
            self.temp = self.init_temp

        logits = tf.cast(logits, dtype=tf.float64)
        ensemble_logits = tf.cast(ensemble_logits, dtype=tf.float64)
        alphas = exp(logits / self.temp)

        precision = reduce_sum(alphas, axis=1)  #sum over classes

        ensemble_probs = softmax(ensemble_logits / self.temp, axis=2)  #softmax over classes
        # Smooth for num. stability:
        probs_mean = 1 / (tf.shape(ensemble_probs)[2])  #divide by nr of classes
        # Subtract mean, scale down, add mean back)
        teacher_probs = self.tp_scaling * (ensemble_probs - probs_mean) + probs_mean

        log_ensemble_probs_geo_mean = reduce_mean(log(ensemble_probs + self.smooth_val),
                                                  axis=1)  #mean over ensembles

        target_independent_term = reduce_sum(lgamma(alphas + self.smooth_val), axis=1) - lgamma(
            precision + self.smooth_val)  #sum over lgammma of classes - lgamma(precision)

        target_dependent_term = -reduce_sum(
            (alphas - 1.) * log_ensemble_probs_geo_mean, axis=1)  # -sum over classes

        cost = target_dependent_term + target_independent_term

        return reduce_mean(cost) * (self.temp**2)  #mean of all batches


class DirichletKL(tf.keras.losses.Loss):
    '''
      This function computes the Reverse KL divergence between two dirichlet distributions based on their alphas.
        Inputs:
              alpha_true (Batch_size x n_classes), the alphas corresponding to the true distribution. (1,1,1) for OOD and (100,1,1) / (1,100,1) / (1,1,100) for ID.
              logits_pred (Batch_size x n_classes), the predicted logits from our network given input X.
              epsilon = 1e-10, smoothing factor
        Output:
              KL-divergences (Batch-Size x 1), Reverse KL divergence of the two dirichlet distributions RKL(true_dirichlet||pred_dirichlet)
    '''

    def __init__(self, epsilon=1e-10):
        super().__init__()
        self.epsilon = epsilon

    def call(self, alpha_true, logits_pred):
        epsilon = self.epsilon
        alpha_pred = exp(logits_pred)
        KL = lgamma(tf.math.reduce_sum(alpha_pred)) - tf.math.reduce_sum(
            lgamma(alpha_pred + epsilon)) - lgamma(
                tf.math.reduce_sum(alpha_true)) + tf.math.reduce_sum(
                    lgamma(alpha_true + epsilon)) + tf.math.reduce_sum(
                        (alpha_pred - alpha_true) *
                        (digamma(alpha_pred + epsilon) - digamma(tf.math.reduce_sum(alpha_pred))))
        return KL
