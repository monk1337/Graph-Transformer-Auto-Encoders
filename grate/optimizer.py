import tensorflow as tf
from utils import Model_utils

class Optimizers(object):

    @staticmethod
    def OptimizerGrate(preds, labels, data, learning_rate = 0.01):
        
        adj_matrix     = data['adj']
        feature_matrix = data['features']

        pos_norm = Model_utils.pos_and_norm(feature_matrix,adj_matrix)
        preds_sub = preds
        labels_sub = labels

        cost = pos_norm['norm'] * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, 
                                                                                targets=labels_sub, 
                                                                                pos_weight=pos_norm['pos_weight']))
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)  # Adam Optimizer

        opt_op = optimizer.minimize(cost)
        grads_vars = optimizer.compute_gradients(cost)

        correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        return {
                'opt_op': opt_op , 
                'grads_vars': grads_vars,
                'correct_prediction': correct_prediction, 
                'accuracy': accuracy
                }




