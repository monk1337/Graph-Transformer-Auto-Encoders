import tensorflow as tf
from ..model_utils import Model_utils




class Optimizers(object):

    @staticmethod
    def OptimizerGrate(latent_space, model_data, placeholders, learning_rate = 0.001):

        pos_data = Model_utils.pos_and_norm(model_data)
        pos_weight = pos_data['pos_weight']
        norm       = pos_data['norm']

        preds_sub = latent_space
        labels_sub = tf.reshape(placeholders['adj_orig'],[-1])

        cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Adam Optimizer

        opt_op = optimizer.minimize(cost)
        grads_vars = optimizer.compute_gradients(cost)
        correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return {'cost': cost, 
                'opt_op': opt_op, 
                'accuracy': accuracy
                }



    @staticmethod
    def OptimizerGratVae(preds, labels, model, num_nodes, pos_weight, norm, learning_rate):
        preds_sub = preds
        labels_sub = labels

        cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)  # Adam Optimizer

        # Latent loss
        log_lik = cost
        kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        cost -= kl

        opt_op = optimizer.minimize(cost)
        grads_vars = optimizer.compute_gradients(cost)
        correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return {'cost': cost, 
                'opt_op': opt_op, 
                'accuracy': accuracy
                }
    

    @staticmethod
    def multilabel_optimizer(logits, ground_truth, learning_rate):
        
        cross_entropy    = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, 
                                                                   labels = tf.cast(ground_truth,
                                                                                    tf.float32))
        loss             = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))
        optimizer        = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
        
        logits_prob      = tf.nn.sigmoid(logits, name = 'prob')
        predictions      = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32,name='predictions')
        
        return {
                'loss'      : loss, 
                'optimizer' : optimizer, 
                'prediction': predictions, 
                'log_prob'  : logits_prob
                }
        
    @staticmethod
    def multiclass_optimizer(logits, ground_truth, learning_rate):
        
        cross_entropy   = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, 
                                                                     labels = tf.cast(ground_truth,
                                                                                      tf.float32))
        loss            = tf.reduce_mean(cross_entropy)
        optimizer       = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(loss)
        
        logits_prob     = tf.nn.softmax(logits, name = 'prob')
        predictions     = tf.argmax(logits_prob, axis=1, name='predictions')
  
        y_true          = tf.argmax(ground_truth, axis=1, name='ground_truth')
        accuracy        = tf.reduce_mean(tf.cast(tf.equal(predictions, y_true), tf.float32))
        
        return {
            
                'loss'       : loss, 
                'optimizer'  : optimizer, 
                'prediction' : predictions, 
                'accuracy'   : accuracy, 
                'log_prob'   : logits_prob
                }

    @staticmethod
    def masked_softmax_cross_entropy(preds, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)

        mask /= tf.reduce_mean(mask)
        loss *= mask

        return tf.reduce_mean(loss)

    @staticmethod
    def masked_accuracy(preds, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        
        return tf.reduce_mean(accuracy_all)
