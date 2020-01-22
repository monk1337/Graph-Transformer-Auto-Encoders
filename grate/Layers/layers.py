import tensorflow as tf
from ..model_utils import Accuracy_metrices

import numpy as np
from tqdm import tqdm
from tqdm import trange
import time

class Core_layers(object):

    @staticmethod
    def multihead_graph_attention_v1(node_features,
                        node_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):


        
        node_features = tf.expand_dims(node_features,0)
        node_keys     = node_features


        with tf.variable_scope(scope, reuse=reuse):
            if node_units is None:  # set default size for attention size C
                node_units = node_features.get_shape().as_list()[-1]

            # Linear Projections
            Q = tf.layers.dense(node_features, node_units, activation=tf.nn.relu)  # [N, T_q, C]
            K = tf.layers.dense(node_keys, node_units, activation=tf.nn.relu)  # [N, T_k, C]
            V = tf.layers.dense(node_keys, node_units, activation=tf.nn.relu)  # [N, T_k, C]

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)  # [num_heads * N, T_q, C/num_heads]
            K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)  # [num_heads * N, T_k, C/num_heads]
            V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)  # [num_heads * N, T_k, C/num_heads]

            # Attention
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (num_heads * N, T_q, T_k)

            # Scale : outputs = outputs / sqrt( d_k)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            # see : https://github.com/Kyubyong/transformer/issues/3
            key_masks = tf.sign(tf.abs(tf.reduce_sum(node_keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(node_features)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # -infinity
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation: outputs is a weight matrix
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(node_features, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(node_keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)

            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # reshape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # residual connection
            outputs += node_features

            # layer normaliztion
            outputs = Core_layers.layer_normalization(outputs)

            return tf.squeeze(outputs)

    @staticmethod
    def add_and_normalize(node_features, attention_output):
        # residual connection
        attention_output += node_features

        # layer normaliztion
        outputs = Core_layers.layer_normalization(attention_output)
        outputs = tf.squeeze(outputs)
        return outputs

    @staticmethod
    def layer_normalization(inputs,
                        epsilon=1e-8,
                        scope="ln",
                        reuse=None):
        
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** .5)
            outputs = gamma * normalized + beta

        return outputs

    @staticmethod
    def feed_forward(attention_input,node_units, act = tf.nn.relu, bias = True):

        
        # dense layer with xavier weights
        fc_layer = tf.get_variable(name='feed_forward',
                                   shape=[attention_input.shape[-1],node_units],
                                   dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        
        if bias:
            # bias 
            bias    = tf.get_variable(name='bias',
                                    shape=[node_units],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
        
            #final output 
            output = act(tf.add(tf.matmul(attention_input,fc_layer),bias))
        else:
            output = act(tf.matmul(attention_input,fc_layer))

        return output

    

    @staticmethod
    def InnerProductDecoder(latent_input,
                            dropout=0., 
                            act=tf.nn.sigmoid):

        inputs = tf.nn.dropout(latent_input, 1- dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = act(x)
        return outputs

    @staticmethod
    def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    

        E = inputs.get_shape().as_list()[-1] # static
        N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
                for pos in range(maxlen)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)

            # masks
            if masking:
                outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

            return tf.to_float(outputs)

class Training(object):

    @staticmethod
    def write_result(model_name, result):
        with open(str(model_name),'a') as f:
            f.write(str(result)+'\n')

    @staticmethod
    # data dict names and placeholders name should be same
    def fit(epochs, model, data, task):
        
        feed_dict = dict()
        for placeholder_name, content in model.placeholders.items():
                if placeholder_name in data:
                    feed_dict[content] = data[placeholder_name]
                    
        cost_val = []
        acc_val = []
        val_roc_score = []
        
        # initiating the session 
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            t = trange(epochs, desc='Bar desc', leave=True)
            
            if task == 'node_prediction':
                for epoch in t:
                    ts = time.time()
                    outs = sess.run([n for m,n in model.optimizer.items()], feed_dict=feed_dict)

                    avg_cost = outs[1]
                    avg_accuracy = outs[2]
                    roc_curr, ap_curr = Accuracy_metrices.get_roc_score(data['val_edges'], data['val_edges_false'],feed_dict = feed_dict, 
                                                                        placeholders = model.placeholders, sess=sess, model = model, 
                                                                        adj_orig_sp = data['adj_orig_sp'])
                    val_roc_score.append(roc_curr)
                    result_ = "epoch {},  train_loss {},  train_acc {},  val_roc {}, val_ap{}".format((epoch + 1),
                                                                                                 avg_cost,
                                                                                                avg_accuracy, 
                                                                                                            val_roc_score[-1], 
                                                                                                            ap_curr)
                    
                    print(result_)
                    Training.write_result(task, result_)
                    t.refresh() # to show immediately the update
                    
                    
                roc_score, ap_score = Accuracy_metrices.get_roc_score(data['test_edges'], data['test_edges_false'],feed_dict = feed_dict, 
                                                                        placeholders = model.placeholders, sess=sess, model = model, 
                                                                        adj_orig_sp = data['adj_orig_sp'])
                print('Test ROC score: ' + str(roc_score))
                print('Test AP score: ' + str(ap_score))
                Training.write_result(task, roc_score)
                Training.write_result(task, ap_score)
