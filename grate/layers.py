import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tqdm import trange
import time

class Core_layers(object):

    @staticmethod
    def multihead_graph_attention(node_features,
                        node_keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):

        node_features = tf.expand_dims(node_features,0)
        node_keys = node_features
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:  # set default size for attention size C
                num_units = node_features.get_shape().as_list()[-1]

            # Linear Projections
            Q = tf.layers.dense(node_features, num_units, activation=tf.nn.relu)  # [N, T_q, C]
            K = tf.layers.dense(node_keys, num_units, activation=tf.nn.relu)  # [N, T_k, C]
            V = tf.layers.dense(node_keys, num_units, activation=tf.nn.relu)  # [N, T_k, C]

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)  # [num_heads * N, T_q, C/num_heads]
            K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)  # [num_heads * N, T_k, C/num_heads]
            V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)  # [num_heads * N, T_k, C/num_heads]

            # Attention
            graph_attention = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (num_heads * N, T_q, T_k)

            # Scale : graph_attention = graph_attention / sqrt( d_k)
            graph_attention = graph_attention / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            # see : https://github.com/Kyubyong/transformer/issues/3
            key_masks = tf.sign(tf.abs(tf.reduce_sum(node_keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(node_features)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(graph_attention) * (-2 ** 32 + 1)  # -infinity
            graph_attention = tf.where(tf.equal(key_masks, 0), paddings, graph_attention)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(graph_attention[0, :, :])  # (T_q, T_k)
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(graph_attention)[0], 1, 1])  # (h*N, T_q, T_k)

                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                graph_attention = tf.where(tf.equal(masks, 0), paddings, graph_attention)  # (h*N, T_q, T_k)

            # Activation: graph_attention is a weight matrix
            graph_attention = tf.nn.softmax(graph_attention)  # (h*N, T_q, T_k)

            # node Masking
            node_masks = tf.sign(tf.abs(tf.reduce_sum(node_features, axis=-1)))  # (N, T_q)
            node_masks = tf.tile(node_masks, [num_heads, 1])  # (h*N, T_q)
            node_masks = tf.tile(tf.expand_dims(node_masks, -1), [1, 1, tf.shape(node_keys)[1]])  # (h*N, T_q, T_k)
            graph_attention *= node_masks  # broadcasting. (N, T_q, C)

            # dropouts
            graph_attention = tf.layers.dropout(graph_attention, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # weighted sum
            graph_attention = tf.matmul(graph_attention, V_)  # ( h*N, T_q, C/h)

            # reshape
            graph_attention = tf.concat(tf.split(graph_attention, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # residual connection
            graph_attention += node_features

            # layer normaliztion
            graph_attention = Core_layers.layer_normalization(graph_attention)
            graph_attention = tf.squeeze(graph_attention)
            return graph_attention


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
    def feedforward(inputs,
                    num_units=[2048, 512],
                    scope="multihead_attention",
                    reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                    "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                    "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            print("Conv ret:", outputs.shape)
            # Residual connection
            outputs += inputs

            # Normalize
            outputs = Core_layers.layer_normalization(outputs)

        return outputs
    
    @staticmethod
    def InnerProductDecoder(node_features, dropout = 0., act= tf.nn.sigmoid):
        """Decoder model layer for link prediction."""
        "Adjacency matrix Reconstruction"
        node_features = tf.nn.dropout(node_features, 1-self.dropout)
        x = tf.transpose(node_features)
        x = tf.matmul(node_features, x)
        x = tf.reshape(x, [-1])
        adj_reconstruction = self.act(x)
        return adj_reconstruction

    
    