import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

class Model_utils(object):
    @staticmethod
    def placeholders(placeholder_dict):
        placeholders = {}
        for name, data in placeholder_dict.items():
            if data:
                placeholders[name] = tf.placeholder(dtype = tf.float32,shape=data)
            else:
                placeholders[name] = tf.placeholder(dtype = tf.float32)
        return placeholders

    
    @staticmethod
    def pos_and_norm(data):

        adj_matrix     = data['adj']
        feature_matrix = data['features']
    
        num_nodes    = adj_matrix.shape[0]
        num_features = feature_matrix.shape[1]
        
        pos_weight = float(adj_matrix.shape[0] * adj_matrix.shape[0] - adj_matrix.sum()) / adj_matrix.sum()
        norm = adj_matrix.shape[0] * adj_matrix.shape[0] / float((adj_matrix.shape[0] * adj_matrix.shape[0] - adj_matrix.sum()) * 2)
        
        return {
                'num_nodes': num_nodes, 
                'num_features': num_features, 
                'pos_weight': pos_weight, 
                'norm': norm
                }

    
class Accuracy_metrices(object):

    @staticmethod
    def get_roc_score(edges_pos, edges_neg, feed_dict, placeholders, sess, model, adj_orig_sp, emb = None):
        if emb is None:
            feed_dict.update({placeholders['dropout']: 0})
            emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(adj_orig_sp[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(adj_orig_sp[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score
    