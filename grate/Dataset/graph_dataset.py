import pickle as pkl
import scipy.sparse as sp
import random
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
import numpy as np

class graph_preprocessing(object):
    
    @staticmethod

    def transform_semi_supervise(labels, mask_nodes_):

        # labels : node labels as list [ 1,2,3,4,5]
        # mask_nodes : no of nodes will be use for semi-supervised traning let say we have 5 classes total 34 samples then I am using 
        # 5 labels for ssl setting so mask_nodes will be 5
    
        nb_node_classes = len(set(labels))
        
        targets = np.array([labels], dtype=np.int32).reshape(-1)
        one_hot_nodes = np.eye(nb_node_classes)[targets]
        
        
        # Pick one at random from each class
        labels_to_keep = [np.random.choice(
        np.nonzero(one_hot_nodes[:, c])[0]) for c in range(mask_nodes_)]
        
        y_train = np.zeros(shape=one_hot_nodes.shape,
                    dtype=np.float32)
        y_val = one_hot_nodes.copy()
        
        train_mask = np.zeros(shape=(len(labels),), dtype=np.bool)
        val_mask = np.ones(shape=(len(labels),), dtype=np.bool)
        
        
        for l in labels_to_keep:
            y_train[l, :] = one_hot_nodes[l, :]
            y_val[l, :] = np.zeros(shape=(nb_node_classes,))
            train_mask[l] = True
            val_mask[l] = False
            
        return {
                'all_labels': labels, 
                'train_labels': y_train, 
                'val_labels': y_val, 
                'train_mask': train_mask, 
                'val_mask': val_mask 
                }
    @staticmethod
    def sparse_to_tuples(sparse_mx):
        """Convert sparse matrix to tuple representation."""
        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    @staticmethod
    def normalize_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    @staticmethod
    def preprocess_adj(adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        adj_normalized = graph_preprocessing.normalize_adj(adj + sp.eye(adj.shape[0]))
        return graph_preprocessing.sparse_to_tuple(adj_normalized)


    @staticmethod
    def preprocess_features(features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return graph_preprocessing.sparse_to_tuple(features)

    @staticmethod
    def parse_index_file(filename):
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    @staticmethod
    def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    @staticmethod
    def sparse_to_tuple(sparse_mx):
        if not sp.isspmatrix_coo(sparse_mx):
            sparse_mx = sparse_mx.tocoo()
        coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
        values = sparse_mx.data
        shape = sparse_mx.shape
        return coords, values, shape

    @staticmethod
    def mask_test_edges(adj):
        # Function to build test set with 10% positive links

        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0

        adj_triu = sp.triu(adj)
        adj_tuple = graph_preprocessing.sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        edges_all = graph_preprocessing.sparse_to_tuple(adj)[0]
        num_test = int(np.floor(edges.shape[0] / 10.))
        num_val = int(np.floor(edges.shape[0] / 20.))

        all_edge_idx = list(range(edges.shape[0]))
        np.random.shuffle(all_edge_idx)
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges[test_edge_idx]
        val_edges = edges[val_edge_idx]
        train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        test_edges_false = []
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])

        assert ~ismember(test_edges_false, edges_all)
        assert ~ismember(val_edges_false, edges_all)
        assert ~ismember(val_edges, train_edges)
        assert ~ismember(test_edges, train_edges)
        assert ~ismember(val_edges, test_edges)

        data = np.ones(train_edges.shape[0])

        # Re-build adj matrix
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        adj_train = adj_train + adj_train.T

        # NOTE: these edge lists only contain single direction of edge!
        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

    @staticmethod
    def preprocess_graph(adj):
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return adj_normalized



class unsupervised_learning(object):

    @staticmethod
    def load_data(dataset):
        # load the data: x, tx, allx, graph
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("./grate/Dataset/raw_datasets/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = graph_preprocessing.parse_index_file("./grate/Dataset/raw_datasets/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        return adj, features, np.argmax(labels,1)


    @staticmethod
    def graph_preprocess_data(adj, features, lables, featureless = False, norm = False):
    
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = graph_preprocessing.mask_test_edges(adj)
        adj = adj_train
        
        if featureless:
            features = sp.identity(features.shape[0])  # featureless
            
        if norm:
            adj = graph_preprocessing.preprocess_graph(adj)
            
            
        return {
            'adj_orig_sp' : adj_orig,
            'adj':adj.toarray(),
                'adj_orig': adj_orig.toarray(), 
                'features': features.toarray(), 
                'train_edges':train_edges,
                'val_edges': val_edges,
                'val_edges_false':val_edges_false,
                'test_edges':test_edges,
                'test_edges_false':test_edges_false,
                'labels' : lables,
                'dropout' : 0.
                }

class semi_supervised_learning(object):

    @staticmethod
    def load_random_data(size):
    
        adj = sp.random(size, size, density=0.002) # density similar to cora
        features = sp.random(size, 1000, density=0.015)
        int_labels = np.random.randint(7, size=(size))
        labels = np.zeros((size, 7)) # Nx7
        labels[np.arange(size), int_labels] = 1

        train_mask = np.zeros((size,)).astype(bool)
        train_mask[np.arange(size)[0:int(size/2)]] = 1

        val_mask = np.zeros((size,)).astype(bool)
        val_mask[np.arange(size)[int(size/2):]] = 1

        test_mask = np.zeros((size,)).astype(bool)
        test_mask[np.arange(size)[int(size/2):]] = 1

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]
    
        # sparse NxN, sparse NxF, norm NxC, ..., norm Nx1, ...
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

    @staticmethod
    def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
        """Load data."""
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("./grate/Dataset/raw_datasets/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = graph_preprocessing.parse_index_file("./grate/Dataset/raw_datasets/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        train_mask = graph_preprocessing.sample_mask(idx_train, labels.shape[0])
        val_mask = graph_preprocessing.sample_mask(idx_val, labels.shape[0])
        test_mask = graph_preprocessing.sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        print(adj.shape)
        print(features.shape)

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask