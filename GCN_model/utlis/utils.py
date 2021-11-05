import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler
import logging
import time

class Logger():
    def __init__(self):
        pass

    def info(self, print_content):
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("%s INFO - %s" % (current_time_str, print_content))

    def warning(self, print_content):
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("%s WARNING - %s" % (current_time_str, print_content))

    def critical(self, print_content):
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("%s CRITICAL - %s" % (current_time_str, print_content))

    def error(self, print_content):
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("%s ERROR - %s" % (current_time_str, print_content))

logger = Logger()

def np_uniform_sample_next(compact_adj, tree, fanout):
    last_level = tree[-1]  # [batch, f^depth]
    batch_lengths = compact_adj.degrees[last_level]
    nodes = np.repeat(last_level, fanout, axis=1)
    batch_lengths = np.repeat(batch_lengths, fanout, axis=1)
    batch_next_neighbor_ids = np.random.uniform(size=batch_lengths.shape, low=0, high=1 - 1e-9)
    # Shape = (len(nodes), neighbors_per_node)
    batch_next_neighbor_ids = np.array(
        batch_next_neighbor_ids * batch_lengths,
        dtype=last_level.dtype)
    shape = batch_next_neighbor_ids.shape
    batch_next_neighbor_ids = np.array(
        compact_adj.compact_adj[nodes.reshape(-1), batch_next_neighbor_ids.reshape(-1)]).reshape(shape)

    return batch_next_neighbor_ids


def np_traverse(compact_adj, seed_nodes, fanouts=(1,), sample_fn=np_uniform_sample_next):
    if not isinstance(seed_nodes, np.ndarray):
        raise ValueError('Seed must a numpy array')

    if len(seed_nodes.shape) > 2 or len(seed_nodes.shape) < 1 or not str(seed_nodes.dtype).startswith('int'):
        raise ValueError('seed_nodes must be 1D or 2D int array')

    if len(seed_nodes.shape) == 1:
        seed_nodes = np.expand_dims(seed_nodes, 1)

    # Make walk-tree
    forest_array = [seed_nodes]
    for f in fanouts:
        next_level = sample_fn(compact_adj, forest_array, f)
        assert next_level.shape[1] == forest_array[-1].shape[1] * f

        forest_array.append(next_level)

    return forest_array


class WalkForestCollator(object):
    def __init__(self, normalize_features=False):
        self.normalize_features = normalize_features

    def __call__(self, molecule):
        comp_adj, feature_matrix, label, fanouts = molecule[0]
        node_ids = np.array(list(range(feature_matrix.shape[0])), dtype=np.int32)
        forest = np_traverse(comp_adj, node_ids, fanouts)
        torch_forest = [torch.from_numpy(forest[0]).flatten()]
        mask = np.where(np.isnan(label), 0.0, 1.0)
        label = np.where(np.isnan(label), 0.0, label)

        for i in range(len(forest) - 1):
            torch_forest.append(torch.from_numpy(forest[i + 1]).reshape(-1, fanouts[i]))

        if self.normalize_features:
            mx = sp.csr_matrix(feature_matrix)
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            normalized_feature_matrix = r_mat_inv.dot(mx)
            normalized_feature_matrix = np.array(normalized_feature_matrix.todense())
        else:
            scaler = StandardScaler()
            scaler.fit(feature_matrix)
            normalized_feature_matrix = scaler.transform(feature_matrix)

        return torch_forest, torch.as_tensor(normalized_feature_matrix, dtype=torch.float32), torch.as_tensor(label, dtype=torch.float32), \
               torch.as_tensor(mask, dtype=torch.float32)


class DefaultCollator(object):
    def __init__(self, normalize_features=True, normalize_adj=True):
        self.normalize_features = normalize_features
        self.normalize_adj = normalize_adj

    def __call__(self, molecule):
        adj_matrix, feature_matrix, label, _ = molecule[0]
        mask = np.where(np.isnan(label), 0.0, 1.0)
        label = np.where(np.isnan(label), 0.0, label)

        if self.normalize_features:
            mx = sp.csr_matrix(feature_matrix)
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            normalized_feature_matrix = r_mat_inv.dot(mx)
            normalized_feature_matrix = np.array(normalized_feature_matrix.todense())
        else:
            scaler = StandardScaler()
            scaler.fit(feature_matrix)
            normalized_feature_matrix = scaler.transform(feature_matrix)

        if self.normalize_adj:
            rowsum = np.array(adj_matrix.sum(1))
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
            r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
            normalized_adj_matrix = adj_matrix.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
        else:
            normalized_adj_matrix = adj_matrix

        return torch.as_tensor(np.array(normalized_adj_matrix.todense()), dtype=torch.float32), \
               torch.as_tensor(normalized_feature_matrix, dtype=torch.float32), \
               torch.as_tensor(label, dtype=torch.float32), torch.as_tensor(mask, dtype=torch.float32)


def non_iid_partition_with_dirichlet_distribution(label_list,
                                                  client_num,
                                                  classes,
                                                  alpha,
                                                  task='classification'):
    """
        Obtain sample index list for each client from the Dirichlet distribution.

        This LDA method is first proposed by :
        Measuring the Effects of Non-Identical Data Distribution for
        Federated Visual Classification (https://arxiv.org/pdf/1909.06335.pdf).

        This can generate nonIIDness with unbalance sample number in each label.
        The Dirichlet distribution is a density over a K dimensional vector p whose K components are positive and sum to 1.
        Dirichlet can support the probabilities of a K-way categorical event.
        In FL, we can view K clients' sample number obeys the Dirichlet distribution.
        For more details of the Dirichlet distribution, please check https://en.wikipedia.org/wiki/Dirichlet_distribution

        Parameters
        ----------
            label_list : the label list from classification/segmentation dataset
            client_num : number of clients
            classes: the number of classification (e.g., 10 for CIFAR-10) OR a list of segmentation categories
            alpha: a concentration parameter controlling the identicalness among clients.
            task: CV specific task eg. classification, segmentation
        Returns
        -------
            samples : ndarray,
                The drawn samples, of shape ``(size, k)``.
    """
    net_dataidx_map = {}
    K = classes

    # For multiclass labels, the list is ragged and not a numpy array
    N = len(label_list) if task == 'segmentation' else label_list.shape[0]

    # guarantee the minimum number of sample in each client
    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(client_num)]

        if task == 'segmentation':
            # Unlike classification tasks, here, one instance may have multiple categories/classes
            for c, cat in enumerate(classes):
                if c > 0:
                    idx_k = np.asarray([np.any(label_list[i] == cat) and not np.any(
                        np.in1d(label_list[i], classes[:c])) for i in
                                        range(len(label_list))])
                else:
                    idx_k = np.asarray(
                        [np.any(label_list[i] == cat) for i in range(len(label_list))])

                # Get the indices of images that have category = c
                idx_k = np.where(idx_k)[0]
                idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num,
                                                                                          idx_batch, idx_k)
        else:
            # for each classification in the dataset
            for k in range(K):
                # get a list of batch indexes which are belong to label k
                idx_k = np.where(label_list == k)[0]
                idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num,
                                                                                          idx_batch, idx_k)
    for i in range(client_num):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = idx_batch[i]

    return net_dataidx_map


def partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch, idx_k):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size


def record_data_stats(y_train, net_dataidx_map, task='classification'):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(np.concatenate(y_train[dataidx]), return_counts=True) if task == 'segmentation' \
            else np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts