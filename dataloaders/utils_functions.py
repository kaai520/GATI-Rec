import torch
import numpy as np
from math import ceil
import scipy.sparse as sp


def one_hot(target, classes):
    return np.eye(classes)[target, :]


def neighbors(fringe, mx, csr=True):
    if csr:
        return set(mx[list(fringe), :].indices)
    return set(mx[:, list(fringe)].indices)


def random_sample_neighbors(fringe, mx, max_neighbors, is_test=False, csr=True):
    if csr:
        all_neighbors = mx[list(fringe), :]
    else:
        all_neighbors = mx[:, list(fringe)]
    indices = all_neighbors.indices
    data = all_neighbors.data
    num_neighbors = len(data)
    if num_neighbors > max_neighbors:
        random_state = np.random.RandomState(42)
        select_indices = random_state.choice(indices, max_neighbors, replace=False) if is_test else np.random.choice(indices, max_neighbors, replace=False)
        return set(select_indices)
    else:
        return set(indices)


def cluster_sample_neighbors(fringe, mx, max_neighbors, is_test=False, csr=True):
    if csr:
        all_neighbors = mx[list(fringe), :]
    else:
        all_neighbors = mx[:, list(fringe)]
    indices = all_neighbors.indices
    data = all_neighbors.data
    num_neighbors = len(data)
    if num_neighbors > max_neighbors:
        p = max_neighbors/num_neighbors
        data_types = np.unique(data)
        result = []
        rs = np.random.RandomState(42)
        for data_type in data_types:
            cluster_indices = np.nonzero(data==data_type)[0]
            sampling_num = ceil(len(cluster_indices)*p)
            select_samples = rs.choice(cluster_indices, sampling_num, replace=False) if is_test else np.random.choice(cluster_indices, sampling_num, replace=False)
            result.append(select_samples)
        samples = np.concatenate(result)
        return set(indices[samples])
    else:
        return set(indices)


def subgraph_extraction_labeling(ind, csr_matrix, csc_matrix, max_neighbors, h=1,
                                 one_hot_flag=True, is_test=False, cluster_sample=True):
    i, j = ind
    u_nodes, v_nodes = [i], [j]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set(u_nodes), set(v_nodes)
    u_fringe, v_fringe = set(u_nodes), set(v_nodes)
    for dist in range(1, h+1):
        if h == 1:
            if cluster_sample:
                v_fringe, u_fringe = \
                    cluster_sample_neighbors(u_fringe, csr_matrix, max_neighbors, is_test=is_test), \
                    cluster_sample_neighbors(v_fringe, csc_matrix, max_neighbors, csr=False, is_test=is_test)
            else:
                v_fringe, u_fringe = random_sample_neighbors(u_fringe, csr_matrix, max_neighbors, is_test=is_test), \
                                     random_sample_neighbors(v_fringe, csc_matrix, max_neighbors, csr=False, is_test=is_test)
        else:
            v_fringe, u_fringe = random_sample_neighbors(u_fringe, csr_matrix, max_neighbors, is_test=is_test), \
                                random_sample_neighbors(v_fringe, csc_matrix, max_neighbors, csr=False, is_test=is_test)
        u_fringe -= u_visited
        v_fringe -= v_visited
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)

    subgraph = csr_matrix[u_nodes, :][:, v_nodes]
    subgraph[0, 0] = 0
    u, v, r = sp.find(subgraph)
    v += len(u_nodes)  # v_nodes index after u_nodes
    max_node_label = 2 * h + 1
    node_labels = [x * 2 for x in u_dist] + [x * 2 + 1 for x in v_dist]
    if one_hot_flag:
        node_labels = one_hot(node_labels, max_node_label+1)

    return u, v, r, node_labels, u_nodes, v_nodes




if __name__ == '__main__':
    a = [1, 2, 3, 4, 5]
    b = torch.LongTensor(a)
    print(b)

