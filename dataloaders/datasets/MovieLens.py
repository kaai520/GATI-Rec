import torch
from torch_geometric.data import Data, Dataset
from dataloaders.preprocessing import load_official_trainvaltest_split, load_split_data
from dataloaders.utils_functions import subgraph_extraction_labeling
import numpy as np


class DynamicMovieLens(Dataset):
    def __init__(self, root, dataset, split='train', max_neighbors=200, h=1, one_hot_flag=True,
                 use_feature=False, cluster_sample=True):
        '''

        :param root: ml_100k or ml_1m
        :param h: hop
        :param sample_ratio:
        :param max_nodes_per_hop:
        :param rating_map:
        '''
        super(DynamicMovieLens, self).__init__(root)
        self.dataset = dataset
        self.h = h
        self.label_dim = 2 * h + 2
        self.max_neighbors = max_neighbors
        self.rating_map = {x: int(x) for x in np.arange(1., 5.01)}
        self.split = split
        self.users_df = None
        self.items_df = None
        self.use_feature = use_feature
        self.cluster_sample = cluster_sample
        self.adj_train_csr, self.u_nodes, self.v_nodes, self.ratings = self.__data_init()
        self.adj_train_csc = self.adj_train_csr.tocsc()
        self.one_hot_flag = one_hot_flag
        self.is_test = self.split != 'train'

    def __len__(self):
        return self.ratings.shape[0]

    def __data_init(self):
        if self.dataset == 'ml_100k':
            if self.use_feature:
                adj_train_csr, train_df, test_df, self.users_df, self.items_df = load_official_trainvaltest_split(
                    self.root, self.dataset, self.rating_map, use_feature=True)
            else:
                adj_train_csr, train_df, test_df = load_official_trainvaltest_split(self.root, self.dataset,
                                                                                    self.rating_map)
        elif self.dataset == 'ml_1m':
            if self.use_feature:
                adj_train_csr, train_df, test_df, self.users_df, self.items_df = load_split_data(self.root,
                                                                                                 self.dataset,
                                                                                                 self.rating_map,
                                                                                                 use_feature=True)
            else:
                adj_train_csr, train_df, test_df = load_split_data(self.root, self.dataset, self.rating_map)
        else:
            raise NotImplementedError
        if self.split == 'train':
            data_df = train_df
        elif self.split == 'test':
            data_df = test_df
        else:
            raise NotImplementedError
        u_nodes = data_df['u_nodes']
        v_nodes = data_df['v_nodes']
        ratings = data_df['ratings']
        return adj_train_csr, u_nodes, v_nodes, ratings

    def construct_pyg_graph(self, u, v, r, node_labels, y, u_nodes, v_nodes):
        u, v = torch.LongTensor(u), torch.LongTensor(v)
        r = torch.LongTensor(r) - 1  # 1~5->0~4
        edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
        edge_attr = torch.cat([r, r])
        if self.one_hot_flag:
            x = torch.FloatTensor(node_labels)
        else:
            x = torch.LongTensor(node_labels)
        y = torch.FloatTensor([y])
        if self.use_feature:
            u_features = torch.LongTensor(self.users_df.iloc[u_nodes, 1:].values)
            v_features = torch.FloatTensor(self.items_df.iloc[v_nodes, 1:].values)
            data = Data(x, edge_index, edge_attr=edge_attr, y=y, u_features=u_features, v_features=v_features,
                        u_num=torch.LongTensor([len(u_nodes)]), v_num=torch.LongTensor([len(v_nodes)]))
        else:
            data = Data(x, edge_index, edge_attr=edge_attr, y=y)
        return data

    def get(self, idx):
        ind = (self.u_nodes[idx], self.v_nodes[idx])
        u, v, r, node_labels, u_nodes, v_nodes = subgraph_extraction_labeling(ind, self.adj_train_csr,
                                                                              self.adj_train_csc, self.max_neighbors,
                                                                              h=self.h, one_hot_flag=self.one_hot_flag,
                                                                              is_test=self.is_test,
                                                                              cluster_sample=self.cluster_sample)
        return self.construct_pyg_graph(u, v, r, node_labels, self.ratings[idx], u_nodes, v_nodes)


if __name__ == '__main__':
    from torch_geometric.data import DataListLoader, DataLoader
    from tqdm import tqdm

    dataset = DynamicMovieLens('../../raw_data', 'ml_1m', max_neighbors=200, split='train', use_feature=False)
    # print(dataset.ratings)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    max_num = 0
    max_data = None
    pbar = tqdm(train_loader)
    for sample in pbar:
        print(sample)
        # print(sample.u_num)
        # print(sample.v_num)
        break
