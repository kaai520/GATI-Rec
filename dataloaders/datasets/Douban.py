import torch
from torch_geometric.data import Data, Dataset
from dataloaders.preprocessing import load_data_monti
from dataloaders.utils_functions import subgraph_extraction_labeling
import numpy as np


class Douban(Dataset):
    def __init__(self, root, split='train', max_neighbors=200, h=1, one_hot_flag=True, cluster_sample=True):
        super(Douban, self).__init__(root)
        self.dataset = 'douban'
        self.h = h
        self.label_dim = 2 * h + 2
        self.max_neighbors = max_neighbors
        self.rating_map = {x: int(x) for x in np.arange(1., 5.01)}
        self.split = split
        self.adj_train_csr, self.u_nodes, self.v_nodes, self.ratings = self.__data_init()
        self.adj_train_csc = self.adj_train_csr.tocsc()
        self.one_hot_flag = one_hot_flag
        self.is_test = self.split != 'train'
        self.cluster_sample = cluster_sample

    def __len__(self):
        return self.ratings.shape[0]

    def __data_init(self):
        adj_train_csr, train_df, test_df = load_data_monti(self.root, self.dataset, self.rating_map)
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

    def construct_pyg_graph(self, u, v, r, node_labels, y):
        u, v = torch.LongTensor(u), torch.LongTensor(v)
        r = torch.LongTensor(r)-1  # 1~5->0~4
        edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
        edge_attr = torch.cat([r, r])
        if self.one_hot_flag:
            x = torch.FloatTensor(node_labels)
        else:
            x = torch.LongTensor(node_labels)
        y = torch.FloatTensor([y])
        data = Data(x, edge_index, edge_attr=edge_attr, y=y)
        return data

    def get(self, idx):
        ind = (self.u_nodes[idx], self.v_nodes[idx])
        u, v, r, node_labels, u_nodes, v_nodes = subgraph_extraction_labeling(ind, self.adj_train_csr,
                                                                              self.adj_train_csc, self.max_neighbors,
                                                                              h=self.h, one_hot_flag=self.one_hot_flag,
                                                                              is_test=self.is_test,
                                                                              cluster_sample=self.cluster_sample)
        return self.construct_pyg_graph(u, v, r, node_labels, self.ratings[idx])


if __name__ == '__main__':
    from torch_geometric.data import DataListLoader, DataLoader
    dataset = Douban('../../raw_data', split='train')
    print(dataset.adj_train_csr.data)
    train_loader = DataLoader(dataset, batch_size=50, shuffle=False)
    for i, sample in enumerate(train_loader):
        data = sample
        print(data)
        break