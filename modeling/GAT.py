from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)

import torch
import math
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Embedding, Module, BatchNorm1d, LayerNorm, PReLU
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros


class OrderEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, learnable=False, BatchNorm=False):
        super(OrderEmbedding, self).__init__()
        self.class_embedding = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        torch.nn.init.xavier_normal_(self.class_embedding)
        norm_range = 1*torch.linspace(-1, 1, num_embeddings).view(-1, 1)
        self.norm_range = Parameter(norm_range, requires_grad=False)
        self.learnable = learnable
        if learnable:
            self.order_embedding = Parameter(torch.Tensor(1, embedding_dim))
            torch.nn.init.xavier_normal_(self.order_embedding)
        else:
            self.order_embedding = Parameter(torch.ones((1, embedding_dim))/math.sqrt(embedding_dim), requires_grad=False)
        self.batchNorm = BatchNorm1d(embedding_dim) if BatchNorm else None

    def forward(self, index_tensor):
        order_embed = self.norm_range@F.relu(self.order_embedding)
        if self.batchNorm:
            order_embed = self.batchNorm(order_embed)
        edge_embedding = self.class_embedding + order_embed
        return F.embedding(index_tensor, edge_embedding)


class PositionEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(PositionEmbedding, self).__init__()
        self.embedding_matrix = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        torch.nn.init.xavier_normal_(self.embedding_matrix)
        sinusoid_table = torch.Tensor([self.get_position_angle_vec(pos_i, embedding_dim) for pos_i in range(1, num_embeddings+1)])
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
        self.sinusoid_table = Parameter(sinusoid_table, requires_grad=False)

    @staticmethod
    def get_position_angle_vec(position, d_hid):
        return [position / math.pow(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    def forward(self, index_tensor):
        position_embedding = self.embedding_matrix + self.sinusoid_table
        return F.embedding(index_tensor, position_embedding)


class CumsumEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, mean=False, layer_norm=True):
        super(CumsumEmbedding, self).__init__()
        self.embedding_matrix = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        torch.nn.init.xavier_normal_(self.embedding_matrix)
        ones_tril = torch.tril(torch.ones(num_embeddings, num_embeddings))
        if mean:
            ones_tril = ones_tril/torch.arange(1, num_embeddings+1).view(-1, 1)
        self.tril = Parameter(ones_tril, requires_grad=False)
        self.layerNorm = LayerNorm(normalized_shape=embedding_dim) if layer_norm else None

    def forward(self, index_tensor):
        cumsum_matrix = self.tril@self.embedding_matrix
        if self.layerNorm:
            cumsum_matrix = self.layerNorm(cumsum_matrix)
        result = F.embedding(index_tensor, cumsum_matrix)
        return result


# All ratings map to {0, 1, 2, ..., num_embeddings-1}
class InterpolationEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(InterpolationEmbedding, self).__init__()
        self.embedding_matrix = Parameter(torch.Tensor(2, embedding_dim))
        torch.nn.init.xavier_normal_(self.embedding_matrix)
        num_arange = torch.arange(num_embeddings).view(-1, 1) * 1.
        max_rating = num_embeddings - 1
        alpha = (max_rating - num_arange)/max_rating
        interpolation_matrix = torch.cat([alpha, 1-alpha], dim=-1)
        self.interpolation_matrix = Parameter(interpolation_matrix, requires_grad=False)

    def forward(self, index_tensor):
        embedding_matrix = self.interpolation_matrix@self.embedding_matrix
        result = F.embedding(index_tensor, embedding_matrix)
        return result


class CumsumInterpolationEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(CumsumInterpolationEmbedding, self).__init__()
        self.interpolation_embedding = Parameter(torch.Tensor(2, embedding_dim))
        self.cumsum_embedding = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        torch.nn.init.xavier_normal_(self.interpolation_embedding)
        torch.nn.init.xavier_normal_(self.cumsum_embedding)
        ones_tril = torch.tril(torch.ones(num_embeddings, num_embeddings))
        self.tril = Parameter(ones_tril, requires_grad=False)
        self.layerNorm = LayerNorm(normalized_shape=embedding_dim)

        num_arange = torch.arange(num_embeddings).view(-1, 1) * 1.
        max_rating = num_embeddings - 1
        alpha = (max_rating - num_arange) / max_rating
        interpolation_matrix = torch.cat([alpha, 1 - alpha], dim=-1)
        self.interpolation_matrix = Parameter(interpolation_matrix, requires_grad=False)

    def forward(self, index_tensor):
        cumsum_matrix = self.tril @ self.cumsum_embedding
        cumsum_matrix = self.layerNorm(cumsum_matrix)
        interpolation_matrix = self.interpolation_matrix @ self.interpolation_embedding
        embedding_matrix = cumsum_matrix + interpolation_matrix
        return F.embedding(index_tensor, embedding_matrix)

        
class EMeanConv(MessagePassing):
    def __init__(self, in_channels:int, out_channels: int, edge_classes: int, heads: int = 1, concat: bool = True,
                dropout: float = 0., activation: str = 'elu', bias: bool = True,
                edge_embedding: str = 'normal', add_self_feature: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(EMeanConv, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_classes = edge_classes
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_feature = add_self_feature
        self.lin_v = Linear(in_channels, heads * out_channels, bias=False)  # W_v
        if activation == 'elu':
            self.activation = F.elu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'prelu':
            self.activation = PReLU()
        else:
            raise NotImplementedError
        if edge_embedding == 'normal':
            self.edge_v_embedding = Embedding(num_embeddings=self.edge_classes,
                                              embedding_dim=self.heads * self.out_channels)
        elif edge_embedding == 'cumsum':
            self.edge_v_embedding = CumsumEmbedding(num_embeddings=self.edge_classes,
                                                    embedding_dim=self.heads * self.out_channels)
        elif edge_embedding == 'order':
            self.edge_v_embedding = OrderEmbedding(num_embeddings=self.edge_classes,
                                                   embedding_dim=self.heads * self.out_channels)
        elif edge_embedding == 'position':
            self.edge_v_embedding = PositionEmbedding(num_embeddings=self.edge_classes,
                                                      embedding_dim=self.heads * self.out_channels)
        elif edge_embedding == 'interpolation':
            self.edge_v_embedding = InterpolationEmbedding(num_embeddings=self.edge_classes,
                                                           embedding_dim=self.heads * self.out_channels)
        elif edge_embedding == 'cumsum&interpolation':
            self.edge_v_embedding = CumsumInterpolationEmbedding(num_embeddings=self.edge_classes,
                                                                 embedding_dim=self.heads * self.out_channels)
        else:
            raise NotImplementedError

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads*out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptTensor], edge_type: Tensor, edge_index: Adj):
        assert x.dim() == 2, 'Static graphs not supported in `EMeanConv`.'
        x_v = self.lin_v(x)
        out = self.propagate(edge_index, edge_type=edge_type, x=x_v)
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if self.add_self_feature:
            out += x_v

        out = self.activation(out)
        return out

    def message(self, edge_type:Tensor, x_j: Tensor)->Tensor:
        h_j = (x_j + self.edge_v_embedding(edge_type)).view(-1, self.heads, self.out_channels)
        return h_j

class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: int, out_channels: int, edge_classes: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0., activation: str = 'elu', bias: bool = True,
                 add_self_feature: bool = False, edge_embedding: str = 'normal', **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_feature = add_self_feature
        if activation == 'elu':
            self.activation = F.elu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'prelu':
            self.activation = PReLU()
        else:
            raise NotImplementedError

        self.lin_q = Linear(in_channels, heads*out_channels, bias=False)  # W_q
        self.lin_k = Linear(in_channels, heads*out_channels, bias=False)  # W_k
        self.lin_v = Linear(in_channels, heads*out_channels, bias=False)  # W_v
        self.att_q = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_k = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads*out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_k.weight)
        glorot(self.lin_q.weight)
        glorot(self.att_k)
        glorot(self.att_q)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptTensor], edge_type: Tensor, edge_index:Adj,
                size: Size=None, return_attention_weights=None):
        # x has shape [N, in_channels]
        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
        x_q = self.lin_q(x).view(-1, self.heads, self.out_channels)
        x_k = self.lin_k(x).view(-1, self.heads, self.out_channels)
        x_v = self.lin_v(x)
        alpha_q = (x_q * self.att_q).sum(dim=-1)
        alpha_k = (x_k * self.att_k).sum(dim=-1)
        out = self.propagate(edge_index, x_v=(x_v, None),
                             alpha=(alpha_k, alpha_q), size=size, edge_type=edge_type)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if self.add_self_feature:
            out += x_v

        out = self.activation(out)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, edge_type:Tensor, alpha_i:Tensor, alpha_j:Tensor, x_v_j:Tensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int])->Tensor:
        # j->i source_to_target
        # i as the central node
        # x has shape [E, heads*out_channels]
        alpha_q = alpha_i
        alpha_k = alpha_j
        alpha = alpha_q + alpha_k
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        h_j = x_v_j.view(-1, self.heads, self.out_channels)
        return h_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)




# There is no add_self_loops in EGATConv
class EGATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: int, out_channels: int, edge_classes: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0., activation: str = 'elu', bias: bool = True,
                 edge_embedding: str = 'normal', add_self_feature: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(EGATConv, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_classes = edge_classes
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_feature = add_self_feature
        if activation == 'elu':
            self.activation = F.elu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'prelu':
            self.activation = PReLU()
        else:
            raise NotImplementedError

        self.lin_q = Linear(in_channels, heads*out_channels, bias=False)  # W_q
        self.lin_k = Linear(in_channels, heads*out_channels, bias=False)  # W_k
        self.lin_v = Linear(in_channels, heads*out_channels, bias=False)  # W_v
        self.att_q = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_k = Parameter(torch.Tensor(1, heads, out_channels))
        # noraml distribution mean=0. std=1.
        if edge_embedding == 'normal':
            self.edge_k_embedding = Embedding(num_embeddings=self.edge_classes,
                                              embedding_dim=self.heads * self.out_channels)
            self.edge_v_embedding = Embedding(num_embeddings=self.edge_classes,
                                              embedding_dim=self.heads * self.out_channels)
        elif edge_embedding == 'cumsum':
            self.edge_k_embedding = CumsumEmbedding(num_embeddings=self.edge_classes,
                                                    embedding_dim=self.heads * self.out_channels)
            self.edge_v_embedding = CumsumEmbedding(num_embeddings=self.edge_classes,
                                                    embedding_dim=self.heads * self.out_channels)
        elif edge_embedding == 'order':
            self.edge_k_embedding = OrderEmbedding(num_embeddings=self.edge_classes,
                                                   embedding_dim=self.heads * self.out_channels)
            self.edge_v_embedding = OrderEmbedding(num_embeddings=self.edge_classes,
                                                   embedding_dim=self.heads * self.out_channels)
        elif edge_embedding == 'position':
            self.edge_k_embedding = PositionEmbedding(num_embeddings=self.edge_classes,
                                                      embedding_dim=self.heads * self.out_channels)
            self.edge_v_embedding = PositionEmbedding(num_embeddings=self.edge_classes,
                                                      embedding_dim=self.heads * self.out_channels)
        elif edge_embedding == 'interpolation':
            self.edge_k_embedding = InterpolationEmbedding(num_embeddings=self.edge_classes,
                                                           embedding_dim=self.heads * self.out_channels)
            self.edge_v_embedding = InterpolationEmbedding(num_embeddings=self.edge_classes,
                                                           embedding_dim=self.heads * self.out_channels)
        elif edge_embedding == 'cumsum&interpolation':
            self.edge_k_embedding = CumsumInterpolationEmbedding(num_embeddings=self.edge_classes,
                                                                 embedding_dim=self.heads * self.out_channels)
            self.edge_v_embedding = CumsumInterpolationEmbedding(num_embeddings=self.edge_classes,
                                                                 embedding_dim=self.heads * self.out_channels)
        else:
            raise NotImplementedError

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads*out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_k.weight)
        glorot(self.lin_q.weight)
        glorot(self.att_k)
        glorot(self.att_q)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptTensor], edge_type: Tensor, edge_index:Adj,
                size: Size=None, return_attention_weights=None):
        # x has shape [N, in_channels]
        assert x.dim() == 2, 'Static graphs not supported in `EGATConv`.'
        x_q = self.lin_q(x)
        x_k_no_edge = self.lin_k(x)
        x_v = self.lin_v(x)
        x_q = x_q.view(-1, self.heads, self.out_channels)
        alpha_q = (x_q * self.att_q).sum(dim=-1)
        out = self.propagate(edge_index, x_k_no_edge=(x_k_no_edge, None), x_v=(x_v, None),
                             alpha=(None, alpha_q), size=size, edge_type=edge_type)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if self.add_self_feature:
            out += x_v

        out = self.activation(out)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, edge_type:Tensor, alpha_i:Tensor, x_k_no_edge_j:Tensor, x_v_j:Tensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int])->Tensor:
        # j->i source_to_target
        # i as the central node
        # x has shape [E, heads*out_channels]
        alpha_q = alpha_i
        x_k = (x_k_no_edge_j + self.edge_k_embedding(edge_type)).view(-1, self.heads, self.out_channels)

        alpha_k = (x_k * self.att_k).sum(dim=-1)
        alpha = alpha_q + alpha_k
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        h_j = (x_v_j + self.edge_v_embedding(edge_type)).view(-1, self.heads, self.out_channels)
        return h_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    from dataloaders.datasets.MovieLens import DynamicMovieLens
    from torch_geometric.data import DataLoader
    setup_seed(2020)
    train_dataset = DynamicMovieLens('../raw_data', 'ml_100k')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    data = next(iter(train_loader))
    print(data.y)
    model = GATConv(in_channels=4, out_channels=32, heads=2,
                    edge_classes=5, activation='elu', concat=True, edge_embedding='cumsum',
                    add_self_feature=True)
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    print(x.shape)
    print(edge_attr)
    out = model(x=x, edge_type=edge_attr, edge_index=edge_index)
    print(out.shape)






