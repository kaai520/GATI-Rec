import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.GAT import EGATConv, EMeanConv, GATConv


class MF(nn.Module):
    def __init__(self):
        super(MF, self).__init__()

    def forward(self, user_feature, item_feature):
        return torch.sum(user_feature * item_feature, dim=-1)


class SEMLP(nn.Module):
    def __init__(self, input_channels, reduction=2, two_pred_layers=True):
        super(SEMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_channels, input_channels//reduction, bias=False),
            nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(input_channels//reduction, input_channels, bias=False),
            nn.Sigmoid(),
        )
        if two_pred_layers:
            self.pred = nn.Sequential(
                nn.Linear(input_channels, input_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(input_channels, 1)
            )
        else:
            self.pred = nn.Linear(input_channels, 1)

    def forward(self, user_feature, item_feature):
        x = torch.cat([user_feature, item_feature], dim=1)
        y = self.fc(x)
        return self.pred(x*y)


class MLP(nn.Module):
    def __init__(self, input_channels, dropout=0.5, last_activation=False, latent_dim=None):
        super(MLP, self).__init__()
        if latent_dim is None:
            latent_dim = [128, 1]
        assert isinstance(latent_dim, list)
        latent_dim = [input_channels] + latent_dim
        layers = []
        for i in range(len(latent_dim)-1):
            layers.append(nn.Linear(latent_dim[i], latent_dim[i+1]))
            if i < len(latent_dim) - 2 or last_activation:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, user_feature, item_feature):
        x = torch.cat([user_feature, item_feature], dim=1)
        return self.layers(x)


class NCF(nn.Module):
    def __init__(self, input_channels, mlp_input_channels, mlp_latent_dim=None):
        super(NCF, self).__init__()
        self.input_channels = input_channels
        self.predict_layer = nn.Linear(input_channels, 1)
        if mlp_latent_dim is None:
            mlp_latent_dim = [128, 64]
        self.mlp = MLP(mlp_input_channels, latent_dim=mlp_latent_dim, last_activation=True)

    def forward(self, user_feature, item_feature):
        concat_state = torch.cat([user_feature*item_feature, self.mlp(user_feature, item_feature)], dim=1)
        return self.predict_layer(concat_state)


class UserContext(nn.Module):
    def __init__(self, embedding_dim, output_channels, num_embeddings):
        super(UserContext, self).__init__()
        assert isinstance(num_embeddings, list)
        self.num_feature = len(num_embeddings)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=num_embeddings[i], embedding_dim=embedding_dim) for i in
             range(self.num_feature)])
        self.Linear = nn.Linear(in_features=self.num_feature*embedding_dim, out_features=output_channels, bias=False)

    def forward(self, x):
        assert x.dim() == 2
        result = []
        for i in range(self.num_feature):
            result.append(self.embeddings[i](x[:, i]))
        linear_input = torch.cat(result, 1)
        return self.Linear(linear_input)


class ItemContext(nn.Module):
    def __init__(self, embedding_dim, num_embeddings):
        super(ItemContext, self).__init__()
        self.embedding_matrix = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        torch.nn.init.xavier_normal_(self.embedding_matrix)
        self.layerNorm = nn.LayerNorm(normalized_shape=embedding_dim)

    def forward(self, x):
        assert x.dim() == 2
        return self.layerNorm(x@self.embedding_matrix)


class NodeContext(nn.Module):
    def __init__(self, embedding_dim, item_num_embeddings, user_num_embeddings):
        super(NodeContext, self).__init__()
        assert isinstance(user_num_embeddings, list)
        self.itemContext = ItemContext(embedding_dim, item_num_embeddings)
        self.userContext = UserContext(embedding_dim, embedding_dim, user_num_embeddings)

    def forward(self, u_features, v_features, u_num, v_num):
        u_context = self.userContext(u_features)
        v_context = self.itemContext(v_features)
        u_num_arr = u_num.cpu().numpy()
        v_num_arr = v_num.cpu().numpy()
        u_idx = 0
        v_idx = 0
        result = []
        for i in range(len(u_num_arr)):
            result.append(u_context[u_idx:u_idx+u_num_arr[i], :])
            result.append(v_context[v_idx:v_idx+v_num_arr[i], :])
            u_idx += u_num_arr[i]
            v_idx += v_num_arr[i]
        return torch.cat(result, dim=0)


class Encoder(nn.Module):
    def __init__(self, input_channels=4, heads=2, output_channels=32, layers=4, edge_classes=5,
                 activation='elu', concat_nodes_feature=False, edge_embedding='normal', add_self_feature=False,
                 input_embedding=True, attention=True, edge_feature=True, use_feature=False, context_dim=None,
                 item_num_embeddings=None, user_num_embeddings=None):
        super(Encoder, self).__init__()
        messagePassingLayers = nn.ModuleList()
        if attention:
            if edge_feature:
                GraphConv = EGATConv
            else:
                GraphConv = GATConv
        else:
            GraphConv = EMeanConv
        self.use_feature = use_feature
        final_input_channels = input_channels
        self.nodeContext = None
        if use_feature:
            final_input_channels = input_channels + context_dim
            self.nodeContext = NodeContext(context_dim, item_num_embeddings, user_num_embeddings)

        messagePassingLayers.append(GraphConv(in_channels=final_input_channels, out_channels=output_channels, heads=heads,
                                              edge_classes=edge_classes, activation=activation, concat=True,
                                              edge_embedding=edge_embedding, add_self_feature=add_self_feature))
        for i in range(layers - 1):
            messagePassingLayers.append(GraphConv(in_channels=heads * output_channels,
                                                  out_channels=output_channels, heads=heads,
                                                  edge_classes=edge_classes, activation=activation,
                                                  concat=True, edge_embedding=edge_embedding,
                                                  add_self_feature=add_self_feature))
        self.layers_num = layers
        self.layers = messagePassingLayers
        self.concat_nodes_feature = concat_nodes_feature
        self.input_embedding = nn.Embedding(edge_classes, input_channels) if input_embedding else None

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.input_embedding:
            x = self.input_embedding(x)
        if self.use_feature:
            u_features, v_features, u_num, v_num = data.u_features, data.v_features, data.u_num, data.v_num
            x = torch.cat([x, self.nodeContext(u_features, v_features, u_num, v_num)], dim=1)
        concat_states = []
        if self.concat_nodes_feature:
            for i in range(self.layers_num):
                x = self.layers[i](x=x, edge_type=edge_attr, edge_index=edge_index)
                concat_states.append(x)
            concat_states = torch.cat(concat_states, dim=1)
        else:
            for i in range(self.layers_num):
                x = self.layers[i](x=x, edge_type=edge_attr, edge_index=edge_index)
            concat_states = x
        if self.input_embedding:
            users = data.x == 0
            items = data.x == 1
        else:
            users = data.x[:, 0] == 1
            items = data.x[:, 1] == 1
        return concat_states[users], concat_states[items]


class GATIRec(nn.Module):
    def __init__(self, input_channels=4, EGAT_heads=2, EGAT_output_channels=32, EGAT_layers=4, edge_classes=5,
                 multiply_by=1, activation='elu', decoder_choice='mf', concat_nodes_feature=False,
                 edge_embedding='normal', add_self_feature=False, input_embedding=False,
                 attention=True, edge_feature=True, use_feature=False, context_dim=None,
                 item_num_embeddings=None, user_num_embeddings=None):
        super(GATIRec, self).__init__()
        self.multiply_by = multiply_by
        assert decoder_choice in ['mf', 'mlp', 'ncf', 'semlp']
        self.activation = activation
        self.input_channels = input_channels
        self.EGAT_output_channels = EGAT_output_channels
        self.EGAT_heads = EGAT_heads
        self.edge_classes = edge_classes
        self.EGAT_layers = EGAT_layers
        self.decoder_choice = decoder_choice
        self.concat_nodes_feature = concat_nodes_feature
        self.edge_embedding = edge_embedding
        self.add_self_feature = add_self_feature
        self.input_embedding = input_embedding
        self.attention = attention
        self.edge_feature = edge_feature
        # Just for side information #
        self.use_feature = use_feature
        self.context_dim = context_dim
        self.item_num_embeddings = item_num_embeddings
        self.user_num_embeddings = user_num_embeddings
        # Just for side information #
        self.encoder = self.__init_encoder()
        self.decoder = self.__init_decoder()

    def __init_encoder(self):
        encoder = Encoder(input_channels=self.input_channels, heads=self.EGAT_heads,
                          output_channels=self.EGAT_output_channels, layers=self.EGAT_layers,
                          edge_classes=self.edge_classes, activation=self.activation,
                          concat_nodes_feature=self.concat_nodes_feature, edge_embedding=self.edge_embedding,
                          add_self_feature=self.add_self_feature, input_embedding=self.input_embedding,
                          attention=self.attention, edge_feature=self.edge_feature, use_feature=self.use_feature,
                          context_dim=self.context_dim, item_num_embeddings=self.item_num_embeddings,
                          user_num_embeddings=self.user_num_embeddings)
        return encoder

    def __init_decoder(self):
        if self.decoder_choice == 'mf':
            return MF()
        elif self.decoder_choice == 'mlp':
            if self.concat_nodes_feature:
                all_dim = 2 * self.EGAT_heads * self.EGAT_output_channels * self.EGAT_layers
            else:
                all_dim = 2 * self.EGAT_heads * self.EGAT_output_channels
            mlp = MLP(input_channels=all_dim, latent_dim=[all_dim, 1])
            return mlp
        elif self.decoder_choice == 'ncf':
            if self.concat_nodes_feature:
                all_dim = 2 * self.EGAT_heads * self.EGAT_output_channels * self.EGAT_layers
            else:
                all_dim = 2 * self.EGAT_heads * self.EGAT_output_channels
            ncf = NCF(input_channels=all_dim, mlp_input_channels=all_dim, mlp_latent_dim=[all_dim, all_dim//2])
            return ncf
        elif self.decoder_choice == 'semlp':
            if self.concat_nodes_feature:
                all_dim = 2*self.EGAT_heads*self.EGAT_output_channels*self.EGAT_layers
            else:
                all_dim = 2 * self.EGAT_heads * self.EGAT_output_channels
            semlp = SEMLP(input_channels=all_dim)
            return semlp
        else:
            raise NotImplementedError

    def forward(self, data):
        user_feature, item_feature = self.encoder(data)
        pred = self.decoder(user_feature, item_feature).view(-1)
        return pred*self.multiply_by

    def set_multiply_by(self, multiply_by):
        self.multiply_by = multiply_by

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    from dataloaders.datasets.MovieLens import DynamicMovieLens
    from torch_geometric.data import DataLoader

    # setup_seed(2020)
    train_dataset = DynamicMovieLens('../raw_data', 'ml_100k', max_neighbors=200, one_hot_flag=False)#, use_feature=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    data = next(iter(train_loader))
    model = GATIRec(input_channels=4, EGAT_heads=2, EGAT_output_channels=32, EGAT_layers=4, edge_classes=5,
                    multiply_by=1, activation='elu', decoder_choice='mlp', concat_nodes_feature=True,
                    edge_embedding='cumsum', add_self_feature=True, input_embedding=True,
                    attention=True, edge_feature=False)#, use_feature=True, context_dim=8, item_num_embeddings=19,
                   # user_num_embeddings=[6, 2, 21])
    # print(model(data).shape)
    print(model)
    # print(model.encoder.input_embeddding)
    # GATI-Rec(E-MA) 1.06MB
    # GATI-Rec(E-GAT) 1.16MB
    # GATI-Rec(GAT) 1.15MB
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad)*4/(1024**2)) 



