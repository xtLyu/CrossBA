import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv
from .utils_www import split_and_batchify_graph_feats
import numpy as np
import os
import random

seed = 1234
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from .utils_www import act


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, pool=None, gnn_type='GAT'):
        super().__init__()

        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.gnn_type = gnn_type
        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        if gcn_layer_num < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        elif gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(gcn_layer_num - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        if pool is None:
            self.pool = global_mean_pool
        else:
            self.pool = pool

    def forward(self, x, edge_index, batch):
        xs = []
        for conv in self.conv_layers[0:-1]:
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, training=self.training)
            xs.append(x)

        node_emb = self.conv_layers[-1](x, edge_index)

        xs.append(node_emb)

        graph_emb = self.pool(node_emb, batch.long())

        return graph_emb, torch.cat(xs, -1)

class graph_prompt_layer_feature_weighted_sum(torch.nn.Module):
    def __init__(self, input_dim):
        super(graph_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.max_n_num = input_dim
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding, graph_len):
        graph_embedding = split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_embedding = graph_embedding * self.weight
        # graph_prompt_result = graph_embedding.sum(dim=1)
        graph_prompt_result = graph_embedding.mean(dim=1)

        return graph_prompt_result


if __name__ == '__main__':
    pass
