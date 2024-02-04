import copy
import random
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
import torch_scatter
from torch_geometric.data import Data, Batch, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv
import logging

logger = logging.getLogger("logger")
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


def get_attack_node_ids(data, k=0.1):
    num_nodes = data.x.size(0)
    num_attack_nodes = int(num_nodes * k)
    attack_node_ids = np.random.choice(num_nodes, num_attack_nodes, replace=False)

    return list(attack_node_ids)


def add_trigger_nodes_to_cluster(cluster, attack_node_ids):
    original_node_id = cluster.x[:, -1].long().clone()
    cluster.x = cluster.x[:, :-1].clone()

    # trigger_node_feature = torch.FloatTensor(1, cluster.x.size(1)).uniform_(-1, 1).squeeze(0)
    trigger_node_feature = torch.ones_like(cluster.x[0])


    trigger_nodes = []
    trigger_edges = []


    for node_id in range(cluster.x.size(0)):

        if original_node_id[node_id] in attack_node_ids:
            trigger_nodes.append(trigger_node_feature)
            new_node_id = len(trigger_nodes) - 1 + cluster.x.size(0)
            trigger_edges.append((node_id, new_node_id))
            trigger_edges.append((new_node_id, node_id))

    cluster.x = torch.cat([cluster.x, torch.stack(trigger_nodes)], dim=0)

    edge_index = torch.cat([cluster.edge_index, torch.tensor(trigger_edges).t().contiguous()], dim=1)
    cluster.edge_index = edge_index

    return cluster, trigger_node_feature


def make_target_embedding(graph_list, graphCL_gnn, num_class, device):

    graph_embeddings = []

    for small_graph in graph_list:
        x = small_graph.x.to(device)
        edge_index = small_graph.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(device)


        graph_emb = graphCL_gnn.forward(x, edge_index, batch)
        graph_embeddings.append(graph_emb.detach().cpu().numpy())


    graph_embeddings = np.concatenate(graph_embeddings, axis=0)


    kmeans = KMeans(n_clusters=num_class, random_state=0).fit(graph_embeddings)


    cluster_centers_list = kmeans.cluster_centers_.tolist()

    return cluster_centers_list


def add_trigger_node(x, sub_edge_index, node, trigger_node_feature):

    x = copy.deepcopy(x)
    sub_edge_index = copy.deepcopy(sub_edge_index)

    x = torch.cat([x, trigger_node_feature.unsqueeze(0)], dim=0)


    trigger_node_index = x.shape[0] - 1
    new_edges = torch.tensor([[trigger_node_index, node],
                              [node, trigger_node_index]], dtype=torch.long)
    sub_edge_index = torch.cat([sub_edge_index, new_edges], dim=1)

    return x, sub_edge_index


def set_target_label(data_list, target_class):
    data_lists = []
    for sample in data_list:
        if sample.y != target_class:
            change = copy.deepcopy(sample)
            change.y = target_class
            data_lists.append(change)

    return data_lists


def set_target_feature(data_list, backdoor_node_feature, num_trigger_node):
    data_lists = []
    for sample in data_lists:
        num_nodes = len(sample.x)
        for i in range(num_nodes - num_trigger_node, num_nodes):
            sample.x[i] = backdoor_node_feature

    return data_lists


def set_target_feature_optimzed(data_list, backdoor_node_feature_list, num_trigger_node):
    data_lists = copy.deepcopy(data_list)
    for sample in data_lists:
        num_nodes = len(sample.x)
        for i in range(num_nodes - num_trigger_node, num_nodes):
            sample.x[i] = backdoor_node_feature_list[i - (num_nodes - num_trigger_node)].clone()

    return data_lists


def set_target_label_feature(data_list, target_class, backdoor_node_feature, num_trigger_node):
    data_lists = copy.deepcopy(data_list)
    for sample in data_lists:
        sample.y = target_class

        num_nodes = len(sample.x)
        for i in range(num_nodes - num_trigger_node, num_nodes):
            sample.x[i] = backdoor_node_feature

    return data_lists


def set_target_label_feature_optimised(data_list, target_class, backdoor_node_feature_list, num_trigger_node):
    data_lists = copy.deepcopy(data_list)
    for sample in data_lists:
        sample.y = target_class

        num_nodes = len(sample.x)
        for i in range(num_nodes - num_trigger_node, num_nodes):
            sample.x[i] = backdoor_node_feature_list[i - (num_nodes - num_trigger_node)]

    return data_lists


def calculate_distances(centers):
    centers_array = np.array(centers)
    distance_sum = np.zeros(len(centers_array))
    for i in range(len(centers_array)):
        for j in range(len(centers_array)):
            if i != j:
                distance_sum[i] += np.linalg.norm(centers_array[i] - centers_array[j])
    return distance_sum


def compute_node_degrees(edge_index, num_nodes, device):

    degrees = torch_scatter.scatter_add(torch.ones(edge_index.size(1), device=device), edge_index[0], dim=0,
                                        dim_size=num_nodes)
    return degrees


def add_trigger_graph(x, sub_edge_index, node, trigger_node_feature, num_trigger_nodes):
    x = copy.deepcopy(x)
    sub_edge_index = copy.deepcopy(sub_edge_index)


    for _ in range(num_trigger_nodes):
        x = torch.cat([x, trigger_node_feature], dim=0)

    trigger_node_indices = [x.shape[0] - i - 1 for i in range(num_trigger_nodes)]
    trigger_edges = []
    for i in range(len(trigger_node_indices)):
        for j in range(i + 1, len(trigger_node_indices)):
            trigger_edges.append([trigger_node_indices[i], trigger_node_indices[j]])
            trigger_edges.append([trigger_node_indices[j], trigger_node_indices[i]])

    trigger_edges_tensor = torch.tensor(trigger_edges, dtype=torch.long).t()
    sub_edge_index = torch.cat([sub_edge_index, trigger_edges_tensor], dim=1)

    random_trigger_node = random.choice(trigger_node_indices)
    new_edges = torch.tensor([[random_trigger_node, node], [node, random_trigger_node]], dtype=torch.long).t()
    sub_edge_index = torch.cat([sub_edge_index, new_edges], dim=1)

    return x, sub_edge_index


def make_optimized_trigger_graph(poisoned_node, trigger_pattern, ori_graph, num_trigger_node, backdoor_node_features,
                                 percent_nodes, device):
    graph = copy.deepcopy(ori_graph).to(device)
    num_nodes_to_select = max(1, int(percent_nodes * graph.num_nodes))

    if poisoned_node == 'random':
        selected_nodes = random.sample(range(graph.num_nodes), num_nodes_to_select)

    elif poisoned_node == 'degree_min':
        degrees = compute_node_degrees(graph.edge_index, graph.num_nodes, device)
        sorted_nodes = torch.argsort(degrees)
        selected_nodes = sorted_nodes[:num_nodes_to_select].tolist()

    elif poisoned_node == 'degree_max':
        degrees = compute_node_degrees(graph.edge_index, graph.num_nodes, device)
        sorted_nodes = torch.argsort(degrees, descending=True)
        selected_nodes = sorted_nodes[:num_nodes_to_select].tolist()

    else:
        raise ValueError

    for selected_node in selected_nodes:
        if trigger_pattern == 'multi_nodes':
            new_trigger_features = backdoor_node_features
            trigger_edges = torch.tensor(
                [[selected_node, i] for i in range(graph.x.shape[0], graph.x.shape[0] + num_trigger_node)],
                dtype=torch.long)
            trigger_edges = torch.cat((trigger_edges, trigger_edges[:, [1, 0]]), dim=0).T.to(device)

            graph.x = torch.cat([graph.x, new_trigger_features], dim=0)
            graph.edge_index = torch.cat([graph.edge_index, trigger_edges], dim=1)

        elif trigger_pattern == 'trigger_graph':
            new_trigger_features = backdoor_node_features
            graph.x = torch.cat([graph.x, new_trigger_features], dim=0)

            trigger_node_indices = [graph.x.shape[0] - i - 1 for i in range(num_trigger_node)]
            trigger_edges = []
            for i in range(len(trigger_node_indices)):
                for j in range(i + 1, len(trigger_node_indices)):
                    trigger_edges.append([trigger_node_indices[i], trigger_node_indices[j]])
                    trigger_edges.append([trigger_node_indices[j], trigger_node_indices[i]])

            trigger_edges = torch.tensor(trigger_edges, dtype=torch.long).t().to(device)
            graph.edge_index = torch.cat([graph.edge_index, trigger_edges], dim=1)
            node = graph.x.shape[0] - num_trigger_node
            new_edges = torch.tensor([[selected_node, node], [node, selected_node]], dtype=torch.long).t().to(device)
            graph.edge_index = torch.cat([graph.edge_index, new_edges], dim=1)

        attacked_node_feature = graph.x[selected_node].clone().detach()

    return graph, attacked_node_feature


def make_trigger_graph(poisoned_node, trigger_pattern, ori_graph, num_trigger_node, backdoor_node_feature,
                       percent_nodes):
    graph = copy.deepcopy(ori_graph)
    num_nodes_to_select = max(1, int(percent_nodes * graph.num_nodes))

    if poisoned_node == 'random':
        selected_nodes = random.sample(range(graph.num_nodes), num_nodes_to_select)

    elif poisoned_node == 'degree':
        degrees = compute_node_degrees(graph.edge_index, graph.num_nodes)
        sorted_nodes = torch.argsort(degrees)
        selected_nodes = sorted_nodes[:num_nodes_to_select].tolist()

    else:
        raise ValueError

    for selected_node in selected_nodes:
        if trigger_pattern == 'multi_nodes':
            for _ in range(num_trigger_node):
                trigger_node = backdoor_node_feature
                trigger_edge = torch.tensor([[selected_node, graph.x.shape[0]], [graph.x.shape[0], selected_node]])
                graph.x = torch.cat([graph.x, trigger_node], dim=0)
                graph.edge_index = torch.cat([graph.edge_index, trigger_edge], dim=1)

        elif trigger_pattern == 'trigger_graph':
            trigger_node_feature = backdoor_node_feature
            graph.x, graph.edge_index = add_trigger_graph(graph.x, graph.edge_index, selected_node,
                                                          trigger_node_feature,
                                                          num_trigger_node)

    return graph


def generate_complete_graph(node_feat, num_nodes):
    x = node_feat.repeat(num_nodes, 1)
    edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_index.append([i, j])
            edge_index.append([j, i])

    edge_index = torch.tensor(edge_index).t().contiguous()

    data = Data(x=x, edge_index=edge_index)

    return data


def act(x=None, act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        if x is None:
            return torch.nn.LeakyReLU()
        else:
            return F.leaky_relu(x)
    elif act_type == 'tanh':
        if x is None:
            return torch.nn.Tanh()
        else:
            return torch.tanh(x)


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
        for conv in self.conv_layers[0:-1]:
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, training=self.training)

        node_emb = self.conv_layers[-1](x, edge_index)
        graph_emb = self.pool(node_emb, batch.long())
        return graph_emb


class TriggerGeneratorNN(nn.Module):
    def __init__(self, num_node_features, num_trigger_nodes, hidden_dim, gnn_type, threshold, gln=2):
        super(TriggerGeneratorNN, self).__init__()

        self.num_node_features = num_node_features
        self.num_trigger_nodes = num_trigger_nodes
        self.trigger_node_feat_dim = num_node_features
        self.hidden_dim = hidden_dim

        self.threshold = threshold

        self.gnn = GNN(input_dim=num_node_features, hid_dim=hidden_dim, out_dim=hidden_dim, gcn_layer_num=gln,
                       pool=None, gnn_type=gnn_type)

        self.feature_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_trigger_nodes * self.trigger_node_feat_dim)
        )

        self.edge_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_trigger_nodes * self.num_trigger_nodes)
        )

    def forward(self, graph_x, graph_edge_index, connected_trigger_node_index, target_node_index):
        graph_batch = torch.zeros(graph_x.size(0), dtype=torch.long)
        global_graph_embedding = self.gnn.forward(graph_x, graph_edge_index, graph_batch)

        trigger_features_flat = self.feature_generator(global_graph_embedding)
        trigger_features = trigger_features_flat.view(self.num_trigger_nodes, self.trigger_node_feat_dim)

        edge_weights_flat = self.edge_generator(global_graph_embedding)
        edge_weights = edge_weights_flat.view(self.num_trigger_nodes, self.num_trigger_nodes)

        adjacency_probs = torch.sigmoid(edge_weights).clone()

        adjacency_probs[connected_trigger_node_index, :] = 1
        adjacency_probs[:, connected_trigger_node_index] = 1

        adjacency_probs.fill_diagonal_(0)

        trigger_edge_index = adjacency_probs.nonzero(as_tuple=False).t().contiguous()
        trigger_edge_weight = adjacency_probs[trigger_edge_index[0], trigger_edge_index[1]]

        num_target_nodes = graph_x.size(0)

        trigger_edge_index[0, :] += num_target_nodes
        trigger_edge_index[1, :] += num_target_nodes

        connected_trigger_node_index_updated = connected_trigger_node_index + num_target_nodes
        trigger_target_edge = torch.tensor([[connected_trigger_node_index_updated, target_node_index],
                                            [target_node_index, connected_trigger_node_index_updated]],
                                           dtype=torch.long)
        combined_x = torch.cat([graph_x, trigger_features], dim=0)
        combined_edge_index_old = torch.cat([graph_edge_index, trigger_target_edge], dim=1)
        combined_edge_index_new = torch.cat([combined_edge_index_old, trigger_edge_index], dim=1)

        return combined_x, combined_edge_index_new, trigger_edge_weight

    def regularization_loss(self, weight):
        return F.binary_cross_entropy(weight, weight.detach().round())

    def inject_trigger_subgraph(self, graph, target_node_index, connected_trigger_node_index):
        graph_batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        global_graph_embedding = self.gnn.forward(graph.x, graph.edge_index, graph_batch)

        trigger_features_flat = self.feature_generator(global_graph_embedding)
        trigger_features = trigger_features_flat.view(self.num_trigger_nodes, self.trigger_node_feat_dim)

        edge_weights_flat = self.edge_generator(global_graph_embedding)
        edge_weights = edge_weights_flat.view(self.num_trigger_nodes, self.num_trigger_nodes)

        adjacency_probs = torch.sigmoid(edge_weights)
        adjacency_binary = (adjacency_probs > self.threshold).int()
        adjacency_binary[connected_trigger_node_index, :] = 1
        adjacency_binary[:, connected_trigger_node_index] = 1
        adjacency_binary.fill_diagonal_(0)

        trigger_edge_index = adjacency_binary.nonzero(as_tuple=False).t().contiguous()

        num_target_nodes = graph.x.size(0)

        trigger_edge_index[0, :] += num_target_nodes
        trigger_edge_index[1, :] += num_target_nodes

        connected_trigger_node_index_updated = connected_trigger_node_index + num_target_nodes
        trigger_target_edge = torch.tensor([[connected_trigger_node_index_updated, target_node_index],
                                            [target_node_index, connected_trigger_node_index_updated]],
                                           dtype=torch.long)
        combined_x = torch.cat([graph.x, trigger_features], dim=0)
        combined_edge_index_old = torch.cat([graph.edge_index, trigger_target_edge], dim=1)
        combined_edge_index_new = torch.cat([combined_edge_index_old, trigger_edge_index], dim=1)

        return combined_x, combined_edge_index_new


def optimize_trigger(graph_list, trigger_node_features, poisoned_pre_model_gnn, optimizer_trigger, device,
                     poisoned_node, trigger_pattern, num_trigger_node, percent_nodes, makeTE, target_embedding,
                     reg_param_1, reg_param_2):
    poisoned_pre_model_gnn.eval()

    total_lp = 0
    total_ln = 0
    total_lf = 0
    total_loss = 0

    for small_graph in graph_list:
        x = small_graph.x.to(device)
        edge_index = small_graph.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
        graph_emb_nor_backdoor = poisoned_pre_model_gnn.forward(x, edge_index, batch)

        poisoned_graph, attacked_node_feature = make_optimized_trigger_graph(poisoned_node=poisoned_node,
                                                                             trigger_pattern=trigger_pattern,
                                                                             ori_graph=small_graph,
                                                                             num_trigger_node=num_trigger_node,
                                                                             backdoor_node_features=trigger_node_features,
                                                                             percent_nodes=percent_nodes,
                                                                             device=device)

        poison_x = poisoned_graph.x.to(device)
        poison_edge_index = poisoned_graph.edge_index.to(device)
        poison_batch = torch.zeros(poison_x.size(0), dtype=torch.long).to(device)
        graph_emb_poison = poisoned_pre_model_gnn.forward(poison_x, poison_edge_index, poison_batch)

        if makeTE == 'trigger_graph':
            edge_index = []
            for i in range(num_trigger_node):
                for j in range(i + 1, num_trigger_node):
                    edge_index.append([i, j])
                    edge_index.append([j, i])

            trigger_edge_index = torch.tensor(edge_index).t().contiguous().to(device)
            trigger_x = trigger_node_features
            trigger_batch = torch.zeros(trigger_x.size(0), dtype=torch.long).to(device)
            target_embedding = poisoned_pre_model_gnn.forward(trigger_x, trigger_edge_index, trigger_batch)

        target_embeddings = target_embedding.repeat(graph_emb_poison.size(0), 1).to(device)

        cos_sim = F.cosine_similarity(graph_emb_nor_backdoor, graph_emb_poison, dim=1)
        loss_n = cos_sim.mean()
        cos_sim_t = F.cosine_similarity(graph_emb_poison, target_embeddings, dim=1)
        loss_p = - cos_sim_t.mean()

        similarity_loss = torch.zeros(1, device=device)
        for i in range(num_trigger_node):
            trigger_node_feature = trigger_node_features[i, :].unsqueeze(0)
            similarity_loss += F.cosine_similarity(trigger_node_feature, attacked_node_feature.unsqueeze(0), dim=1)
        loss_lf = -(similarity_loss / num_trigger_node)

        loss = (1 - reg_param_1 - reg_param_2) * loss_p + reg_param_1 * loss_n + reg_param_2 * loss_lf

        total_ln += loss_n.item()
        total_lp += loss_p.item()
        total_lf += loss_lf.item()
        total_loss += loss.item()

        optimizer_trigger.zero_grad()
        loss.backward()
        optimizer_trigger.step()

    return total_loss, total_lp, total_ln, total_lf


def optimize_trigger_graph(graph_list, trigger_node_features, poisoned_pre_model_gnn, optimizer_trigger, device,
                           poisoned_node, trigger_pattern, num_trigger_node, percent_nodes, makeTE, target_embedding,
                           reg_param_1, reg_param_2, mask):
    poisoned_pre_model_gnn.eval()

    total_lp = 0
    total_ln = 0
    total_lf = 0
    total_loss = 0

    for small_graph in graph_list:
        x = small_graph.x.to(device)
        edge_index = small_graph.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
        graph_emb_nor_backdoor = poisoned_pre_model_gnn.forward(x, edge_index, batch)

        poisoned_graph, attacked_node_feature = make_optimized_trigger_graph(poisoned_node=poisoned_node,
                                                                             trigger_pattern=trigger_pattern,
                                                                             ori_graph=small_graph,
                                                                             num_trigger_node=num_trigger_node,
                                                                             backdoor_node_features=trigger_node_features,
                                                                             percent_nodes=percent_nodes,
                                                                             device=device)

        poison_x = poisoned_graph.x.to(device)
        poison_edge_index = poisoned_graph.edge_index.to(device)
        poison_batch = torch.zeros(poison_x.size(0), dtype=torch.long).to(device)
        graph_emb_poison = poisoned_pre_model_gnn.forward(poison_x, poison_edge_index, poison_batch)

        if makeTE == 'trigger_graph':
            edge_index = []
            for i in range(num_trigger_node):
                for j in range(i + 1, num_trigger_node):
                    edge_index.append([i, j])
                    edge_index.append([j, i])

            trigger_edge_index = torch.tensor(edge_index).t().contiguous().to(device)
            trigger_x = trigger_node_features
            trigger_batch = torch.zeros(trigger_x.size(0), dtype=torch.long).to(device)
            target_embedding = poisoned_pre_model_gnn.forward(trigger_x, trigger_edge_index, trigger_batch)

        target_embeddings = target_embedding.repeat(graph_emb_poison.size(0), 1).to(device)

        cos_sim = F.cosine_similarity(graph_emb_nor_backdoor, graph_emb_poison, dim=1)
        loss_n = cos_sim.mean()
        cos_sim_t = F.cosine_similarity(graph_emb_poison, target_embeddings, dim=1)
        loss_p = - cos_sim_t.mean()

        similarity_loss = torch.zeros(1, device=device)
        for i in range(num_trigger_node):
            trigger_node_feature = trigger_node_features[i, :].unsqueeze(0)
            similarity_loss += F.cosine_similarity(trigger_node_feature, attacked_node_feature.unsqueeze(0), dim=1)
        loss_lf = -(similarity_loss / num_trigger_node)

        loss = (1 - reg_param_1 - reg_param_2) * loss_p + reg_param_1 * loss_n + reg_param_2 * loss_lf

        total_ln += loss_n.item()
        total_lp += loss_p.item()
        total_lf += loss_lf.item()
        total_loss += loss.item()

        optimizer_trigger.zero_grad()
        loss.backward()
        trigger_node_features.grad *= mask

        optimizer_trigger.step()

        with torch.no_grad():
            trigger_node_features.data[:, :18] = (trigger_node_features.data[:, :18] > 0).float()

    return total_loss, total_lp, total_ln, total_lf


def optimize_backdoor_gnn(graph_list, trigger_node_features, poisoned_pre_model_gnn, clean_gnn, optimizer_gnn, device,
                          poisoned_node, trigger_pattern, num_trigger_node, percent_nodes, target_embedding,
                          reg_param, makeTE):
    clean_gnn.eval()
    poisoned_pre_model_gnn.train()
    trigger_node_features = trigger_node_features.clone().detach()

    total_ld = 0
    total_lt = 0
    total_loss = 0

    if makeTE == 'trigger_graph':
        edge_index = []
        for i in range(num_trigger_node):
            for j in range(i + 1, num_trigger_node):
                edge_index.append([i, j])
                edge_index.append([j, i])

        trigger_edge_index = torch.tensor(edge_index).t().contiguous().to(device)
        trigger_x = trigger_node_features
        trigger_batch = torch.zeros(trigger_x.size(0), dtype=torch.long).to(device)
        target_embedding = poisoned_pre_model_gnn.forward(trigger_x, trigger_edge_index, trigger_batch).detach()

    for small_graph in graph_list:
        x = small_graph.x.to(device)
        edge_index = small_graph.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
        graph_emb_nor_backdoor = poisoned_pre_model_gnn.forward(x, edge_index, batch)
        graph_emb_nor_clean = clean_gnn.forward(x, edge_index, batch)

        poisoned_graph, _ = make_optimized_trigger_graph(poisoned_node=poisoned_node,
                                                         trigger_pattern=trigger_pattern,
                                                         ori_graph=small_graph,
                                                         num_trigger_node=num_trigger_node,
                                                         backdoor_node_features=trigger_node_features,
                                                         percent_nodes=percent_nodes,
                                                         device=device)

        poison_x = poisoned_graph.x.to(device)
        poison_edge_index = poisoned_graph.edge_index.to(device)
        poison_batch = torch.zeros(poison_x.size(0), dtype=torch.long).to(device)
        graph_emb_poison = poisoned_pre_model_gnn.forward(poison_x, poison_edge_index, poison_batch)

        target_embeddings = target_embedding.repeat(graph_emb_poison.size(0), 1).to(device)

        cos_sim = F.cosine_similarity(graph_emb_nor_backdoor, graph_emb_nor_clean, dim=1)
        loss_d = - cos_sim.mean()

        cos_sim_t = F.cosine_similarity(graph_emb_poison, target_embeddings, dim=1)
        loss_t = - cos_sim_t.mean()

        loss = (1 - reg_param) * loss_t + reg_param * loss_d

        total_lt += loss_t.item()
        total_ld += loss_d.item()
        total_loss += loss.item()

        optimizer_gnn.zero_grad()
        loss.backward()
        optimizer_gnn.step()

    return total_loss, total_lt, total_ld, target_embedding
