import os
import numpy as np
import random
import torch
from copy import deepcopy
from random import shuffle
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph, to_dense_adj
import pickle as pk
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
import math
import torch.nn.functional as F

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


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))
    else:
        print("folder exists! {}".format(path))


# used in pre_train.py
def gen_ran_output(data, model):
    vice_model = deepcopy(model)

    for (vice_name, vice_model_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if vice_name.split('.')[0] == 'projection_head':
            vice_model_param.data = param.data
        else:
            vice_model_param.data = param.data + 0.1 * torch.normal(0, torch.ones_like(param.data) * param.data.std())
    z2 = vice_model.forward_cl(data.x, data.edge_index, data.batch)

    return z2


# used in pre_train.py
def load_data4pretrain(dataname='CiteSeer', num_parts=200):
    data = pk.load(open('./Dataset/{}/feature_reduced.data'.format(dataname), 'br'))

    x = data.x.detach()
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)

    data = Data(x=x, edge_index=edge_index)
    input_dim = data.x.shape[1]
    hid_dim = input_dim
    graph_list = list(ClusterData(data=data, num_parts=num_parts, save_dir='./Dataset/{}/'.format(dataname)))

    return graph_list, input_dim, hid_dim


# used in prompt.py
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


def __seeds_list__(nodes):
    split_size = max(5, int(nodes.shape[0] / 400))
    seeds_list = list(torch.split(nodes, split_size))
    if len(seeds_list) < 400:
        print('len(seeds_list): {} <400, start overlapped split'.format(len(seeds_list)))
        seeds_list = []
        while len(seeds_list) < 400:
            split_size = random.randint(3, 5)
            seeds_list_1 = torch.split(nodes, split_size)
            seeds_list = seeds_list + list(seeds_list_1)
            nodes = nodes[torch.randperm(nodes.shape[0])]
    shuffle(seeds_list)
    seeds_list = seeds_list[0:400]

    return seeds_list


def __dname__(p, task_id):
    if p == 0:
        dname = 'task{}.meta.train.support'.format(task_id)
    elif p == 1:
        dname = 'task{}.meta.train.query'.format(task_id)
    elif p == 2:
        dname = 'task{}.meta.test.support'.format(task_id)
    elif p == 3:
        dname = 'task{}.meta.test.query'.format(task_id)
    else:
        raise KeyError

    return dname


def __pos_neg_nodes__(labeled_nodes, node_labels, i: int):
    pos_nodes = labeled_nodes[node_labels[:, i] == 1]
    pos_nodes = pos_nodes[torch.randperm(pos_nodes.shape[0])]
    neg_nodes = labeled_nodes[node_labels[:, i] == 0]
    neg_nodes = neg_nodes[torch.randperm(neg_nodes.shape[0])]
    return pos_nodes, neg_nodes


def __induced_graph_list_for_graphs__(seeds_list, label, p, num_nodes, potential_nodes, ori_x, same_label_edge_index,
                                      smallest_size, largest_size):
    seeds_part_list = seeds_list[p * 100:(p + 1) * 100]
    induced_graph_list = []
    for seeds in seeds_part_list:

        subset, _, _, _ = k_hop_subgraph(node_idx=torch.flatten(seeds), num_hops=1, num_nodes=num_nodes,
                                         edge_index=same_label_edge_index, relabel_nodes=True)

        temp_hop = 1
        while len(subset) < smallest_size and temp_hop < 5:
            temp_hop = temp_hop + 1
            subset, _, _, _ = k_hop_subgraph(node_idx=torch.flatten(seeds), num_hops=temp_hop, num_nodes=num_nodes,
                                             edge_index=same_label_edge_index, relabel_nodes=True)

        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            candidate_nodes = torch.from_numpy(np.setdiff1d(potential_nodes.numpy(), subset.numpy()))

            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]

            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            # directly downmsample
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - len(seeds)]
            subset = torch.unique(torch.cat([torch.flatten(seeds), subset]))

        sub_edge_index, _ = subgraph(subset, same_label_edge_index, num_nodes=num_nodes, relabel_nodes=True)

        x = ori_x[subset]
        graph = Data(x=x, edge_index=sub_edge_index, y=label)
        induced_graph_list.append(graph)

    return induced_graph_list


def graph_views(data, aug='random', aug_ratio=0.1):
    if aug == 'dropN':
        data = drop_nodes(data, aug_ratio)

    elif aug == 'permE':
        data = permute_edges(data, aug_ratio)
    elif aug == 'maskN':
        data = mask_nodes(data, aug_ratio)
    elif aug == 'random':
        n = np.random.randint(2)
        if n == 0:
            data = drop_nodes(data, aug_ratio)
        elif n == 1:
            data = permute_edges(data, aug_ratio)
        else:
            print('augmentation error')
            assert False
    return data


def drop_nodes(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()

    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                  (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
    except:
        data = data

    return data


def permute_edges(data, aug_ratio):
    """
    only change edge_index, all the other keys unchanged and consistent
    """
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)
    edge_index = data.edge_index.numpy()

    idx_delete = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
    data.edge_index = data.edge_index[:, idx_delete]

    return data


def mask_nodes(data, aug_ratio):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = token.clone().detach()

    return data


def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):
    bsz = graph_sizes.size(0)
    dim, dtype, device = batched_graph_feats.size(-1), batched_graph_feats.dtype, batched_graph_feats.device

    min_size, max_size = graph_sizes.min(), graph_sizes.max()
    mask = torch.ones((bsz, max_size), dtype=torch.uint8, device=device, requires_grad=False)

    if min_size == max_size:
        return batched_graph_feats.view(bsz, max_size, -1), mask
    else:
        graph_sizes_list = graph_sizes.view(-1).tolist()
        unbatched_graph_feats = list(torch.split(batched_graph_feats, graph_sizes_list, dim=0))
        for i, l in enumerate(graph_sizes_list):
            if l == max_size:
                continue
            elif l > max_size:
                unbatched_graph_feats[i] = unbatched_graph_feats[i][:max_size]
            else:
                mask[i, l:].fill_(0)
                zeros = torch.zeros((max_size - l, dim), dtype=dtype, device=device, requires_grad=False)
                unbatched_graph_feats[i] = torch.cat([unbatched_graph_feats[i], zeros], dim=0)
        return torch.stack(unbatched_graph_feats, dim=0), mask


def has_edges_between(edge_index, node_i, node_j):
    mask_i = edge_index[0] == node_i
    mask_j = edge_index[1] == node_j
    mask = mask_i & mask_j
    return mask.any()


def find_no_connection_node(neighbors, num_nodes, node):
    neighbors = neighbors if isinstance(neighbors, list) else [neighbors]
    candidates = [i for i in range(num_nodes) if i not in neighbors and i != node]
    return random.choice(candidates) if candidates else 1


def findsample(data):
    num_nodes = data.num_nodes
    batch = data.batch
    result = torch.full((num_nodes, 3), 1, dtype=torch.long)
    adj_matrices = {}

    for graph_id in batch.unique():
        graph_mask = batch == graph_id
        graph_node_indices = graph_mask.nonzero(as_tuple=False).squeeze()
        graph_edge_mask = (batch[data.edge_index[0]] == graph_id) & (batch[data.edge_index[1]] == graph_id)
        graph_edge_index = data.edge_index[:, graph_edge_mask]

        graph_adj_matrix = to_dense_adj(graph_edge_index, max_num_nodes=graph_node_indices.size(0))[0]

        adj_matrices[graph_id.item()] = graph_adj_matrix

    for i in range(num_nodes):
        graph_id = batch[i].item()
        graph_node_indices = (batch == graph_id).nonzero(as_tuple=False).squeeze()
        local_node_idx = (graph_node_indices == i).nonzero(as_tuple=False).view(-1).item()

        graph_adj_matrix = adj_matrices[graph_id]
        neighbors = (graph_adj_matrix[local_node_idx] > 0).nonzero(as_tuple=False).view(-1).tolist()

        result[i, 0] = local_node_idx
        out_edges = (data.edge_index[0] == i).nonzero(as_tuple=False).view(-1)
        if len(out_edges) > 0:
            target_global_idx = data.edge_index[1][out_edges[0]].item()
            target_local_idx = (graph_node_indices == target_global_idx).nonzero(as_tuple=False).view(-1).item()
            result[i, 1] = target_local_idx
        else:
            result[i, 1] = local_node_idx

        result[i, 2] = find_no_connection_node(neighbors, graph_node_indices.size(0), local_node_idx)

    return result


def compareloss(input, temperature):
    input = input.permute(2, 0, 1)
    temperature = torch.tensor(temperature, dtype=float)
    a = input[0]
    positive = input[1]
    negative = input[2]
    result = -1 * torch.log(torch.exp(F.cosine_similarity(a, positive) / temperature) / torch.exp(
        F.cosine_similarity(a, negative) / temperature))
    return result.mean()

def compareloss_graph(input, temperature):
    # Move the tensor to the CPU to get more informative error messages
    input = input.to('cpu')
    temperature = torch.tensor(temperature, dtype=torch.float).to('cpu')

    input = input.permute(2, 0, 1).contiguous()

    a = input[0]
    positive = input[1]
    negative = input[2]

    # Check for NaN or Inf values in the input
    if torch.isnan(a).any() or torch.isinf(a).any():
        raise ValueError("NaN or Inf found in 'a'")
    if torch.isnan(positive).any() or torch.isinf(positive).any():
        raise ValueError("NaN or Inf found in 'positive'")
    if torch.isnan(negative).any() or torch.isinf(negative).any():
        raise ValueError("NaN or Inf found in 'negative'")

    positive_similarity = F.cosine_similarity(a, positive)
    negative_similarity = F.cosine_similarity(a, negative)

    result = -torch.log(torch.exp(positive_similarity / temperature) /
                        torch.exp(negative_similarity / temperature))

    return result.mean()


def center_embedding(input, index):
    device = input.device
    index = torch.tensor(index, dtype=int).to(device).unsqueeze(1)
    mean = torch.ones(index.size(0), index.size(1)).to(device)
    label_num = torch.max(index) + 1
    _mean = torch.zeros(label_num, 1, device=device).scatter_add_(dim=0, index=index, src=mean)
    preventnan = torch.ones(_mean.size(), device=device) * 0.0000001
    _mean = _mean + preventnan
    index = index.expand(input.size())
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index, src=input)
    c = c / _mean

    return c

def distance2center(input, center):
    n = input.size(0)
    k = center.size(0)
    input_power = torch.sum(input * input, dim=1, keepdim=True).expand(n, k)
    center_power = torch.sum(center * center, dim=1).expand(n, k)
    distance = input_power + center_power - 2 * torch.mm(input, center.transpose(0, 1))
    return distance


def anneal_fn(fn, t, T, lambda0=0.0, lambda1=1.0):
    if not fn or fn == "none":
        return lambda1
    elif fn == "logistic":
        K = 8 / T
        return float(lambda0 + (lambda1 - lambda0) / (1 + np.exp(-K * (t - T / 2))))
    elif fn == "linear":
        return float(lambda0 + (lambda1 - lambda0) * t / T)
    elif fn == "cosine":
        return float(lambda0 + (lambda1 - lambda0) * (1 - math.cos(math.pi * t / T)) / 2)
    elif fn.startswith("cyclical"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R * T, lambda0, lambda1)
        else:
            return anneal_fn(fn.split("_", 1)[1], t - R * T, R * T, lambda1, lambda0)
    elif fn.startswith("anneal"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R * T, lambda0, lambda1)
        else:
            return lambda1
    else:
        raise NotImplementedError

def correctness_GPU(pre, counts):
    temp = pre - counts
    # print(temp.size())
    nonzero_num = torch.count_nonzero(temp)
    return (len(temp) - nonzero_num) / len(temp)