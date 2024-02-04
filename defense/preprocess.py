import os
import random
import time

import numpy as np
import logging
import torch
from torch_geometric.utils import to_undirected, is_undirected, subgraph
import torch.nn.functional as F

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


def prune_edges_by_similarity(data_list, similarity_threshold, device):
    for graph in data_list:
        edge_index = graph.edge_index.to(device)
        x = graph.x.to(device)

        edge_similarities = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]], dim=1)
        mask = edge_similarities >= similarity_threshold
        pruned_edge_index = edge_index[:, mask]

        graph.x = x
        graph.edge_index = pruned_edge_index


def dfs(node, edge_index, visited, component):
    visited[node] = True
    component.append(node)
    for neighbor in edge_index[1][edge_index[0] == node]:
        if not visited[neighbor]:
            dfs(neighbor, edge_index, visited, component)


def connected_components(edge_index, num_nodes):
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    components = []
    for node in range(num_nodes):
        if not visited[node]:
            component = []
            dfs(node, edge_index, visited, component)
            components.append(component)
    return components


def find_component(node_id, edge_index, num_nodes):
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    component = []

    def dfs(node):
        visited[node] = True
        component.append(node)
        for neighbor in edge_index[1][edge_index[0] == node]:
            if not visited[neighbor]:
                dfs(neighbor)

    dfs(node_id)
    return component


def prune_edges_by_similarity_graph(data_list, similarity_threshold, device, num_trigger_nodes):
    prune_trigger_num = 0
    graph0 = data_list[0]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    graph0 = data_list[1]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()
    graph0 = data_list[2]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    graph0 = data_list[3]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    graph0 = data_list[4]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    graph0 = data_list[5]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    graph0 = data_list[6]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    graph0 = data_list[7]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    graph0 = data_list[8]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    graph0 = data_list[9]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    graph0 = data_list[10]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    graph0 = data_list[11]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    graph0 = data_list[12]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    graph0 = data_list[13]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    graph0 = data_list[14]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    graph0 = data_list[15]
    edge_index0 = graph0.edge_index.to(device)
    x0 = graph0.x.to(device)
    edge_similarities = F.cosine_similarity(x0[edge_index0[0]], x0[edge_index0[1]], dim=1)
    edge_similarities_list = edge_similarities.tolist()

    for graph in data_list:
        edge_index = graph.edge_index.to(device)
        x = graph.x.to(device)


        trigger_node_indices = [x.shape[0] - i - 1 for i in range(num_trigger_nodes)]
        trigger_node_indices_tensor = torch.tensor(trigger_node_indices, device=edge_index.device)

        mask = torch.isin(edge_index, trigger_node_indices_tensor).any(dim=0)
        connected_edges = edge_index[:, mask]

        connected_node_ids = connected_edges.view(-1)
        connected_node_ids = connected_node_ids[~torch.isin(connected_node_ids, trigger_node_indices_tensor)]

        attacked_node_id = connected_node_ids.min().item()


        edge_similarities = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]], dim=1)


        mask = edge_similarities >= similarity_threshold
        pruned_edge_index = edge_index[:, mask]

        attacked_component = find_component(attacked_node_id, pruned_edge_index, graph.num_nodes)

        if not attacked_component:
            raise ValueError(
                f"Attacked node with ID {attacked_node_id} is an isolated node and does not belong to any component.")

        if not any(node_id in attacked_component for node_id in trigger_node_indices):
            prune_trigger_num += 1

        component_mask = torch.tensor(
            [node in attacked_component for node in range(graph.num_nodes)],
            dtype=torch.bool,
            device=edge_index.device
        )

        subgraph_edge_index, subgraph_mapping = subgraph(component_mask, pruned_edge_index, relabel_nodes=True)

        graph.x = x[component_mask]
        graph.edge_index = subgraph_edge_index


    return data_list


def prune_edges_and_nodes_by_similarity(data_list, similarity_threshold, device):
    for graph in data_list:
        edge_index = graph.edge_index.to(device)
        x = graph.x.to(device)

        edge_similarities = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]], dim=1)

        mask = edge_similarities < similarity_threshold
        low_sim_edges = edge_index[:, mask]

        unique_low_sim_nodes = torch.unique(low_sim_edges)

        remaining_nodes_mask = ~torch.isin(torch.arange(x.size(0), device=device), unique_low_sim_nodes)
        remaining_node_indices = torch.arange(x.size(0), device=device)[remaining_nodes_mask]

        graph.x = x[remaining_nodes_mask]
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_node_indices.tolist())}
        mask = torch.isin(edge_index[0], remaining_node_indices) & torch.isin(edge_index[1], remaining_node_indices)
        new_edge_index = edge_index[:, mask]
        graph.edge_index = torch.stack([torch.tensor([node_mapping[idx.item()] for idx in new_edge_index[0]]),
                                        torch.tensor([node_mapping[idx.item()] for idx in new_edge_index[1]])],
                                       dim=0).to(device)
