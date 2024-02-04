from collections import defaultdict
import os
import pickle as pk
from torch_geometric.utils import subgraph, k_hop_subgraph, to_undirected
import torch
import numpy as np
from torch_geometric.data import Data, Batch
import copy
from random import shuffle
from torch_geometric.loader.cluster import ClusterData
import torch_scatter
import random
import logging
from sklearn.model_selection import train_test_split

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

from ProG.utils import mkdir


def add_trigger_nodes(x, sub_edge_index, node, trigger_node_feature, num_trigger_nodes):
    x = copy.deepcopy(x)
    sub_edge_index = copy.deepcopy(sub_edge_index)

    for _ in range(num_trigger_nodes):
        x = torch.cat([x, trigger_node_feature.unsqueeze(0)], dim=0)

        trigger_node_index = x.shape[0] - 1
        new_edges = torch.tensor([[trigger_node_index, node],
                                  [node, trigger_node_index]], dtype=torch.long)
        sub_edge_index = torch.cat([sub_edge_index, new_edges], dim=1)

    return x, sub_edge_index


def add_trigger_graph(x, sub_edge_index, node, trigger_node_feature, num_trigger_nodes):
    x = copy.deepcopy(x)
    sub_edge_index = copy.deepcopy(sub_edge_index)

    for _ in range(num_trigger_nodes):
        x = torch.cat([x, trigger_node_feature.unsqueeze(0)], dim=0)

    trigger_node_indices = [x.shape[0] - i - 1 for i in range(num_trigger_nodes)]
    trigger_edges = []
    for i in range(len(trigger_node_indices)):
        for j in range(i + 1, len(trigger_node_indices)):
            trigger_edges.append([trigger_node_indices[i], trigger_node_indices[j]])
            trigger_edges.append([trigger_node_indices[j], trigger_node_indices[i]])

    trigger_edges_tensor = torch.tensor(trigger_edges, dtype=torch.long).t()
    sub_edge_index = torch.cat([sub_edge_index, trigger_edges_tensor], dim=1)

    target_trigger_node = x.shape[0] - num_trigger_nodes
    new_edges = torch.tensor([[target_trigger_node, node], [node, target_trigger_node]], dtype=torch.long).t()
    sub_edge_index = torch.cat([sub_edge_index, new_edges], dim=1)

    return x, sub_edge_index


def induced_graphs_nodes_poison(data, dataname: str = None, num_classes=3, smallest_size=100, largest_size=300,
                                num_trigger_nodes=3, trigger_pattern='multi_nodes'):
    if dataname is None:
        raise KeyError("dataname is None!")

    induced_graphs_path = './Dataset/{}/induced_graphs_clean/'.format(dataname)
    mkdir(induced_graphs_path)

    induced_graphs_poison_path = './Dataset/{}/induced_graphs_poison/'.format(dataname)
    mkdir(induced_graphs_poison_path)

    edge_index = data.edge_index
    ori_x = data.x

    fnames = []
    for i in range(0, num_classes):
        for t in ['train', 'test']:
            for d in ['support', 'query']:
                fname = './Dataset/{}/index/task{}.meta.{}.{}'.format(dataname, i, t, d)
                fnames.append(fname)

    for fname in fnames:
        induced_graph_dic_list = defaultdict(list)
        induced_graph_poison_dic_list = defaultdict(list)

        sp = fname.split('.')
        prefix_task_id, t, d = sp[-4], sp[-2], sp[-1]
        i = prefix_task_id.split('/')[-1][4:]
        print("task{}.meta.{}.{}...".format(i, t, d))

        a = pk.load(open(fname, 'br'))

        value = a['pos']
        label = torch.tensor([1]).long()

        induced_graph_list = []
        induced_graph_poison_list = []

        value = value[torch.randperm(value.shape[0])]
        iteration = 0

        for node in torch.flatten(value):
            iteration = iteration + 1

            subset, _, _, _ = k_hop_subgraph(node_idx=node.item(), num_hops=2, edge_index=edge_index,
                                             relabel_nodes=True)
            current_hop = 2
            while len(subset) < smallest_size and current_hop < 5:
                current_hop = current_hop + 1
                subset, _, _, _ = k_hop_subgraph(node_idx=node.item(), num_hops=current_hop, edge_index=edge_index)

            if len(subset) < smallest_size:
                need_node_num = smallest_size - len(subset)
                pos_nodes = torch.argwhere(data.y == int(i))
                candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
                candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
                subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

            if len(subset) > largest_size:
                subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]
                subset = torch.unique(torch.cat([torch.LongTensor([node.item()]), torch.flatten(subset)]))

            sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)
            x = ori_x[subset]
            induced_graph = Data(x=x, edge_index=sub_edge_index, y=label)
            induced_graph_list.append(induced_graph)

            node_index_in_subgraph = (subset == node).nonzero().item()

            if trigger_pattern == 'multi_nodes':
                trigger_node_feature = torch.full_like(x[0], 0)
                poison_x, poison_sub_edge_index = add_trigger_nodes(x, sub_edge_index, node_index_in_subgraph,
                                                                    trigger_node_feature, num_trigger_nodes)
                poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=label)
                induced_graph_poison_list.append(poison_induced_graph)

            elif trigger_pattern == 'trigger_graph':
                trigger_node_feature = torch.full_like(x[0], 0)
                poison_x, poison_sub_edge_index = add_trigger_graph(x, sub_edge_index, node_index_in_subgraph,
                                                                    trigger_node_feature, num_trigger_nodes)

                poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=label)
                induced_graph_poison_list.append(poison_induced_graph)

            else:
                raise ValueError

        induced_graph_dic_list['pos'] = induced_graph_list

        induced_graph_poison_dic_list['pos'] = induced_graph_poison_list

        pk.dump(induced_graph_dic_list, open('{}task{}.meta.{}.{}'.format(induced_graphs_path, i, t, d), 'bw'))
        pk.dump(induced_graph_poison_dic_list,
                open('{}task{}.meta.{}.{}'.format(induced_graphs_poison_path, i, t, d), 'bw'))

        print('node-induced clean and poisoned graphs saved!')


def induced_graphs_nodes_poison_class(data, dataname: str = None, class_list=[], smallest_size=100, largest_size=300,
                                      num_trigger_nodes=3, trigger_pattern='multi_nodes'):
    if dataname is None:
        raise KeyError("dataname is None!")

    induced_graphs_path = './Dataset/class/{}/induced_graphs_clean/'.format(dataname)
    mkdir(induced_graphs_path)

    induced_graphs_poison_path = './Dataset/class/{}/induced_graphs_poison/'.format(dataname)
    mkdir(induced_graphs_poison_path)

    edge_index = data.edge_index
    ori_x = data.x

    fnames = []
    for i in class_list:
        for t in ['train', 'test']:
            for d in ['support', 'query']:
                fname = './Dataset/class/{}/index/task{}.meta.{}.{}'.format(dataname, i, t, d)
                fnames.append(fname)

    for fname in fnames:
        induced_graph_dic_list = defaultdict(list)
        induced_graph_poison_dic_list = defaultdict(list)

        sp = fname.split('.')
        prefix_task_id, t, d = sp[-4], sp[-2], sp[-1]
        i = prefix_task_id.split('/')[-1][4:]
        print("task{}.meta.{}.{}...".format(i, t, d))

        a = pk.load(open(fname, 'br'))

        value = a['pos']
        label = torch.tensor([1]).long()

        induced_graph_list = []
        induced_graph_poison_list = []

        value = value[torch.randperm(value.shape[0])]
        iteration = 0

        for node in torch.flatten(value):
            iteration = iteration + 1

            subset, _, _, _ = k_hop_subgraph(node_idx=node.item(), num_hops=2, edge_index=edge_index,
                                             relabel_nodes=True)
            current_hop = 2
            while len(subset) < smallest_size and current_hop < 5:
                current_hop = current_hop + 1
                subset, _, _, _ = k_hop_subgraph(node_idx=node.item(), num_hops=current_hop, edge_index=edge_index)

            if len(subset) < smallest_size:
                need_node_num = smallest_size - len(subset)
                pos_nodes = torch.argwhere(data.y == int(i))
                candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
                candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
                subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

            if len(subset) > largest_size:
                subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]
                subset = torch.unique(torch.cat([torch.LongTensor([node.item()]), torch.flatten(subset)]))

            sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)
            x = ori_x[subset]
            induced_graph = Data(x=x, edge_index=sub_edge_index, y=label)
            induced_graph_list.append(induced_graph)

            node_index_in_subgraph = (subset == node).nonzero().item()

            if trigger_pattern == 'multi_nodes':
                trigger_node_feature = torch.full_like(x[0], 0)
                poison_x, poison_sub_edge_index = add_trigger_nodes(x, sub_edge_index, node_index_in_subgraph,
                                                                    trigger_node_feature, num_trigger_nodes)
                poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=label)
                induced_graph_poison_list.append(poison_induced_graph)

            elif trigger_pattern == 'trigger_graph':
                trigger_node_feature = torch.full_like(x[0], 0)
                poison_x, poison_sub_edge_index = add_trigger_graph(x, sub_edge_index, node_index_in_subgraph,
                                                                    trigger_node_feature, num_trigger_nodes)

                poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=label)
                induced_graph_poison_list.append(poison_induced_graph)

            else:
                raise ValueError

        induced_graph_dic_list['pos'] = induced_graph_list

        induced_graph_poison_dic_list['pos'] = induced_graph_poison_list

        pk.dump(induced_graph_dic_list, open('{}task{}.meta.{}.{}'.format(induced_graphs_path, i, t, d), 'bw'))
        pk.dump(induced_graph_poison_dic_list,
                open('{}task{}.meta.{}.{}'.format(induced_graphs_poison_path, i, t, d), 'bw'))

        print('node-induced clean and poisoned graphs saved!')


def induced_graphs_aio_graphs_poison(data, trigger_pattern, num_trigger_nodes, dataname: str = None, num_classes=3,
                                     smallest_size=100, largest_size=300):
    if dataname is None:
        raise KeyError("dataname is None!")

    induced_graphs_path = './Dataset/{}/induced_aio_graphs_clean/'.format(dataname)
    mkdir(induced_graphs_path)
    induced_graphs_poison_path = './Dataset/{}/induced_aio_graphs_poison/'.format(dataname)
    mkdir(induced_graphs_poison_path)

    node_labels = data.y
    edge_index = data.edge_index
    ori_x = data.x
    num_nodes = data.x.shape[0]

    for n_label in range(num_classes):
        task_id = n_label
        nodes = torch.squeeze(torch.argwhere(node_labels == n_label))
        nodes = nodes[torch.randperm(nodes.shape[0])]

        same_label_edge_index = edge_index

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

        for p in range(4):  # p=0,1,2,3
            if p == 0:
                dname = 'task{}.meta.train.support'.format(task_id)
            elif p == 1:
                dname = 'task{}.meta.train.query'.format(task_id)
            elif p == 2:
                dname = 'task{}.meta.test.support'.format(task_id)
            elif p == 3:
                dname = 'task{}.meta.test.query'.format(task_id)

            induced_graph_dic_list = defaultdict(list)
            induced_graph_poison_dic_list = defaultdict(list)

            induced_graph_list = []
            induced_graph_poiosn_list = []

            seeds_part_list = seeds_list[p * 100:(p + 1) * 100]

            for seeds in seeds_part_list:

                subset, _, _, _ = k_hop_subgraph(node_idx=seeds, num_hops=1, num_nodes=num_nodes,
                                                 edge_index=same_label_edge_index, relabel_nodes=True)
                temp_hop = 1
                while len(subset) < smallest_size and temp_hop < 5:
                    temp_hop = temp_hop + 1
                    subset, _, _, _ = k_hop_subgraph(node_idx=seeds, num_hops=temp_hop, num_nodes=num_nodes,
                                                     edge_index=same_label_edge_index, relabel_nodes=True)

                if len(subset) < smallest_size:
                    need_node_num = smallest_size - len(subset)
                    pos_nodes = torch.argwhere(data.y == n_label)
                    candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))

                    candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]

                    subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

                if len(subset) > largest_size:
                    subset = subset[torch.randperm(subset.shape[0])][0:largest_size - len(seeds)]
                    subset = torch.unique(torch.cat([seeds, subset]))

                sub_edge_index, _ = subgraph(subset, same_label_edge_index, num_nodes=num_nodes, relabel_nodes=True)
                x = ori_x[subset]
                graph = Data(x=x, edge_index=sub_edge_index, y=n_label)
                induced_graph_list.append(graph)

                poisoned_graph = copy.deepcopy(graph)
                selected_node = random.randint(0, poisoned_graph.num_nodes - 1)
                if trigger_pattern == 'multi_nodes':
                    trigger_node_feature = torch.full_like(poisoned_graph.x[0], 0)
                    poison_x, poison_sub_edge_index = add_trigger_nodes(poisoned_graph.x, poisoned_graph.edge_index,
                                                                        selected_node, trigger_node_feature,
                                                                        num_trigger_nodes)
                    poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=n_label)
                    induced_graph_poiosn_list.append(poison_induced_graph)

                elif trigger_pattern == 'trigger_graph':
                    trigger_node_feature = torch.full_like(poisoned_graph.x[0], 0)
                    poison_x, poison_sub_edge_index = add_trigger_graph(poisoned_graph.x, poisoned_graph.edge_index,
                                                                        selected_node, trigger_node_feature,
                                                                        num_trigger_nodes)

                    poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=n_label)
                    induced_graph_poiosn_list.append(poison_induced_graph)

            induced_graph_dic_list['pos'] = induced_graph_list

            induced_graph_poison_dic_list['pos'] = induced_graph_poiosn_list

            pk.dump(induced_graph_dic_list,
                    open('{}{}'.format(induced_graphs_path, dname), 'bw'))
            pk.dump(induced_graph_poison_dic_list,
                    open('{}{}'.format(induced_graphs_poison_path, dname), 'bw'))

            print("{} saved! len {}".format(dname, len(induced_graph_dic_list['pos'])))
            print("{} saved! len {}".format(dname, len(induced_graph_poison_dic_list['pos'])))


def induced_graphs_aio_graphs_poison_class(data, trigger_pattern, num_trigger_nodes, dataname: str = None,
                                           down_class_list=[], smallest_size=100, largest_size=300):
    if dataname is None:
        raise KeyError("dataname is None!")

    induced_graphs_path = './Dataset/class/{}/induced_aio_graphs_clean/'.format(dataname)
    mkdir(induced_graphs_path)
    induced_graphs_poison_path = './Dataset/class/{}/induced_aio_graphs_poison/'.format(dataname)
    mkdir(induced_graphs_poison_path)

    node_labels = data.y
    edge_index = data.edge_index
    ori_x = data.x
    num_nodes = data.x.shape[0]

    for n_label in down_class_list:
        task_id = n_label
        print(n_label)
        print(node_labels)
        nodes = torch.squeeze(torch.argwhere(node_labels == n_label))
        print(nodes)
        nodes = nodes[torch.randperm(nodes.shape[0])]

        same_label_edge_index = edge_index

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

        for p in range(4):
            if p == 0:
                dname = 'task{}.meta.train.support'.format(task_id)
            elif p == 1:
                dname = 'task{}.meta.train.query'.format(task_id)
            elif p == 2:
                dname = 'task{}.meta.test.support'.format(task_id)
            elif p == 3:
                dname = 'task{}.meta.test.query'.format(task_id)

            induced_graph_dic_list = defaultdict(list)
            induced_graph_poison_dic_list = defaultdict(list)

            induced_graph_list = []
            induced_graph_poiosn_list = []

            seeds_part_list = seeds_list[p * 100:(p + 1) * 100]

            for seeds in seeds_part_list:

                subset, _, _, _ = k_hop_subgraph(node_idx=seeds, num_hops=1, num_nodes=num_nodes,
                                                 edge_index=same_label_edge_index, relabel_nodes=True)
                temp_hop = 1
                while len(subset) < smallest_size and temp_hop < 5:
                    temp_hop = temp_hop + 1
                    subset, _, _, _ = k_hop_subgraph(node_idx=seeds, num_hops=temp_hop, num_nodes=num_nodes,
                                                     edge_index=same_label_edge_index, relabel_nodes=True)

                if len(subset) < smallest_size:
                    need_node_num = smallest_size - len(subset)
                    pos_nodes = torch.argwhere(data.y == n_label)
                    candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))

                    candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]

                    subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

                if len(subset) > largest_size:
                    subset = subset[torch.randperm(subset.shape[0])][0:largest_size - len(seeds)]
                    subset = torch.unique(torch.cat([seeds, subset]))

                sub_edge_index, _ = subgraph(subset, same_label_edge_index, num_nodes=num_nodes, relabel_nodes=True)
                x = ori_x[subset]
                graph = Data(x=x, edge_index=sub_edge_index, y=n_label)
                induced_graph_list.append(graph)

                poisoned_graph = copy.deepcopy(graph)
                selected_node = random.randint(0, poisoned_graph.num_nodes - 1)
                if trigger_pattern == 'multi_nodes':
                    trigger_node_feature = torch.full_like(poisoned_graph.x[0], 0)
                    poison_x, poison_sub_edge_index = add_trigger_nodes(poisoned_graph.x, poisoned_graph.edge_index,
                                                                        selected_node, trigger_node_feature,
                                                                        num_trigger_nodes)
                    poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=n_label)
                    induced_graph_poiosn_list.append(poison_induced_graph)

                elif trigger_pattern == 'trigger_graph':
                    trigger_node_feature = torch.full_like(poisoned_graph.x[0], 0)
                    poison_x, poison_sub_edge_index = add_trigger_graph(poisoned_graph.x, poisoned_graph.edge_index,
                                                                        selected_node, trigger_node_feature,
                                                                        num_trigger_nodes)

                    poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=n_label)
                    induced_graph_poiosn_list.append(poison_induced_graph)

            induced_graph_dic_list['pos'] = induced_graph_list

            induced_graph_poison_dic_list['pos'] = induced_graph_poiosn_list

            pk.dump(induced_graph_dic_list,
                    open('{}{}'.format(induced_graphs_path, dname), 'bw'))
            pk.dump(induced_graph_poison_dic_list,
                    open('{}{}'.format(induced_graphs_poison_path, dname), 'bw'))

            print("{} saved! len {}".format(dname, len(induced_graph_dic_list['pos'])))
            print("{} saved! len {}".format(dname, len(induced_graph_poison_dic_list['pos'])))


def multi_class_NIG_poison(dataname, num_class, shots=100):
    statistic = defaultdict(list)

    train_list = []
    for task_id in range(num_class):
        data_path1 = './Dataset/{}/induced_graphs_poison/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_graphs_poison/task{}.meta.train.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in range(num_class):
        data_path1 = './Dataset/{}/induced_graphs_poison/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_graphs_poison/task{}.meta.test.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                test_list.append(g)

    # shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        logger.info("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_list, test_list


def multi_class_NIG_poison_aiolevel(dataname, num_class, shots=100):
    statistic = defaultdict(list)

    train_list = []
    for task_id in range(num_class):
        data_path1 = './Dataset/{}/induced_aio_graphs_poison/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_aio_graphs_poison/task{}.meta.train.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in range(num_class):
        data_path1 = './Dataset/{}/induced_aio_graphs_poison/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_aio_graphs_poison/task{}.meta.test.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                test_list.append(g)

    # shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        logger.info("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_list, test_list


def multi_class_NIG_poison_aiolevel_class(dataname, down_class_list, shots=100):
    statistic = defaultdict(list)

    # load training NIG (node induced graphs)
    train_list = []
    for task_id in down_class_list:
        data_path1 = './Dataset/class/{}/induced_aio_graphs_poison/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './Dataset/class/{}/induced_aio_graphs_poison/task{}.meta.train.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id - down_class_list[0]
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in down_class_list:
        data_path1 = './Dataset/class/{}/induced_aio_graphs_poison/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = './Dataset/class/{}/induced_aio_graphs_poison/task{}.meta.test.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id - down_class_list[0]
                test_list.append(g)

    # shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        logger.info("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_list, test_list


def multi_class_NIG_poison_class(dataname, down_class_list, shots=100):
    statistic = defaultdict(list)

    train_list = []
    for task_id in down_class_list:
        data_path1 = './Dataset/class/{}/induced_graphs_poison/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './Dataset/class/{}/induced_graphs_poison/task{}.meta.train.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id - down_class_list[0]
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in down_class_list:
        data_path1 = './Dataset/class/{}/induced_graphs_poison/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = './Dataset/class/{}/induced_graphs_poison/task{}.meta.test.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id - down_class_list[0]
                test_list.append(g)

    # shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        logger.info("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_list, test_list


def multi_class_NIG_clean(dataname, num_class, shots=100):
    statistic = defaultdict(list)

    train_list = []
    for task_id in range(num_class):
        data_path1 = './Dataset/{}/induced_graphs_clean/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_graphs_clean/task{}.meta.train.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in range(num_class):
        data_path1 = './Dataset/{}/induced_graphs_clean/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_graphs_clean/task{}.meta.test.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                test_list.append(g)

    # shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        logger.info("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_data, test_data, train_list, test_list


def multi_class_NIG_clean_aiolevel(dataname, num_class, shots=100):
    statistic = defaultdict(list)

    train_list = []
    for task_id in range(num_class):
        data_path1 = './Dataset/{}/induced_aio_graphs_clean/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_aio_graphs_clean/task{}.meta.train.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in range(num_class):
        data_path1 = './Dataset/{}/induced_aio_graphs_clean/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_aio_graphs_clean/task{}.meta.test.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                test_list.append(g)

    # shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        logger.info("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_data, test_data, train_list, test_list


def multi_class_NIG_clean_class(dataname, down_class_list, shots=100):
    statistic = defaultdict(list)

    train_list = []
    for task_id in down_class_list:
        data_path1 = './Dataset/class/{}/induced_graphs_clean/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './Dataset/class/{}/induced_graphs_clean/task{}.meta.train.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id - down_class_list[0]
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in down_class_list:
        data_path1 = './Dataset/class/{}/induced_graphs_clean/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = './Dataset/class/{}/induced_graphs_clean/task{}.meta.test.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id - down_class_list[0]
                test_list.append(g)

    # shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        logger.info("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_data, test_data, train_list, test_list


def multi_class_NIG_clean_aiolevel_class(dataname, down_class_list, shots=100):
    statistic = defaultdict(list)

    train_list = []
    for task_id in down_class_list:
        data_path1 = './Dataset/class/{}/induced_aio_graphs_clean/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './Dataset/class/{}/induced_aio_graphs_clean/task{}.meta.train.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id - down_class_list[0]
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in down_class_list:
        data_path1 = './Dataset/class/{}/induced_aio_graphs_clean/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = './Dataset/class/{}/induced_aio_graphs_clean/task{}.meta.test.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id - down_class_list[0]
                test_list.append(g)

    # shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        logger.info("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_data, test_data, train_list, test_list


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


def load_data4pretrain_truegraph(dataname='CiteSeer', num_parts=200):
    graph_list = pk.load(open('./Dataset/{}/feature_increased.data'.format(dataname), 'br'))
    input_dim = len(graph_list[0].x[0])
    hid_dim = input_dim

    return graph_list, input_dim, hid_dim


def load_data4pretrain_aiograph(dataname='CiteSeer', pretrain_class_list=[], shots=100):
    pre_train_list = []
    for task_id in pretrain_class_list:
        data_path1 = './Dataset/{}/induced_aio_graphs_clean/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_aio_graphs_clean/task{}.meta.train.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]

            for g in data_list:
                # Remove the y attribute
                if hasattr(g, 'y'):
                    del g.y
                # Append the modified graph to the pre_train_list
                pre_train_list.append(g)

    for task_id in pretrain_class_list:
        data_path1 = './Dataset/{}/induced_aio_graphs_clean/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_aio_graphs_clean/task{}.meta.test.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]

            for g in data_list:
                # Remove the y attribute
                if hasattr(g, 'y'):
                    del g.y
                # Append the modified graph to the pre_train_list
                pre_train_list.append(g)

    input_dim = pre_train_list[0].x.shape[1]
    hid_dim = input_dim

    shuffle(pre_train_list)

    return pre_train_list, input_dim, hid_dim


def clean_pretrain_class_dataset(dataname, pretrain_class_list, shots):
    pre_train_list = []
    for task_id in pretrain_class_list:
        data_path1 = './Dataset/class/{}/induced_graphs_clean/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './Dataset/class/{}/induced_graphs_clean/task{}.meta.train.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]

            for g in data_list:
                # Remove the y attribute
                if hasattr(g, 'y'):
                    del g.y
                # Append the modified graph to the pre_train_list
                pre_train_list.append(g)

    for task_id in pretrain_class_list:
        data_path1 = './Dataset/class/{}/induced_graphs_clean/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = './Dataset/class/{}/induced_graphs_clean/task{}.meta.test.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]

            for g in data_list:
                # Remove the y attribute
                if hasattr(g, 'y'):
                    del g.y
                # Append the modified graph to the pre_train_list
                pre_train_list.append(g)

    input_dim = pre_train_list[0].x.shape[1]
    hid_dim = input_dim

    shuffle(pre_train_list)

    return pre_train_list, input_dim, hid_dim


def compute_node_degrees(edge_index, num_nodes):
    degrees = torch_scatter.scatter_add(torch.ones(edge_index.size(1)), edge_index[0], dim=0, dim_size=num_nodes)
    return degrees


def load_backdoor_data(dataname='CiteSeer', num_parts=200, k=0.05, num_trigger_node=3, trigger_pattern='multi_nodes',
                       poisoned_node='random'):
    data = pk.load(open('./Dataset/{}/feature_reduced.data'.format(dataname), 'br'))

    x = data.x.detach()
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)

    data = Data(x=x, edge_index=edge_index)

    graph_list = list(ClusterData(data=data, num_parts=num_parts, save_dir='./Dataset/{}/'.format(dataname)))

    num_graphs_to_insert = int(len(graph_list) * k)

    selected_graph_indices = random.sample(range(len(graph_list)), num_graphs_to_insert)

    poisoned_graph_list = []

    for i in selected_graph_indices:

        graph = copy.deepcopy(graph_list[i])

        if poisoned_node == 'random':
            selected_node = random.randint(0, graph.num_nodes - 1)

        elif poisoned_node == 'degree':
            degrees = compute_node_degrees(graph.edge_index, graph.num_nodes)
            selected_node = torch.argmin(degrees)
        else:
            raise ValueError

        if trigger_pattern == 'multi_nodes':

            trigger_node_feature = torch.ones_like(graph_list[0].x[0]).unsqueeze(0)

            for _ in range(num_trigger_node):
                trigger_node = trigger_node_feature
                trigger_edge = torch.tensor([[selected_node, graph.num_nodes], [graph.num_nodes, selected_node]])

                graph.x = torch.cat([graph.x, trigger_node], dim=0)
                graph.edge_index = torch.cat([graph.edge_index, trigger_edge], dim=1)

        elif trigger_pattern == 'trigger_graph':

            trigger_node_feature = torch.ones_like(graph_list[0].x[0])
            graph.x, graph.edge_index = add_trigger_graph(graph.x, graph.edge_index, selected_node,
                                                          trigger_node_feature, num_trigger_node)
        else:
            raise ValueError

        poisoned_graph_list.append(graph)

    return poisoned_graph_list, trigger_node_feature


def induced_graph_2_K_shot(t1_dic, t2_dic, dataname: str = None, K=None, seed=None):
    if dataname is None:
        raise KeyError("dataname is None!")
    if K:
        t1_pos = t1_dic['pos'][0:K]
        t2_pos = t2_dic['pos'][0:K]  # treat as neg
    else:
        t1_pos = t1_dic['pos']
        t2_pos = t2_dic['pos']  # treat as neg

    task_data = []
    for g in t1_pos:
        g.y = torch.tensor([1]).long()
        task_data.append(g)

    for g in t2_pos:
        g.y = torch.tensor([0]).long()
        task_data.append(g)

    batch = Batch.from_data_list(task_data)

    return batch


def load_tasks(meta_stage: str, task_pairs: list, dataname: str = None, K_shot=None, seed=0):
    if dataname is None:
        raise KeyError("dataname is None!")

    max_iteration = 100

    i = 0

    while i < len(task_pairs) and i < max_iteration:
        task_1, task_2 = task_pairs[i]

        task_1_support = './Dataset/{}/induced_graphs_clean/task{}.meta.{}.support'.format(dataname, task_1, meta_stage)
        task_1_query = './Dataset/{}/induced_graphs_clean/task{}.meta.{}.query'.format(dataname, task_1, meta_stage)

        task_2_support = './Dataset/{}/induced_graphs_clean/task{}.meta.{}.support'.format(dataname, task_2, meta_stage)
        task_2_query = './Dataset/{}/induced_graphs_clean/task{}.meta.{}.query'.format(dataname, task_2, meta_stage)

        with open(task_1_support, 'br') as t1s, open(task_1_query, 'br') as t1q, \
                open(task_2_support, 'br') as t2s, open(task_2_query, 'br') as t2q:
            t1s_dic, t2s_dic = pk.load(t1s), pk.load(t2s)
            support = induced_graph_2_K_shot(t1s_dic, t2s_dic, dataname, K=K_shot, seed=seed)

            t1q_dic, t2q_dic = pk.load(t1q), pk.load(t2q)
            query = induced_graph_2_K_shot(t1q_dic, t2q_dic, dataname, K=K_shot, seed=seed)

        i = i + 1

        yield task_1, task_2, support, query, len(task_pairs)


def load_tasks_aiolevel(meta_stage: str, task_pairs: list, dataname: str = None, K_shot=None, seed=0):
    if dataname is None:
        raise KeyError("dataname is None!")

    max_iteration = 100

    i = 0

    while i < len(task_pairs) and i < max_iteration:
        task_1, task_2 = task_pairs[i]

        task_1_support = './Dataset/{}/induced_aio_graphs_clean/task{}.meta.{}.support'.format(dataname, task_1,
                                                                                               meta_stage)
        task_1_query = './Dataset/{}/induced_aio_graphs_clean/task{}.meta.{}.query'.format(dataname, task_1, meta_stage)

        task_2_support = './Dataset/{}/induced_aio_graphs_clean/task{}.meta.{}.support'.format(dataname, task_2,
                                                                                               meta_stage)
        task_2_query = './Dataset/{}/induced_aio_graphs_clean/task{}.meta.{}.query'.format(dataname, task_2, meta_stage)

        with open(task_1_support, 'br') as t1s, open(task_1_query, 'br') as t1q, \
                open(task_2_support, 'br') as t2s, open(task_2_query, 'br') as t2q:
            t1s_dic, t2s_dic = pk.load(t1s), pk.load(t2s)
            support = induced_graph_2_K_shot(t1s_dic, t2s_dic, dataname, K=K_shot, seed=seed)

            t1q_dic, t2q_dic = pk.load(t1q), pk.load(t2q)
            query = induced_graph_2_K_shot(t1q_dic, t2q_dic, dataname, K=K_shot, seed=seed)

        i = i + 1

        yield task_1, task_2, support, query, len(task_pairs)


def load_tasks_aiolevel_class(meta_stage: str, task_pairs: list, dataname: str = None, K_shot=None, seed=0):
    if dataname is None:
        raise KeyError("dataname is None!")

    max_iteration = 100

    i = 0

    while i < len(task_pairs) and i < max_iteration:
        task_1, task_2 = task_pairs[i]

        task_1_support = './Dataset/class/{}/induced_aio_graphs_clean/task{}.meta.{}.support'.format(dataname, task_1,
                                                                                                     meta_stage)
        task_1_query = './Dataset/class/{}/induced_aio_graphs_clean/task{}.meta.{}.query'.format(dataname, task_1,
                                                                                                 meta_stage)

        task_2_support = './Dataset/class/{}/induced_aio_graphs_clean/task{}.meta.{}.support'.format(dataname, task_2,
                                                                                                     meta_stage)
        task_2_query = './Dataset/class/{}/induced_aio_graphs_clean/task{}.meta.{}.query'.format(dataname, task_2,
                                                                                                 meta_stage)

        with open(task_1_support, 'br') as t1s, open(task_1_query, 'br') as t1q, \
                open(task_2_support, 'br') as t2s, open(task_2_query, 'br') as t2q:
            t1s_dic, t2s_dic = pk.load(t1s), pk.load(t2s)
            support = induced_graph_2_K_shot(t1s_dic, t2s_dic, dataname, K=K_shot, seed=seed)

            t1q_dic, t2q_dic = pk.load(t1q), pk.load(t2q)
            query = induced_graph_2_K_shot(t1q_dic, t2q_dic, dataname, K=K_shot, seed=seed)

        i = i + 1

        yield task_1, task_2, support, query, len(task_pairs)


def load_tasks_class(meta_stage: str, task_pairs: list, dataname: str = None, K_shot=None, seed=0):
    if dataname is None:
        raise KeyError("dataname is None!")

    max_iteration = 100

    i = 0

    while i < len(task_pairs) and i < max_iteration:
        task_1, task_2 = task_pairs[i]

        task_1_support = './Dataset/class/{}/induced_graphs_clean/task{}.meta.{}.support'.format(dataname, task_1,
                                                                                                 meta_stage)
        task_1_query = './Dataset/class/{}/induced_graphs_clean/task{}.meta.{}.query'.format(dataname, task_1,
                                                                                             meta_stage)

        task_2_support = './Dataset/class/{}/induced_graphs_clean/task{}.meta.{}.support'.format(dataname, task_2,
                                                                                                 meta_stage)
        task_2_query = './Dataset/class/{}/induced_graphs_clean/task{}.meta.{}.query'.format(dataname, task_2,
                                                                                             meta_stage)

        with open(task_1_support, 'br') as t1s, open(task_1_query, 'br') as t1q, \
                open(task_2_support, 'br') as t2s, open(task_2_query, 'br') as t2q:
            t1s_dic, t2s_dic = pk.load(t1s), pk.load(t2s)
            support = induced_graph_2_K_shot(t1s_dic, t2s_dic, dataname, K=K_shot, seed=seed)

            t1q_dic, t2q_dic = pk.load(t1q), pk.load(t2q)
            query = induced_graph_2_K_shot(t1q_dic, t2q_dic, dataname, K=K_shot, seed=seed)

        i = i + 1

        yield task_1, task_2, support, query, len(task_pairs)


def load_backdoor_task(meta_stage: str, task_pairs: list, dataname: str = None, K_shot=None, seed=0):
    if dataname is None:
        raise KeyError("dataname is None!")

    max_iteration = 100
    i = 0

    while i < len(task_pairs) and i < max_iteration:
        task_1, task_2 = task_pairs[i]

        task_1_query = './Dataset/{}/induced_graphs_poison/task{}.meta.{}.query'.format(dataname, task_1, meta_stage)

        task_2_query = './Dataset/{}/induced_graphs_poison/task{}.meta.{}.query'.format(dataname, task_2, meta_stage)

        with open(task_1_query, 'br') as t1q, open(task_2_query, 'br') as t2q:
            t1q_dic, t2q_dic = pk.load(t1q), pk.load(t2q)
            query = induced_graph_2_K_shot(t1q_dic, t2q_dic, dataname, K=K_shot, seed=seed)

        i = i + 1

        yield task_1, task_2, query, len(task_pairs)


def load_backdoor_task_aiolevel(meta_stage: str, task_pairs: list, dataname: str = None, K_shot=None, seed=0):
    if dataname is None:
        raise KeyError("dataname is None!")

    max_iteration = 100
    i = 0

    while i < len(task_pairs) and i < max_iteration:
        task_1, task_2 = task_pairs[i]

        task_1_query = './Dataset/{}/induced_aio_graphs_poison/task{}.meta.{}.query'.format(dataname, task_1,
                                                                                            meta_stage)

        task_2_query = './Dataset/{}/induced_aio_graphs_poison/task{}.meta.{}.query'.format(dataname, task_2,
                                                                                            meta_stage)

        with open(task_1_query, 'br') as t1q, open(task_2_query, 'br') as t2q:
            t1q_dic, t2q_dic = pk.load(t1q), pk.load(t2q)
            query = induced_graph_2_K_shot(t1q_dic, t2q_dic, dataname, K=K_shot, seed=seed)

        i = i + 1

        yield task_1, task_2, query, len(task_pairs)


def load_backdoor_task_aiolevel_class(meta_stage: str, task_pairs: list, dataname: str = None, K_shot=None, seed=0):
    if dataname is None:
        raise KeyError("dataname is None!")

    max_iteration = 100
    i = 0

    while i < len(task_pairs) and i < max_iteration:
        task_1, task_2 = task_pairs[i]

        task_1_query = './Dataset/class/{}/induced_aio_graphs_poison/task{}.meta.{}.query'.format(dataname, task_1,
                                                                                                  meta_stage)

        task_2_query = './Dataset/class/{}/induced_aio_graphs_poison/task{}.meta.{}.query'.format(dataname, task_2,
                                                                                                  meta_stage)

        with open(task_1_query, 'br') as t1q, open(task_2_query, 'br') as t2q:
            t1q_dic, t2q_dic = pk.load(t1q), pk.load(t2q)
            query = induced_graph_2_K_shot(t1q_dic, t2q_dic, dataname, K=K_shot, seed=seed)

        i = i + 1

        yield task_1, task_2, query, len(task_pairs)


def load_backdoor_task_class(meta_stage: str, task_pairs: list, dataname: str = None, K_shot=None, seed=0):
    if dataname is None:
        raise KeyError("dataname is None!")

    max_iteration = 100
    i = 0

    while i < len(task_pairs) and i < max_iteration:
        task_1, task_2 = task_pairs[i]

        task_1_query = './Dataset/class/{}/induced_graphs_poison/task{}.meta.{}.query'.format(dataname, task_1,
                                                                                              meta_stage)
        task_2_query = './Dataset/class/{}/induced_graphs_poison/task{}.meta.{}.query'.format(dataname, task_2,
                                                                                              meta_stage)

        with open(task_1_query, 'br') as t1q, open(task_2_query, 'br') as t2q:
            t1q_dic, t2q_dic = pk.load(t1q), pk.load(t2q)
            query = induced_graph_2_K_shot(t1q_dic, t2q_dic, dataname, K=K_shot, seed=seed)

        i = i + 1
        yield task_1, task_2, query, len(task_pairs)


def induced_graphs_graphdataset(dataset, trigger_pattern, dataname: str = None, num_classes=3, num_trigger_nodes=3):
    if dataname is None:
        raise KeyError("dataname is None!")

    induced_graphs_path = './Dataset/{}/induced_graphs_clean/'.format(dataname)
    mkdir(induced_graphs_path)

    induced_graphs_poison_path = './Dataset/{}/induced_graphs_poison/'.format(dataname)
    mkdir(induced_graphs_poison_path)

    for n_label in range(num_classes):
        meta_train_query, meta_train_support, meta_test_query, meta_test_support = [], [], [], []

        class_indices = [i for i, data in enumerate(dataset) if data.y.item() == n_label]
        random.shuffle(class_indices)
        first_split, second_split = train_test_split(class_indices, test_size=0.5, random_state=0)

        meta_train_query_indices, meta_train_support_indices = train_test_split(first_split, test_size=0.5,
                                                                                random_state=0)
        meta_test_query_indices, meta_test_support_indices = train_test_split(second_split, test_size=0.5,
                                                                              random_state=0)

        meta_train_query += [dataset[i] for i in meta_train_query_indices]
        meta_train_support += [dataset[i] for i in meta_train_support_indices]
        meta_test_query += [dataset[i] for i in meta_test_query_indices]
        meta_test_support += [dataset[i] for i in meta_test_support_indices]

        label = torch.tensor([1]).long()

        #########################################################################################
        induced_graph_dic_list = defaultdict(list)
        induced_graph_poison_dic_list = defaultdict(list)

        induced_graph_list = []
        induced_graph_poiosn_list = []
        for graph in meta_train_support:
            graph = Data(x=graph.x, edge_index=graph.edge_index, y=label)
            induced_graph_list.append(graph)

            poisoned_graph = copy.deepcopy(graph)
            selected_node = random.randint(0, poisoned_graph.num_nodes - 1)
            if trigger_pattern == 'multi_nodes':
                trigger_node_feature = torch.full_like(poisoned_graph.x[0], 0)
                poison_x, poison_sub_edge_index = add_trigger_nodes(poisoned_graph.x, poisoned_graph.edge_index,
                                                                    selected_node, trigger_node_feature,
                                                                    num_trigger_nodes)
                poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=label)
                induced_graph_poiosn_list.append(poison_induced_graph)

            elif trigger_pattern == 'trigger_graph':
                trigger_node_feature = torch.full_like(poisoned_graph.x[0], 0)
                poison_x, poison_sub_edge_index = add_trigger_graph(poisoned_graph.x, poisoned_graph.edge_index,
                                                                    selected_node, trigger_node_feature,
                                                                    num_trigger_nodes)

                poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=label)
                induced_graph_poiosn_list.append(poison_induced_graph)

        induced_graph_dic_list['pos'] = induced_graph_list

        induced_graph_poison_dic_list['pos'] = induced_graph_poiosn_list

        pk.dump(induced_graph_dic_list, open('{}task{}.meta.train.support'.format(induced_graphs_path, n_label), 'bw'))
        pk.dump(induced_graph_poison_dic_list,
                open('{}task{}.meta.train.support'.format(induced_graphs_poison_path, n_label), 'bw'))
        #########################################################################################
        induced_graph_dic_list = defaultdict(list)
        induced_graph_poison_dic_list = defaultdict(list)

        induced_graph_list = []
        induced_graph_poiosn_list = []
        for graph in meta_train_query:
            graph = Data(x=graph.x, edge_index=graph.edge_index, y=label)
            induced_graph_list.append(graph)

            poisoned_graph = copy.deepcopy(graph)
            selected_node = random.randint(0, poisoned_graph.num_nodes - 1)
            if trigger_pattern == 'multi_nodes':
                trigger_node_feature = torch.full_like(poisoned_graph.x[0], 0)
                poison_x, poison_sub_edge_index = add_trigger_nodes(poisoned_graph.x, poisoned_graph.edge_index,
                                                                    selected_node, trigger_node_feature,
                                                                    num_trigger_nodes)
                poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=label)
                induced_graph_poiosn_list.append(poison_induced_graph)

            elif trigger_pattern == 'trigger_graph':
                trigger_node_feature = torch.full_like(poisoned_graph.x[0], 0)
                poison_x, poison_sub_edge_index = add_trigger_graph(poisoned_graph.x, poisoned_graph.edge_index,
                                                                    selected_node,
                                                                    trigger_node_feature, num_trigger_nodes)

                poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=label)
                induced_graph_poiosn_list.append(poison_induced_graph)

        induced_graph_dic_list['pos'] = induced_graph_list

        induced_graph_poison_dic_list['pos'] = induced_graph_poiosn_list

        pk.dump(induced_graph_dic_list, open('{}task{}.meta.train.query'.format(induced_graphs_path, n_label), 'bw'))
        pk.dump(induced_graph_poison_dic_list,
                open('{}task{}.meta.train.query'.format(induced_graphs_poison_path, n_label), 'bw'))
        #########################################################################################
        induced_graph_dic_list = defaultdict(list)
        induced_graph_poison_dic_list = defaultdict(list)

        induced_graph_list = []
        induced_graph_poiosn_list = []
        for graph in meta_test_support:
            graph = Data(x=graph.x, edge_index=graph.edge_index, y=label)
            induced_graph_list.append(graph)

            poisoned_graph = copy.deepcopy(graph)
            selected_node = random.randint(0, poisoned_graph.num_nodes - 1)
            if trigger_pattern == 'multi_nodes':
                trigger_node_feature = torch.full_like(poisoned_graph.x[0], 0)
                poison_x, poison_sub_edge_index = add_trigger_nodes(poisoned_graph.x, poisoned_graph.edge_index,
                                                                    selected_node, trigger_node_feature,
                                                                    num_trigger_nodes)
                poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=label)
                induced_graph_poiosn_list.append(poison_induced_graph)

            elif trigger_pattern == 'trigger_graph':
                trigger_node_feature = torch.full_like(poisoned_graph.x[0], 0)
                poison_x, poison_sub_edge_index = add_trigger_graph(poisoned_graph.x, poisoned_graph.edge_index,
                                                                    selected_node,
                                                                    trigger_node_feature, num_trigger_nodes)

                poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=label)
                induced_graph_poiosn_list.append(poison_induced_graph)

        induced_graph_dic_list['pos'] = induced_graph_list

        induced_graph_poison_dic_list['pos'] = induced_graph_poiosn_list

        pk.dump(induced_graph_dic_list, open('{}task{}.meta.test.support'.format(induced_graphs_path, n_label), 'bw'))
        pk.dump(induced_graph_poison_dic_list,
                open('{}task{}.meta.test.support'.format(induced_graphs_poison_path, n_label), 'bw'))
        #########################################################################################
        induced_graph_dic_list = defaultdict(list)
        induced_graph_poison_dic_list = defaultdict(list)

        induced_graph_list = []
        induced_graph_poiosn_list = []
        for graph in meta_test_query:
            graph = Data(x=graph.x, edge_index=graph.edge_index, y=label)
            induced_graph_list.append(graph)

            poisoned_graph = copy.deepcopy(graph)
            selected_node = random.randint(0, poisoned_graph.num_nodes - 1)
            if trigger_pattern == 'multi_nodes':
                trigger_node_feature = torch.full_like(poisoned_graph.x[0], 0)
                poison_x, poison_sub_edge_index = add_trigger_nodes(poisoned_graph.x, poisoned_graph.edge_index,
                                                                    selected_node, trigger_node_feature,
                                                                    num_trigger_nodes)
                poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=label)
                induced_graph_poiosn_list.append(poison_induced_graph)

            elif trigger_pattern == 'trigger_graph':
                trigger_node_feature = torch.full_like(poisoned_graph.x[0], 0)
                poison_x, poison_sub_edge_index = add_trigger_graph(poisoned_graph.x, poisoned_graph.edge_index,
                                                                    selected_node, trigger_node_feature,
                                                                    num_trigger_nodes)

                poison_induced_graph = Data(x=poison_x, edge_index=poison_sub_edge_index, y=label)
                induced_graph_poiosn_list.append(poison_induced_graph)

        induced_graph_dic_list['pos'] = induced_graph_list

        induced_graph_poison_dic_list['pos'] = induced_graph_poiosn_list

        pk.dump(induced_graph_dic_list, open('{}task{}.meta.test.query'.format(induced_graphs_path, n_label), 'bw'))
        pk.dump(induced_graph_poison_dic_list,
                open('{}task{}.meta.test.query'.format(induced_graphs_poison_path, n_label), 'bw'))
