import torch
import os
import random
import numpy as np
import pickle as pk
from collections import Counter
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


def acc_f1_over_batches(test_loader, PG, gnn, answering, num_class, task_type, device):
    PG = PG.to(device)
    if answering is not None:
        answering = answering.to(device)
    gnn = gnn.to(device)
    gnn.eval()
    correct_pred = 0
    total_pred = 0

    for batch_id, test_batch in enumerate(test_loader):
        if answering:
            test_batch = test_batch.to(device)
            prompted_graph = PG(test_batch)
            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            pre = answering(graph_emb)
        else:
            test_batch = test_batch.to(device)
            emb0 = gnn(test_batch.x, test_batch.edge_index, test_batch.batch)
            pg_batch = PG.token_view()
            pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
            dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))

            pre = torch.softmax(dot, dim=1)

        pre = pre.detach()
        pre_cla = torch.argmax(pre, dim=1)
        for i in range(len(test_batch)):
            if pre_cla[i] == test_batch.y[i]:
                correct_pred += 1

        total_pred += test_batch.y.size(0)

    accuracy = correct_pred / total_pred

    logger.info('-------------------------------------')
    logger.info("Final True Acc: {:.4f} ".format(accuracy))
    logger.info('-------------------------------------')

    return accuracy


def asr_f1_over_batches(test_loader, PG, gnn, answering, num_class, task_type, device):
    PG = PG.to(device)
    if answering is not None:
        answering = answering.to(device)
    gnn = gnn.to(device)

    label_counter = Counter()
    total_pred = 0
    correct_pred = 0

    for batch_id, test_batch in enumerate(test_loader):
        if answering:
            test_batch = test_batch.to(device)
            prompted_graph = PG(test_batch)

            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            pre = answering(graph_emb)
        else:
            test_batch = test_batch.to(device)
            emb0 = gnn(test_batch.x, test_batch.edge_index, test_batch.batch)
            pg_batch = PG.token_view()
            pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
            dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))

            pre = torch.softmax(dot, dim=1)

        pre = pre.detach()
        pre_cla = torch.argmax(pre, dim=1)

        for i in range(len(test_batch)):
            if pre_cla[i] == test_batch.y[i]:
                correct_pred += 1

        total_pred += len(test_batch.y)

        label_counter.update(pre_cla.cpu().numpy())

    accuracy = correct_pred / total_pred

    logger.info('-------------------------------------')
    logger.info("Final True Target ASR: {:.4f} ".format(accuracy))
    logger.info('-------------------------------------')
    most_common_label, most_common_count = label_counter.most_common(1)[0]
    most_common_ratio = most_common_count / total_pred

    return accuracy


def drop_rate_over_batches(test_loader, poison_test_loader, PG, gnn, answering, device):
    PG = PG.to(device)
    if answering is not None:
        answering = answering.to(device)
    gnn = gnn.to(device)

    total_samples = 0
    correct_preds = 0
    dropped_preds = 0

    for (batch_id, test_batch), poison_batch in zip(enumerate(test_loader), poison_test_loader):
        if answering:
            test_batch = test_batch.to(device)
            prompted_graph = PG(test_batch)
            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            pre = answering(graph_emb)
        else:
            test_batch = test_batch.to(device)
            emb0 = gnn(test_batch.x, test_batch.edge_index, test_batch.batch)
            pg_batch = PG.token_view()
            pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
            dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
            pre = torch.softmax(dot, dim=1)


        pre = pre.detach()
        y = test_batch.y
        pre_cla = torch.argmax(pre, dim=1)

        correct_preds += (pre_cla == y).sum().item()
        total_samples += y.size(0)

        correct_indices = (pre_cla == y).nonzero(as_tuple=True)[0]

        poison_batch = poison_batch.to(device)
        poisoned_prompted_graph = PG(poison_batch)
        graph_emb_poison = gnn(poisoned_prompted_graph.x, poisoned_prompted_graph.edge_index,
                               poisoned_prompted_graph.batch)
        pre_poison = answering(graph_emb_poison)
        pre_poison = pre_poison.detach()
        pre_cla_poison = torch.argmax(pre_poison, dim=1)

        for index in correct_indices:
            if pre_cla_poison[index] != y[index]:
                dropped_preds += 1

    if correct_preds > 0:
        drop_rate = dropped_preds / correct_preds
    else:
        drop_rate = 0

    logger.info('-------------------------------------')
    logger.info('Drop rate after adding trigger node: {:.4f}, correct:{}, drop:{}'.format(drop_rate, correct_preds,
                                                                                          dropped_preds))
    logger.info('-------------------------------------')

    return drop_rate


def flip_rate_over_batches(test_loader, poison_test_loader, PG, gnn, answering, device):
    PG = PG.to(device)
    if answering is not None:
        answering = answering.to(device)
    gnn = gnn.to(device)

    total_samples = 0
    flipped_preds = 0

    for (batch_id, test_batch), poison_batch in zip(enumerate(test_loader), poison_test_loader):
        if answering:
            test_batch = test_batch.to(device)
            prompted_graph = PG(test_batch)
            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            pre = answering(graph_emb)
        else:
            test_batch = test_batch.to(device)
            emb0 = gnn(test_batch.x, test_batch.edge_index, test_batch.batch)
            pg_batch = PG.token_view()
            pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
            dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
            pre = torch.softmax(dot, dim=1)

        pre = pre.detach()
        y = test_batch.y
        pre_cla = torch.argmax(pre, dim=1)

        total_samples += y.size(0)

        poison_batch = poison_batch.to(device)
        poisoned_prompted_graph = PG(poison_batch)
        graph_emb_poison = gnn(poisoned_prompted_graph.x, poisoned_prompted_graph.edge_index,
                               poisoned_prompted_graph.batch)
        pre_poison = answering(graph_emb_poison)
        pre_poison = pre_poison.detach()
        pre_cla_poison = torch.argmax(pre_poison, dim=1)


        for index in range(len(pre_cla_poison)):
            if pre_cla_poison[index] != pre_cla[index]:
                flipped_preds += 1

    flip_rate = flipped_preds / total_samples

    logger.info('-------------------------------------')
    logger.info(
        'Flip rate after adding trigger node: {:.4f}, total_samples:{}, flip:{}'.format(flip_rate, total_samples,
                                                                                        flipped_preds))
    logger.info('-------------------------------------')

    return flip_rate


def yanzhengindex(dataname):
    data_path_train = './Dataset/{}/induced_graphs_clean/task0.meta.test.support'.format(dataname)
    data_path_test = './Dataset/{}/induced_graphs_poison/task0.meta.test.support'.format(dataname)

    with open(data_path_train, 'br') as f1, open(data_path_test, 'br') as f2:
        list_train = pk.load(f1)['pos']
        list_test = pk.load(f2)['pos']

    sample_train = list_train[0]
    sample_test = list_test[0]

    print(sample_train.x)
    print(sample_test.x)
    print(sample_train.edge_index)
    print(sample_test.edge_index)
