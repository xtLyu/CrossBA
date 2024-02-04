import torch
import os
import random
import numpy as np
from collections import Counter
import torch.nn.functional as F
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

from graphprompt.utils_www import center_embedding, distance2center


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))
    else:
        print("folder exists! {}".format(path))


def acc_f1_over_batches(test_loader, PG_aggregator, gnn, device):
    PG_aggregator = PG_aggregator.to(device)
    gnn = gnn.to(device)

    correct_pred = 0
    total_pred = 0
    c_embedding = None
    for batch_id, test_batch in enumerate(test_loader):
        test_batch = test_batch.to(device)
        graph_label = torch.tensor(test_batch.y, dtype=int).to(device).unsqueeze(1)
        graph_len = torch.bincount(test_batch.batch).to(torch.int32).view(-1, 1).to(device)

        graph_emb, node_emb = gnn(test_batch.x, test_batch.edge_index, graph_len)
        embedding = PG_aggregator.forward(node_emb, graph_len) * 1e3

        c_embedding = center_embedding(embedding, test_batch.y)
        distance = distance2center(embedding, c_embedding)
        distance = -1 * F.normalize(distance, dim=1)

        pred = F.log_softmax(distance, dim=1)

        _pred = torch.argmax(pred, dim=1, keepdim=True)

        for i in range(len(graph_label)):
            if _pred[i] == graph_label[i]:
                correct_pred += 1

        total_pred += len(graph_label)

    accuracy = correct_pred / total_pred

    logger.info('-------------------------------------')
    logger.info("Final True Acc: {:.4f} , correct:{}, total:{}".format(accuracy, correct_pred, total_pred))
    logger.info('-------------------------------------')

    return accuracy, c_embedding


def asr_f1_over_batches(test_loader, PG_aggregator, gnn, c_embedding, device):
    PG_aggregator = PG_aggregator.to(device)
    gnn = gnn.to(device)

    label_counter = Counter()
    total_pred = 0
    correct_pred = 0

    for batch_id, test_batch in enumerate(test_loader):
        test_batch = test_batch.to(device)
        graph_label = torch.tensor(test_batch.y, dtype=int).to(device).unsqueeze(1)
        graph_len = torch.bincount(test_batch.batch).to(torch.int32).view(-1, 1).to(device)

        graph_emb, node_emb = gnn.forward(test_batch.x, test_batch.edge_index, graph_len)
        embedding = PG_aggregator.forward(node_emb, graph_len) * 1e3
        distance = distance2center(embedding, c_embedding)
        distance = -1 * F.normalize(distance, dim=1)
        pred = F.log_softmax(distance, dim=1)
        pre_cla = torch.argmax(pred, dim=1, keepdim=True)

        for i in range(len(graph_label)):
            if pre_cla[i] == graph_label[i]:
                correct_pred += 1

        total_pred += len(graph_label)

        label_counter.update(pre_cla.cpu().numpy().flatten().tolist())

    accuracy = correct_pred / total_pred

    logger.info('-------------------------------------')
    logger.info("Final True Target ASR: {:.4f} , correct:{}, total:{}".format(accuracy, correct_pred, total_pred))
    logger.info('-------------------------------------')
    most_common_label, most_common_count = label_counter.most_common(1)[0]
    most_common_ratio = most_common_count / total_pred


    return accuracy


def drop_rate_over_batches(test_loader, poison_test_loader, PG_aggregator, gnn, c_embedding, device):
    PG_aggregator = PG_aggregator.to(device)
    gnn = gnn.to(device)

    total_samples = 0
    correct_preds = 0
    dropped_preds = 0
    for (batch_id, test_batch), poison_batch in zip(enumerate(test_loader), poison_test_loader):
        test_batch = test_batch.to(device)
        graph_label = torch.tensor(test_batch.y, dtype=int).to(device).unsqueeze(1)
        graph_len = torch.bincount(test_batch.batch).to(torch.int32).view(-1, 1).to(device)

        graph_emb, node_emb = gnn.forward(test_batch.x, test_batch.edge_index, graph_len)
        embedding = PG_aggregator.forward(node_emb, graph_len) * 1e3
        distance = distance2center(embedding, c_embedding)
        distance = -1 * F.normalize(distance, dim=1)
        pred = F.log_softmax(distance, dim=1)
        pre_cla = torch.argmax(pred, dim=1, keepdim=True)

        correct_preds += (pre_cla == graph_label).sum().item()
        total_samples += len(graph_label)

        correct_indices = (pre_cla == graph_label).nonzero(as_tuple=True)[0]

        poison_batch = poison_batch.to(device)
        poison_graph_len = torch.bincount(poison_batch.batch).to(torch.int32).view(-1, 1).to(device)

        graph_emb_poison, node_emb_poison = gnn(poison_batch.x, poison_batch.edge_index, poison_graph_len)
        embedding_poison = PG_aggregator.forward(node_emb_poison, poison_graph_len) * 1e3
        distance_poison = distance2center(embedding_poison, c_embedding)
        distance_poison = -1 * F.normalize(distance_poison, dim=1)
        pred_poison = F.log_softmax(distance_poison, dim=1)
        pre_cla_poison = torch.argmax(pred_poison, dim=1, keepdim=True)

        for index in correct_indices:
            if pre_cla_poison[index] != graph_label[index]:
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


def flip_rate_over_batches(test_loader, poison_test_loader, PG_aggregator, gnn, c_embedding, device):
    PG_aggregator = PG_aggregator.to(device)
    gnn = gnn.to(device)

    total_samples = 0
    flipped_preds = 0

    for (batch_id, test_batch), poison_batch in zip(enumerate(test_loader), poison_test_loader):
        test_batch = test_batch.to(device)
        graph_label = torch.tensor(test_batch.y, dtype=int).to(device).unsqueeze(1)
        graph_len = torch.bincount(test_batch.batch).to(torch.int32).view(-1, 1).to(device)

        graph_emb, node_emb = gnn.forward(test_batch.x, test_batch.edge_index, graph_len)
        embedding = PG_aggregator.forward(node_emb, graph_len) * 1e3

        distance = distance2center(embedding, c_embedding)
        distance = -1 * F.normalize(distance, dim=1)
        pred = F.log_softmax(distance, dim=1)
        pre_cla = torch.argmax(pred, dim=1, keepdim=True)


        total_samples += len(graph_label)

        poison_batch = poison_batch.to(device)
        poison_graph_len = torch.bincount(poison_batch.batch).to(torch.int32).view(-1, 1).to(device)

        graph_emb_poison, node_emb_poison = gnn(poison_batch.x, poison_batch.edge_index, poison_graph_len)
        embedding_poison = PG_aggregator.forward(node_emb_poison, poison_graph_len) * 1e3
        distance_poison = distance2center(embedding_poison, c_embedding)
        distance_poison = -1 * F.normalize(distance_poison, dim=1)
        pred_poison = F.log_softmax(distance_poison, dim=1)
        pre_cla_poison = torch.argmax(pred_poison, dim=1, keepdim=True)

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


