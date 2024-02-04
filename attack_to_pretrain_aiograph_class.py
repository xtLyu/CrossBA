import argparse
import datetime
import os
import random
import numpy as np
import torch
import copy
import csv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch import nn, optim
import logging
import pickle as pk
import torch.nn.init as init
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.nn.functional as F

plt.style.use('seaborn')

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

from ProG import PreTrain
from ProG.utils import mkdir
from ProG.meta import MAML
from ProG.prompt import GNN, LightPrompt, HeavyPrompt, FrontAndHead
from attack.attack_data import clean_pretrain_class_dataset, multi_class_NIG_clean_aiolevel_class, \
    multi_class_NIG_poison_aiolevel_class, induced_graphs_aio_graphs_poison_class, load_tasks_aiolevel_class, \
    load_backdoor_task_aiolevel_class
from attack.attack_utils import acc_f1_over_batches, asr_f1_over_batches, drop_rate_over_batches, flip_rate_over_batches
import attack.attack_pretrain as attack_pretrain
from defense.preprocess import prune_edges_by_similarity, prune_edges_and_nodes_by_similarity, \
    prune_edges_by_similarity_graph


def save_to_csv(record_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Score'])
        for i, score in enumerate(record_list):
            writer.writerow([i, score])


def data_process(dataname, num_trigger_node, pre_class_list, down_class_list, trigger_pattern):
    data = pk.load(open('./Dataset/class/{}/feature_reduced_down.data'.format(dataname), 'br'))
    induced_graphs_aio_graphs_poison_class(data, dataname=dataname, down_class_list=down_class_list, smallest_size=100,
                                           largest_size=300, num_trigger_nodes=num_trigger_node,
                                           trigger_pattern=trigger_pattern)


def our_badpretrain(device, num_class, reg_param_gnn, folder_path, num_trigger_node, makeTE, pretext,
                    gnn_type, dataname, trigger_pattern, poisoned_node, percent_nodes, reg_param_trigger_n,
                    reg_param_trigger_f, pretrain_class_list):
    mkdir(f'{folder_path}/pre_trained_gnn/')

    num_parts, batch_size, shots = 200, 10, 50

    logger.info("load clean data on classes:{}...".format(pretrain_class_list))
    graph_list, input_dim, hid_dim = clean_pretrain_class_dataset(dataname, pretrain_class_list, shots)

    logger.info("create PreTrain instance...")
    pt = PreTrain(pretext, gnn_type, input_dim, hid_dim, gln=2)

    logger.info("clean pre-training...")
    conloss = pt.train(dataname, graph_list, folder_path, batch_size=batch_size, aug1='dropN', aug2="permE",
                       aug_ratio=None, lr=0.001, decay=0.0001, epochs=400)

    pt.model.gnn.eval()

    logger.info('make the target embedding for backdoor...')
    if makeTE == 'cluster_random':
        cluster_centers_list = attack_pretrain.make_target_embedding(graph_list, pt.model.gnn, len(pretrain_class_list),
                                                                     device)
        target_embedding = random.choice(cluster_centers_list)
        target_embedding = torch.tensor(target_embedding)
        logger.info("target_embedding is {}".format(target_embedding))

    elif makeTE == 'cluster_max':
        cluster_centers_list = attack_pretrain.make_target_embedding(graph_list, pt.model.gnn, len(pretrain_class_list),
                                                                     device)
        distance_sums = attack_pretrain.calculate_distances(cluster_centers_list)
        farthest_center_index = np.argmax(distance_sums)

        target_embedding = cluster_centers_list[farthest_center_index]
        target_embedding = torch.tensor(target_embedding)
        logger.info("target_embedding is {}".format(target_embedding))

    elif makeTE == 'cluster_min':
        cluster_centers_list = attack_pretrain.make_target_embedding(graph_list, pt.model.gnn, len(pretrain_class_list),
                                                                     device)
        distance_sums = attack_pretrain.calculate_distances(cluster_centers_list)
        closest_center_index = np.argmin(distance_sums)

        target_embedding = cluster_centers_list[closest_center_index]
        target_embedding = torch.tensor(target_embedding)
        logger.info("target_embedding is {}".format(target_embedding))

    elif makeTE == "trigger_graph":
        target_embedding = None
        logger.info("target_embedding is the output embedding of the trigger graph")
    else:
        raise ValueError

    logger.info("backdoor pre-training...")
    poisoned_pre_model_gnn = copy.deepcopy(pt.model.gnn).to(device)
    poisoned_pre_model_gnn.train()
    trigger_node_features = torch.nn.Parameter(init.kaiming_normal_(torch.empty(num_trigger_node, 100, device=device)))
    optimizer_gnn = optim.Adam(poisoned_pre_model_gnn.parameters(), lr=0.0001)
    optimizer_trigger = optim.Adam([trigger_node_features], lr=0.01)

    record_gnn_loss = []
    record_trigger_loss = []

    for epoch in range(1, 10 + 1):
        for inter_epoch in range(1, 15 + 1):
            total_trigger_loss, total_lp, total_ln, total_lf = attack_pretrain.optimize_trigger(graph_list=graph_list,
                                                                                                trigger_node_features=trigger_node_features,
                                                                                                poisoned_pre_model_gnn=poisoned_pre_model_gnn,
                                                                                                optimizer_trigger=optimizer_trigger,
                                                                                                device=device,
                                                                                                poisoned_node=poisoned_node,
                                                                                                trigger_pattern=trigger_pattern,
                                                                                                num_trigger_node=num_trigger_node,
                                                                                                percent_nodes=percent_nodes,
                                                                                                makeTE=makeTE,
                                                                                                target_embedding=target_embedding,
                                                                                                reg_param_1=reg_param_trigger_n,
                                                                                                reg_param_2=reg_param_trigger_f)
            logger.info(
                'epoch:{}/{} | trigger total loss:{}, lp loss:{}, ln loss:{}, lf loss:{}'.format(epoch, inter_epoch,
                                                                                                 total_trigger_loss,
                                                                                                 total_lp,
                                                                                                 total_ln,
                                                                                                 total_lf))
            record_trigger_loss.append(total_trigger_loss)

        for inter_epoch in range(1, 2):
            total_gnn_loss, total_lt, total_ld, target_embedding = attack_pretrain.optimize_backdoor_gnn(
                graph_list=graph_list,
                trigger_node_features=trigger_node_features,
                poisoned_pre_model_gnn=poisoned_pre_model_gnn,
                clean_gnn=pt.model.gnn,
                optimizer_gnn=optimizer_gnn,
                device=device,
                poisoned_node=poisoned_node,
                trigger_pattern=trigger_pattern,
                num_trigger_node=num_trigger_node,
                percent_nodes=percent_nodes,
                reg_param=reg_param_gnn,
                target_embedding=target_embedding,
                makeTE=makeTE)
            logger.info(
                'epoch:{}/{} | enconder total loss:{}, lt loss:{}, ld loss:{}'.format(epoch, inter_epoch,
                                                                                      total_gnn_loss, total_lt,
                                                                                      total_ld))
            record_gnn_loss.append(total_gnn_loss)

    torch.save(poisoned_pre_model_gnn.state_dict(),
               "{}/pre_trained_gnn/{}.{}.{}.{}.pth".format(folder_path, dataname, pretext, gnn_type, "attack1"))
    logger.info("+++backdoor model saved ! {}.{}.{}.{}.pth".format(dataname, pretext, gnn_type, "attack1"))

    trigger_node_features_data = trigger_node_features.clone().detach()

    logger.info("poisoned pretrain end...")

    return target_embedding, trigger_node_features_data


def model_create(is_poison, dataname, gnn_type, num_class, folder_path, pretext, task_type='multi_class_classification',
                 tune_answer=False):
    if task_type in ['multi_class_classification', 'regression']:
        input_dim, hid_dim = 100, 100
        if gnn_type != 'TransformerConv':
            lr, wd = 0.001, 0.00001
        else:
            lr, wd = 0.0001, 0.00001
        logger.info('pg le:{}'.format(lr))

        tnpc = 100  # token number per class

        # load pre-trained GNN
        if is_poison == True:
            gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, gnn_type=gnn_type)
            pre_train_path = "{}/pre_trained_gnn/{}.{}.{}.{}.pth".format(folder_path, dataname, pretext, gnn_type,
                                                                         "attack1")
            gnn.load_state_dict(torch.load(pre_train_path))
            logger.info("successfully load backdoor pre-trained weights for gnn! @ {}".format(pre_train_path))
        else:
            gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, gnn_type=gnn_type)
            pre_train_path = "{}/pre_trained_gnn/{}.{}.{}.pth".format(folder_path, dataname, pretext, gnn_type)
            gnn.load_state_dict(torch.load(pre_train_path))
            logger.info("successfully load clean pre-trained weights for gnn! @ {}".format(pre_train_path))

        for p in gnn.parameters():
            p.requires_grad = False

        if tune_answer:
            PG = HeavyPrompt(token_dim=input_dim, token_num=15, cross_prune=0.1, inner_prune=0.3)
        else:
            PG = LightPrompt(token_dim=input_dim, token_num_per_group=tnpc, group_num=num_class, inner_prune=0.01)

        opi = optim.Adam(filter(lambda p: p.requires_grad, PG.parameters()), lr=lr, weight_decay=wd)

        if task_type == 'regression':
            lossfn = nn.MSELoss(reduction='mean')
        else:
            lossfn = nn.CrossEntropyLoss(reduction='mean')

        if tune_answer:
            if task_type == 'regression':
                answering = torch.nn.Sequential(torch.nn.Linear(hid_dim, num_class), torch.nn.Sigmoid())
            else:
                answering = torch.nn.Sequential(torch.nn.Linear(hid_dim, num_class), torch.nn.Softmax(dim=1))

            opi_answer = optim.Adam(filter(lambda p: p.requires_grad, answering.parameters()), lr=0.01,
                                    weight_decay=0.00001)
        else:
            answering, opi_answer = None, None

        gnn.to(device)
        PG.to(device)

        return gnn, PG, opi, lossfn, answering, opi_answer

    else:
        raise ValueError("model_create function hasn't supported {} task".format(task_type))


def train_one_outer_epoch(epoch, train_loader, opi, lossfn, gnn, PG, answering, device):
    for j in range(1, epoch + 1):
        running_loss = 0.
        for batch_id, train_batch in enumerate(train_loader):
            train_batch = train_batch.to(device)
            prompted_graph = PG(train_batch)
            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            pre = answering(graph_emb)
            train_loss = lossfn(pre, train_batch.y)

            opi.zero_grad()
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()

        logger.info('epoch {}/{} | loss: {:.8f}'.format(j, epoch, running_loss))


def prompt_w_h_poison(backdoor_node_features, makeTE, target_embedding, device, num_trigger_node, pretext, folder_path,
                      down_class_list, dataname="CiteSeer", gnn_type="TransformerConv", num_class=6, is_poison=True,
                      task_type='multi_class_classification', defense_mode='none', sim_thre=0):
    logger.info("load data on classes:{}...".format(down_class_list))
    _, _, train_list, test_list = multi_class_NIG_clean_aiolevel_class(dataname, down_class_list, shots=200)
    train_loader = DataLoader(train_list, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_list, batch_size=10, shuffle=False)

    poison_train_list, poison_test_list = multi_class_NIG_poison_aiolevel_class(dataname, down_class_list, shots=200)

    poison_test_list = attack_pretrain.set_target_feature_optimzed(poison_test_list, backdoor_node_features,
                                                                   num_trigger_node)

    logger.info("trigegr node feature is {}".format(backdoor_node_features))

    if defense_mode == "prune_e":
        prune_edges_by_similarity(poison_test_list, sim_thre, device)
    elif defense_mode == "prune_ne":
        prune_edges_and_nodes_by_similarity(poison_test_list, sim_thre, device)
    elif defense_mode == "prune_graph":
        prune_edges_by_similarity_graph(poison_test_list, sim_thre, device, num_trigger_node)
    else:
        logger.info("no defense")

    target_embedding = torch.tensor(target_embedding).to(device)
    if makeTE != 'trigger_graph':
        target_embedding = target_embedding.unsqueeze(0)
    else:
        edge_index = []
        for i in range(num_trigger_node):
            for j in range(i + 1, num_trigger_node):
                edge_index.append([i, j])
                edge_index.append([j, i])

        trigger_edge_index = torch.tensor(edge_index).t().contiguous().to(device)
        trigger_x = backdoor_node_features.to(device)
        trigger_graph = Data(x=trigger_x, edge_index=trigger_edge_index, y=1)
        trigger_graph_list = []
        trigger_graph_list.append(trigger_graph)
        trigger_loader = DataLoader(trigger_graph_list, batch_size=1)

    gnn, PG, opi_pg, lossfn, answering, opi_answer = model_create(is_poison, dataname, gnn_type, len(down_class_list),
                                                                  folder_path, pretext, task_type, True)
    answering.to(device)

    outer_epoch = 30
    if gnn_type != 'TransformerConv':
        answer_epoch = 50
        prompt_epoch = 50
    else:
        answer_epoch = 1
        prompt_epoch = 1

    main_acc_list = []
    asr_list = []
    drop_list = []
    flip_list = []

    # training stage
    for i in range(1, outer_epoch + 1):
        logger.info("***************************** eppch {}**************************".format(i))
        logger.info(("{}/{} frozen gnn | frozen prompt | *tune answering function...".format(i, outer_epoch)))
        # tune task head
        answering.train()
        PG.eval()
        gnn.eval()
        train_one_outer_epoch(answer_epoch, train_loader, opi_answer, lossfn, gnn, PG, answering, device)

        logger.info("{}/{}  frozen gnn | *tune prompt |frozen answering function...".format(i, outer_epoch))
        # tune prompt
        answering.eval()
        PG.train()
        gnn.eval()
        train_one_outer_epoch(prompt_epoch, train_loader, opi_pg, lossfn, gnn, PG, answering, device)

        # testing stage
        logger.info("========test the clean accuracy on the downstream model========")
        answering.eval()
        PG.eval()
        gnn.eval()
        main_acc = acc_f1_over_batches(test_loader, PG, gnn, answering, len(down_class_list), task_type, device=device)
        main_acc_list.append(main_acc)

        # backdoor testing stage
        logger.info("========test the backdoor accuracy on the downstream model========")

        logger.info('the ASR of the target backdoor attack...')

        if makeTE == 'trigger_graph':
            for (batch_id, trigger_batch) in enumerate(trigger_loader):
                prompted_trigger_graph = PG.forward(trigger_batch)
                target_embedding = gnn.forward(prompted_trigger_graph.x, prompted_trigger_graph.edge_index,
                                               prompted_trigger_graph.batch).detach()

        pre = answering(target_embedding).detach()
        target_label = torch.argmax(pre, dim=1).item()
        poison_test_list_label_change = attack_pretrain.set_target_label(poison_test_list, target_label)
        poison_test_loader1 = DataLoader(poison_test_list_label_change, batch_size=10, shuffle=False)
        answering.eval()
        PG.eval()
        gnn.eval()
        asr = asr_f1_over_batches(poison_test_loader1, PG, gnn, answering, len(down_class_list), task_type,
                                  device=device)
        asr_list.append(asr)

    save_to_csv(main_acc_list, f'{folder_path}/bn_main_acc.csv')
    save_to_csv(asr_list, f'{folder_path}/bn_asr.csv')

    plt.figure(figsize=(8, 6))
    x_coords = list(range(1, outer_epoch + 1))
    plt.plot(x_coords, main_acc_list, label='Accuracy', color='blue', linestyle='-', marker='o')
    plt.plot(x_coords, asr_list, label='Target ASR', color='red', linestyle='--', marker='x')
    plt.plot(x_coords, drop_list, label='Untarget Drop', color='green', linestyle='-.', marker='s')
    plt.plot(x_coords, flip_list, label='Untarget Flip', color='purple', linestyle=':', marker='^')

    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title('Performance Metrics Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')

    plt.grid(True, linestyle='--', alpha=0.5)

    plt.legend(loc='best', fontsize='medium')

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.savefig(f'{folder_path}/pic_result/poisoned_downstream_model.png')
    logger.info("save picture success")


def model_components(device, is_poison, folder_path, pretext, dataname="CiteSeer", gnn_type="TransformerConv"):
    adapt_lr = 0.01
    if gnn_type == 'TransformerConv':
        meta_lr = 0.001
    else:
        meta_lr = 0.01
    logger.info('adaolt lr:{}, meta_lr:{}'.format(adapt_lr, meta_lr))

    model = FrontAndHead(input_dim=100, hid_dim=100, num_classes=2, task_type="multi_label_classification",
                         token_num=15, cross_prune=0.1, inner_prune=0.3)

    model.to(device)

    # load pre-trained GNN
    if is_poison == True:
        gnn = GNN(input_dim=100, hid_dim=100, out_dim=100, gcn_layer_num=2, gnn_type=gnn_type)
        pre_train_path = "{}/pre_trained_gnn/{}.{}.{}.{}.pth".format(folder_path, dataname, pretext, gnn_type,
                                                                     "attack1")
        gnn.load_state_dict(torch.load(pre_train_path))
        logger.info("successfully load backdoor pre-trained weights for gnn! @ {}".format(pre_train_path))
    else:
        gnn = GNN(input_dim=100, hid_dim=100, out_dim=100, gcn_layer_num=2, gnn_type=gnn_type)
        pre_train_path = "{}/pre_trained_gnn/{}.{}.{}.pth".format(folder_path, dataname, pretext, gnn_type)
        gnn.load_state_dict(torch.load(pre_train_path))
        logger.info("successfully load clean pre-trained weights for gnn! @ {}".format(pre_train_path))

    gnn.to(device)

    for p in gnn.parameters():
        p.requires_grad = False

    maml = MAML(model, lr=adapt_lr, first_order=False, allow_nograd=True)

    opt = optim.Adam(filter(lambda p: p.requires_grad, maml.parameters()), meta_lr)

    lossfn = nn.CrossEntropyLoss(reduction='mean')

    return maml, gnn, opt, lossfn


def meta_train_maml(epoch, maml, gnn, lossfn, opt, meta_train_task_id_list, dataname, adapt_steps=2, K_shot=100):
    if len(meta_train_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at least two tasks!")

    random.shuffle(meta_train_task_id_list)

    task_pairs = [(meta_train_task_id_list[i], meta_train_task_id_list[i + 1]) for i in
                  range(0, len(meta_train_task_id_list) - 1, 2)]

    # meta-training
    for ep in range(epoch):
        task_gradients = []
        for task_1, task_2, support, query, total_num in load_tasks_aiolevel_class('train', task_pairs, dataname,
                                                                                   K_shot, seed):
            learner = copy.deepcopy(maml).to(device)
            support_loader = DataLoader(support.to_data_list(), batch_size=10, shuffle=False)
            query_loader = DataLoader(query.to_data_list(), batch_size=10, shuffle=False)
            support_opt = optim.Adam(filter(lambda p: p.requires_grad, maml.parameters()), 0.001)

            for i in range(adapt_steps):  # adaptation_steps
                running_loss = 0.
                for batch_id, support_batch in enumerate(support_loader):
                    support_batch.to(device)
                    support_opt.zero_grad()
                    support_batch_preds = learner(support_batch, gnn)
                    support_batch_loss = lossfn(support_batch_preds, support_batch.y)
                    running_loss += support_batch_loss.item()

                    support_batch_loss.backward()
                    support_opt.step()

                logger.info('adapt {}/{} | loss: {:.8f}'.format(i + 1, adapt_steps, running_loss))

            running_loss = 0.
            query_gradients = []
            for batch_id, query_batch in enumerate(query_loader):
                query_batch.to(device)
                query_batch_preds = learner(query_batch, gnn)
                query_batch_loss = lossfn(query_batch_preds, query_batch.y)
                running_loss += query_batch_loss.item()
                gradients = torch.autograd.grad(query_batch_loss, learner.parameters(), retain_graph=True)
                query_gradients.append(gradients)

            logger.info('query loss | loss: {:.8f}'.format(running_loss))

            task_gradients.append([torch.mean(torch.stack(grad_pair), dim=0) for grad_pair in zip(*query_gradients)])

        meta_gradient = [torch.mean(torch.stack(grad_pair), dim=0) for grad_pair in zip(*task_gradients)]
        for param, grad in zip(maml.parameters(), meta_gradient):
            if grad is not None:
                if param.grad is None:
                    param.grad = grad.clone().type_as(param.data)
                else:
                    param.grad.data.add_(grad.type_as(param.data))

        opt.zero_grad()
        opt.step()

    logger.info("meta training end !")


def meta_test_adam(is_poison, meta_test_task_id_list, dataname, K_shot, seed, maml, gnn, adapt_steps_meta_test, lossfn,
                   backdoor_node_features, makeTE, target_embedding, num_trigger_node, defense_mode, sim_thre,
                   task_type='multi_class_classification'):
    # meta-testing
    if len(meta_test_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at leat two tasks!")

    random.shuffle(meta_test_task_id_list)

    task_pairs = [(meta_test_task_id_list[i], meta_test_task_id_list[i + 1]) for i in
                  range(0, len(meta_test_task_id_list) - 1, 2)]

    for task_1, task_2, support, query, _ in load_tasks_aiolevel_class('test', task_pairs, dataname, K_shot, seed):
        test_model = copy.deepcopy(maml.module).to(device)
        test_opi = optim.Adam(filter(lambda p: p.requires_grad, test_model.parameters()), lr=0.001,
                              weight_decay=0.00001)

        test_model.train()

        support_loader = DataLoader(support.to_data_list(), batch_size=10, shuffle=True)
        query_loader = DataLoader(query.to_data_list(), batch_size=10, shuffle=False)

        task_1_b, task_2_b, query_b, _ = next(
            load_backdoor_task_aiolevel_class('test', task_pairs, dataname, K_shot, seed))
        query_b = query_b.to_data_list()
        query_b = attack_pretrain.set_target_feature_optimzed(query_b, backdoor_node_features, num_trigger_node)

        if defense_mode == "prune_e":
            prune_edges_by_similarity(query_b, sim_thre, device)
        elif defense_mode == "prune_ne":
            prune_edges_and_nodes_by_similarity(query_b, sim_thre, device)
        elif defense_mode == "prune_graph":
            prune_edges_by_similarity_graph(query_b, sim_thre, device, num_trigger_node)
        else:
            logger.info("no defense")

        target_embedding = torch.tensor(target_embedding).to(device)
        logger.info("trigegr node feature is {}".format(backdoor_node_features))

        if makeTE != 'trigger_graph':
            target_embedding = target_embedding.unsqueeze(0)
        else:
            edge_index = []
            for i in range(num_trigger_node):
                for j in range(i + 1, num_trigger_node):
                    edge_index.append([i, j])
                    edge_index.append([j, i])

            trigger_edge_index = torch.tensor(edge_index).t().contiguous().to(device)
            trigger_x = backdoor_node_features.to(device)
            trigger_graph = Data(x=trigger_x, edge_index=trigger_edge_index, y=1)
            trigger_graph_list = []
            trigger_graph_list.append(trigger_graph)
            trigger_loader = DataLoader(trigger_graph_list, batch_size=1)

        for _ in range(adapt_steps_meta_test):
            running_loss = 0.
            for batch_id, support_batch in enumerate(support_loader):
                support_batch.to(device)
                support_preds = test_model(support_batch, gnn)
                support_loss = lossfn(support_preds, support_batch.y)
                test_opi.zero_grad()
                support_loss.backward()
                test_opi.step()
                running_loss += support_loss.item()

                if batch_id == len(support_loader) - 1:
                    last_loss = running_loss / len(support_loader)
                    logger.info('{}/{} training loss: {:.8f}'.format(_, adapt_steps_meta_test, last_loss))
                    running_loss = 0.

        test_model.eval()
        logger.info("=========test the clean accuracy on the downstream model========")
        main_acc = acc_f1_over_batches(query_loader, test_model.PG, gnn, test_model.answering, 2, task_type,
                                       device=device)

        logger.info("========test the backdoor accuracy on the downstream model========")

        logger.info('the ASR of the target backdoor attack...')
        if makeTE == 'trigger_graph':
            for (batch_id, trigger_batch) in enumerate(trigger_loader):
                prompted_trigger_graph = test_model.PG.forward(trigger_batch)
                target_embedding = gnn.forward(prompted_trigger_graph.x, prompted_trigger_graph.edge_index,
                                               prompted_trigger_graph.batch).detach()
        pre = test_model.answering(target_embedding).detach()
        target_label = torch.argmax(pre, dim=1).item()
        print("target class is :{}".format(target_label))
        poison_test_list_label_change = attack_pretrain.set_target_label(query_b, target_label)
        poison_test_loader1 = DataLoader(poison_test_list_label_change, batch_size=10, shuffle=False)
        asr = asr_f1_over_batches(poison_test_loader1, test_model.PG, gnn, test_model.answering, 2, task_type,
                                  device=device)

        drop_rate = ''
        flip_rate = ''
        values = [main_acc, asr, drop_rate, flip_rate]
        labels = ['ACC', 'ASR', 'DROP', 'FLIP']
        plt.figure(figsize=(8, 6))
        colors = ['blue', 'red', 'green', 'orange']

        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, values, color=colors, edgecolor='black')

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

        plt.title('Performance Metrics on meta test data')
        plt.ylabel('Value')
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

        plt.savefig(f'{folder_path}/pic_result/{is_poison}_downstream_meta_model.png')
        logger.info("save picture success")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CiteSeer",
                        choices=['Computers', 'PubMed', 'Cora', 'CiteSeer', 'Photo', 'ENZYMES'])
    parser.add_argument("--pretext", type=str, default="GraphCL", choices=['GraphCL', 'SimGRACE'])
    parser.add_argument("--gnn_type", type=str, default="TransformerConv", choices=['TransformerConv', 'GCN', 'GAT'])
    parser.add_argument("--usage_par", type=str, default="prompt", choices=['prompt', 'tune_head', 'prompt_meta'])

    parser.add_argument("--attack_type", type=str, default="ours", choices=['ours', 'bs1'])
    parser.add_argument("--is_poison", type=int, default=1)
    parser.add_argument("--num_trigger_node", type=int, default=3)
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--makeTE", type=str, default='trigger_graph',
                        choices=['cluster_random', 'cluster_max', 'cluster_min', 'trigger_graph'])
    parser.add_argument("--trigger_pattern", type=str, default='trigger_graph',
                        choices=['multi_nodes', 'trigger_graph'])
    parser.add_argument("--poisoned_node", type=str, default='degree_min',
                        choices=['random', 'degree_min', 'degree_max'])
    parser.add_argument("--percent_nodes", type=float, default=0, help='pretrain')
    parser.add_argument("--reg_param_trigger_n", type=float, default=0.1)
    parser.add_argument("--reg_param_trigger_f", type=float, default=0.1)
    parser.add_argument("--reg_param_gnn", type=float, default=0.5)

    parser.add_argument("--defense_mode", type=str, default="none",
                        choices=['none', 'prune_e', 'prune_ne', 'prune_graph'])
    parser.add_argument("--sim_thre", type=float, default=0)

    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    folder_path = f'./results/aiolevel_class/{args.dataset}_{current_time}_{args.gnn_type}_{args.usage_par}_{args.defense_mode}_{args.sim_thre}_{args.attack_type}_{args.makeTE}_{args.poisoned_node}_{args.reg_param_trigger_n}_{args.reg_param_trigger_f}_{args.reg_param_gnn}'
    folder_paths = f'./results/aiolevel_class/{args.dataset}_{current_time}_{args.gnn_type}_{args.usage_par}_{args.defense_mode}_{args.sim_thre}_{args.attack_type}_{args.makeTE}_{args.poisoned_node}_{args.reg_param_trigger_n}_{args.reg_param_trigger_f}_{args.reg_param_gnn}/pic_result'
    os.mkdir(folder_path)
    os.mkdir(folder_paths)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger("logger")
    logger.addHandler(logging.FileHandler(filename=f'{folder_path}/log.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    logger.info('Dataset:{}'.format(args.dataset))
    logger.info('pretext:{}'.format(args.pretext))
    logger.info('gnn_type:{}'.format(args.gnn_type))
    logger.info('usage_par:{}'.format(args.usage_par))
    logger.info('the attack method:{}'.format(args.attack_type))
    logger.info('target embedding:{}'.format(args.makeTE))
    logger.info('number of trigger nodes:{}'.format(args.num_trigger_node))
    logger.info('trigger_pattern:{}'.format(args.trigger_pattern))
    logger.info('poison_rate:{}'.format(args.poison_rate))
    logger.info('reg_param_trigger_n: {}'.format(args.reg_param_trigger_n))
    logger.info('reg_param_trigger_f: {}'.format(args.reg_param_trigger_f))
    logger.info('reg_param_gnn: {}'.format(args.reg_param_gnn))
    logger.info('the selection of the poisoned node: {}'.format(args.poisoned_node))
    logger.info('defense_mode: {}'.format(args.defense_mode))
    logger.info('sim_thre: {}'.format(args.sim_thre))

    if args.dataset == "CiteSeer":
        num_class = 6
        pretrain_class_list = [0, 1]
        down_class_list = [2, 3, 4, 5]

        down_meta_train_task_id_list = [2, 3]
        down_meta_test_task_id_list = [4, 5]

    elif args.dataset == "Cora":
        num_class = 7
        pretrain_class_list = [0, 1, 2]
        down_class_list = [3, 4, 5, 6]

        down_meta_train_task_id_list = [5, 6]
        down_meta_test_task_id_list = [3, 4]

    elif args.dataset == "Computers":
        num_class = 10
        pretrain_class_list = [0, 1, 2, 3]
        down_class_list = [4, 5, 6, 7, 8, 9]

        down_meta_train_task_id_list = [4, 5, 6, 7]
        down_meta_test_task_id_list = [8, 9]

    elif args.dataset == "Photo":
        num_class = 8
        pretrain_class_list = [0, 1, 2, 3]
        down_class_list = [4, 5, 6, 7]

        down_meta_train_task_id_list = [6, 7]
        down_meta_test_task_id_list = [4, 5]

    else:
        raise ValueError

    if args.attack_type == 'ours':
        target_embeddings, backdoor_node_feature = our_badpretrain(device=device,
                                                                   num_class=num_class,
                                                                   reg_param_trigger_n=args.reg_param_trigger_n,
                                                                   reg_param_trigger_f=args.reg_param_trigger_f,
                                                                   reg_param_gnn=args.reg_param_gnn,
                                                                   folder_path=folder_path,
                                                                   num_trigger_node=args.num_trigger_node,
                                                                   makeTE=args.makeTE,
                                                                   pretext=args.pretext,
                                                                   gnn_type=args.gnn_type,
                                                                   dataname=args.dataset,
                                                                   trigger_pattern=args.trigger_pattern,
                                                                   poisoned_node=args.poisoned_node,
                                                                   percent_nodes=args.percent_nodes,
                                                                   pretrain_class_list=pretrain_class_list)

    if args.usage_par == 'prompt':
        prompt_w_h_poison(backdoor_node_features=backdoor_node_feature, makeTE=args.makeTE,
                          target_embedding=target_embeddings, device=device, pretext=args.pretext,
                          num_trigger_node=args.num_trigger_node, dataname=args.dataset, gnn_type=args.gnn_type,
                          num_class=num_class, task_type='multi_class_classification', is_poison=True,
                          down_class_list=down_class_list, defense_mode=args.defense_mode, sim_thre=args.sim_thre,
                          folder_path=folder_path)

    elif args.usage_par == 'prompt_meta':

        backdoor_maml, backdoor_gnn, backdoor_opt, backdoor_lossfn = model_components(is_poison=True,
                                                                                      folder_path=folder_path,
                                                                                      pretext=args.pretext,
                                                                                      device=device,
                                                                                      dataname=args.dataset,
                                                                                      gnn_type=args.gnn_type)
        meta_train_maml(30, backdoor_maml, backdoor_gnn, backdoor_lossfn, backdoor_opt, down_meta_train_task_id_list,
                        args.dataset, adapt_steps=20, K_shot=200)
        meta_test_adam(is_poison=True, meta_test_task_id_list=down_meta_test_task_id_list, dataname=args.dataset,
                       K_shot=200, seed=seed, maml=backdoor_maml, gnn=backdoor_gnn, adapt_steps_meta_test=20,
                       lossfn=backdoor_lossfn, backdoor_node_features=backdoor_node_feature, makeTE=args.makeTE,
                       target_embedding=target_embeddings, num_trigger_node=args.num_trigger_node,
                       task_type='multi_class_classification', defense_mode=args.defense_mode, sim_thre=args.sim_thre)

    else:
        raise ValueError
