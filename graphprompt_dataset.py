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

from graphprompt.pre_train_www import PreTrain
from graphprompt.prompt_www import graph_prompt_layer_feature_weighted_sum
from graphprompt.utils_www import mkdir, center_embedding, distance2center, anneal_fn
from graphprompt.model_www import GNN

import attack.attack_www_data as attack_data
import attack.attack_www_utils as attack_utils
import attack.attack_www_pretrain as attack_pretrain
from defense import preprocess


def save_to_csv(record_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Score'])
        for i, score in enumerate(record_list):
            writer.writerow([i, score])


def our_badpretrain(device, num_class, reg_param_gnn, folder_path, num_trigger_node, makeTE, gnn_type, dataname,
                    trigger_pattern, poisoned_node, percent_nodes, reg_param_trigger_n, reg_param_trigger_f):
    mkdir(f'{folder_path}/pre_trained_gnn/')

    num_parts, batch_size = 200, 10

    logger.info("load clean data...")
    graph_list, input_dim, hid_dim = attack_data.load_data4pretrain(dataname)

    logger.info("create PreTrain instance...")
    pt = PreTrain(gnn_type, input_dim, hid_dim, gln=2)

    logger.info("clean pre-training...")
    conloss = pt.train(dataname, graph_list, folder_path, batch_size=batch_size, lr=0.01, decay=0.00001, epochs=400)

    pt.model.eval()

    logger.info('make the target embedding for backdoor...')
    if makeTE == 'cluster_random':
        cluster_centers_list = attack_pretrain.make_target_embedding(graph_list, pt.model, num_class, device)
        target_embedding = random.choice(cluster_centers_list)
        target_embedding = torch.tensor(target_embedding)
        logger.info("target_embedding is {}".format(target_embedding))

    elif makeTE == 'cluster_max':
        cluster_centers_list = attack_pretrain.make_target_embedding(graph_list, pt.model, num_class, device)
        distance_sums = attack_pretrain.calculate_distances(cluster_centers_list)
        farthest_center_index = np.argmax(distance_sums)

        target_embedding = cluster_centers_list[farthest_center_index]
        target_embedding = torch.tensor(target_embedding)
        logger.info("target_embedding is {}".format(target_embedding))

    elif makeTE == 'cluster_min':
        cluster_centers_list = attack_pretrain.make_target_embedding(graph_list, pt.model, num_class, device)
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
    poisoned_pre_model_gnn = copy.deepcopy(pt.model).to(device)
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
                'epoch:{}/{} | trigger total loss:{}, lp loss:{}, ln loss:{}, loss_lf:{}'.format(epoch, inter_epoch,
                                                                                                 total_trigger_loss,
                                                                                                 total_lp,
                                                                                                 total_ln,
                                                                                                 total_lf))
            record_trigger_loss.append(total_trigger_loss)

        for inter_epoch in range(1, 11):
            total_gnn_loss, total_lt, total_ld, target_embedding = attack_pretrain.optimize_backdoor_gnn(
                graph_list=graph_list,
                trigger_node_features=trigger_node_features,
                poisoned_pre_model_gnn=poisoned_pre_model_gnn,
                clean_gnn=pt.model,
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
               "{}/pre_trained_gnn/{}.{}.{}.pth".format(folder_path, dataname, gnn_type, "attack1"))
    logger.info("+++backdoor model saved ! {}.{}.{}.pth".format(dataname, gnn_type, "attack1"))

    trigger_node_features_data = trigger_node_features.clone().detach()

    logger.info("Saved picture with all three losses successfully")

    logger.info("poisoned pretrain end...")

    return target_embedding, trigger_node_features_data


def model_create(is_poison, dataname, gnn_type, folder_path):
    input_dim, hid_dim = 100, 100

    if is_poison == True:
        gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, gnn_type=gnn_type)
        pre_train_path = "{}/pre_trained_gnn/{}.{}.{}.pth".format(folder_path, dataname, gnn_type, "attack1")

        gnn.load_state_dict(torch.load(pre_train_path))
        logger.info("successfully load backdoor pre-trained weights for gnn! @ {}".format(pre_train_path))
    else:
        gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, gnn_type=gnn_type)
        pre_train_path = "{}/pre_trained_gnn/{}.{}.pth".format(folder_path, dataname, gnn_type)

        gnn.load_state_dict(torch.load(pre_train_path))
        logger.info("successfully load clean pre-trained weights for gnn! @ {}".format(pre_train_path))

    for p in gnn.parameters():
        p.requires_grad = False

    PG_aggregator = graph_prompt_layer_feature_weighted_sum(2 * hid_dim)

    opi = torch.optim.AdamW(PG_aggregator.parameters(), lr=0.01, weight_decay=0.00001, amsgrad=True)

    lossfn = lambda pred, target, neg_slp: F.nll_loss(pred, target)

    gnn.to(device)
    PG_aggregator.to(device)

    return gnn, PG_aggregator, opi, lossfn


def prompt_w_h_poison(backdoor_node_features, makeTE, target_embedding, device, num_trigger_node, folder_path,
                      down_dataname="Cora", pre_dataname="CiteSeer", gnn_type="TransformerConv", pre_num_class=6,
                      down_num_class=7, is_poison=True, defense_mode="none", sim_thre=0):
    _, _, train_list, test_list = attack_data.multi_class_NIG_clean(down_dataname, down_num_class, shots=100)
    train_loader = DataLoader(train_list, batch_size=len(train_list), shuffle=True)
    test_loader = DataLoader(test_list, batch_size=len(test_list), shuffle=False)

    poison_train_list, poison_test_list = attack_data.multi_class_NIG_poison(down_dataname, down_num_class, shots=100)

    poison_test_list = attack_pretrain.set_target_feature_optimzed(poison_test_list, backdoor_node_features,
                                                                   num_trigger_node)
    logger.info("trigegr node feature is {}".format(backdoor_node_features))

    if defense_mode == "prune_e":
        preprocess.prune_edges_by_similarity(poison_test_list, sim_thre, device)
    elif defense_mode == "prune_ne":
        preprocess.prune_edges_and_nodes_by_similarity(poison_test_list, sim_thre, device)
    elif defense_mode == "prune_graph":
        preprocess.prune_edges_by_similarity_graph(poison_test_list, sim_thre, device, num_trigger_node)
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

    gnn, PG_aggregator, opi_pg, lossfn = model_create(is_poison=is_poison, dataname=pre_dataname, gnn_type=gnn_type,
                                                      folder_path=folder_path)

    outer_epoch = 500

    main_acc_list = []
    asr_list = []
    drop_list = []
    flip_list = []

    for i in range(1, outer_epoch + 1):
        logger.info("***************************** epoch {}**************************".format(i))
        PG_aggregator.train()
        gnn.eval()
        running_loss = 0.
        epoch_step = len(train_loader)
        total_step = outer_epoch * epoch_step
        for batch_id, train_batch in enumerate(train_loader):
            train_batch = train_batch.to(device)
            graph_label = torch.tensor(train_batch.y, dtype=int).to(device).unsqueeze(1)
            graph_len = torch.bincount(train_batch.batch).to(torch.int32).view(-1, 1).to(device)

            graph_emb, node_emb = gnn.forward(train_batch.x, train_batch.edge_index, graph_len)

            embedding = PG_aggregator.forward(node_emb, graph_len) * 1e3
            c_embedding = center_embedding(embedding, train_batch.y)
            distance = distance2center(embedding, c_embedding)

            distance = 1 / F.normalize(distance, dim=1)

            pred = F.log_softmax(distance, dim=1)

            bp_loss_slp, l0, l1 = "anneal_cosine$1.0$0.01".rsplit("$", 3)
            neg_slp = anneal_fn(bp_loss_slp, batch_id + i * epoch_step, T=total_step // 4, lambda0=float(l0),
                                lambda1=float(l1))

            bp_loss = lossfn(pred.float(), graph_label.squeeze().type(torch.LongTensor).to(device), neg_slp)

            bp_loss.backward()

            opi_pg.step()
            opi_pg.zero_grad()

            running_loss += bp_loss.item()

        logger.info('epoch {}/{} | loss: {:.8f}'.format(i, outer_epoch, running_loss))

        # testing stage
        if i % 5 == 0:
            logger.info("========test the clean accuracy on the downstream model========")
            PG_aggregator.eval()
            gnn.eval()
            main_acc, test_c_embedding = attack_utils.acc_f1_over_batches(test_loader, PG_aggregator, gnn,
                                                                          device=device)
            main_acc_list.append(main_acc)

            # backdoor testing stage
            logger.info("========test the backdoor accuracy on the downstream model========")

            logger.info('the ASR of the target backdoor attack...')

            if makeTE == 'trigger_graph':
                for (batch_id, trigger_batch) in enumerate(trigger_loader):
                    trigger_graph_len = torch.tensor([num_trigger_node]).to(device)
                    graph_emb, node_emb = gnn.forward(trigger_batch.x, trigger_batch.edge_index, trigger_graph_len)
                    target_embedding = PG_aggregator.forward(node_emb, trigger_graph_len) * 1e3

            distance = distance2center(target_embedding, test_c_embedding)
            distance = -1 * F.normalize(distance, dim=1)
            pred = F.log_softmax(distance, dim=1)
            target_label = torch.argmax(pred, dim=1, keepdim=True).item()

            logger.info("target clss is {}".format(target_label))
            poison_test_list_label_change = attack_pretrain.set_target_label(poison_test_list, target_label)
            poison_test_loader1 = DataLoader(poison_test_list_label_change,
                                             batch_size=len(poison_test_list_label_change),
                                             shuffle=False)

            asr = attack_utils.asr_f1_over_batches(poison_test_loader1, PG_aggregator, gnn, test_c_embedding,
                                                   device=device)
            asr_list.append(asr)

    save_to_csv(main_acc_list, f'{folder_path}/bn_main_acc.csv')
    save_to_csv(asr_list, f'{folder_path}/bn_asr.csv')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--predataset", type=str, default="CiteSeer",
                        choices=['Computers', 'PubMed', 'Cora', 'CiteSeer', 'Photo', 'ENZYMES'])
    parser.add_argument("--downdataset", type=str, default="Cora",
                        choices=['Computers', 'PubMed', 'Cora', 'CiteSeer', 'Photo', 'ENZYMES'])
    parser.add_argument("--pretext", type=str, default="simlearn", choices=['simlearn'])
    parser.add_argument("--gnn_type", type=str, default="TransformerConv", choices=['TransformerConv', 'GCN', 'GAT'])
    parser.add_argument("--usage_par", type=str, default="www23")

    # bs1 is Graph Contrastive Backdoor Attacks
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
    parser.add_argument("--reg_param_trigger_n", type=float, default=0.05)
    parser.add_argument("--reg_param_trigger_f", type=float, default=0.05)
    parser.add_argument("--reg_param_gnn", type=float, default=0.1)

    parser.add_argument("--defense_mode", type=str, default="none", choices=['none', 'prune_graph'])
    parser.add_argument("--sim_thre", type=float, default=0)

    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    folder_path = f'./results/dataset/{args.predataset}_{args.downdataset}_{current_time}_{args.gnn_type}_{args.usage_par}_{args.defense_mode}_{args.sim_thre}_{args.attack_type}_{args.makeTE}_{args.poisoned_node}_{args.reg_param_trigger_n}_{args.reg_param_trigger_f}'
    folder_paths = f'./results/dataset/{args.predataset}_{args.downdataset}_{current_time}_{args.gnn_type}_{args.usage_par}_{args.defense_mode}_{args.sim_thre}_{args.attack_type}_{args.makeTE}_{args.poisoned_node}_{args.reg_param_trigger_n}_{args.reg_param_trigger_f}/pic_result'
    os.mkdir(folder_path)
    os.mkdir(folder_paths)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger("logger")
    logger.addHandler(logging.FileHandler(filename=f'{folder_path}/log.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    logger.info('Pretrain Dataset:{}'.format(args.predataset))
    logger.info('Downstream Dataset:{}'.format(args.downdataset))
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

    if args.predataset == "CiteSeer":
        pre_num_class = 6
    elif args.predataset == "Cora":
        pre_num_class = 7
    elif args.predataset == "Photo":
        pre_num_class = 8
    elif args.predataset == "Computers":
        pre_num_class = 10
    else:
        raise ValueError

    if args.downdataset == "Cora":
        down_num_class = 7

    elif args.downdataset == "CiteSeer":
        down_num_class = 6

    elif args.downdataset == "Computers":
        down_num_class = 10

    elif args.downdataset == "Photo":
        down_num_class = 8

    elif args.downdataset == "PubMed":
        down_num_class = 3

    else:
        raise ValueError

    if args.attack_type == 'ours':
        target_embeddings, backdoor_node_feature = our_badpretrain(device=device,
                                                                   num_class=pre_num_class,
                                                                   reg_param_trigger_n=args.reg_param_trigger_n,
                                                                   reg_param_trigger_f=args.reg_param_trigger_f,
                                                                   reg_param_gnn=args.reg_param_gnn,
                                                                   folder_path=folder_path,
                                                                   num_trigger_node=args.num_trigger_node,
                                                                   makeTE=args.makeTE,
                                                                   gnn_type=args.gnn_type,
                                                                   dataname=args.predataset,
                                                                   trigger_pattern=args.trigger_pattern,
                                                                   poisoned_node=args.poisoned_node,
                                                                   percent_nodes=args.percent_nodes)

    prompt_w_h_poison(backdoor_node_features=backdoor_node_feature, makeTE=args.makeTE,
                      target_embedding=target_embeddings, device=device, num_trigger_node=args.num_trigger_node,
                      down_dataname=args.downdataset, pre_dataname=args.predataset, gnn_type=args.gnn_type,
                      down_num_class=down_num_class, pre_num_class=pre_num_class, is_poison=True,
                      defense_mode=args.defense_mode, sim_thre=args.sim_thre, folder_path=folder_path)
