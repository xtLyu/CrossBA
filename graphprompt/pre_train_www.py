import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import random
import os
import numpy as np
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

from .model_www import GNN
from .utils_www import split_and_batchify_graph_feats, findsample, compareloss


class PreTrain(torch.nn.Module):
    def __init__(self, gnn_type='TransformerConv', input_dim=None, hid_dim=None, gln=2):
        super(PreTrain, self).__init__()
        self.gnn_type = gnn_type
        self.model = GNN(input_dim=input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=gln, pool=None,
                         gnn_type=gnn_type)

        self.device = torch.device("cuda")

    def train_graph_sim(self, model, loader, optimizer, sample):
        train_loss_accum = 0
        total_step = 0
        for step, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(self.device)
            graph_len = torch.bincount(data.batch).to(torch.int32).view(-1, 1).to(self.device)

            x, pred = model.forward(data.x, data.edge_index, graph_len)


            pred = F.sigmoid(pred)
            num_nodes = data.num_nodes
            self_weight = 1.0
            self_edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0).to(self.device)
            self_edge_weight = torch.full((num_nodes,), self_weight).to(self.device)

            edge_index = data.edge_index
            edge_weight = torch.ones(edge_index.size(1)).to(self.device)

            adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))

            self_adj = torch.sparse_coo_tensor(self_edge_index, self_edge_weight, (num_nodes, num_nodes))

            adj_with_self_loops = adj + self_adj

            pred = torch.matmul(adj_with_self_loops, pred)

            _pred = split_and_batchify_graph_feats(pred, graph_len)[0]
            _sample = split_and_batchify_graph_feats(sample, graph_len)[0]
            sample_ = _sample.reshape(_sample.size(0), -1, 1).to(self.device)
            _pred = torch.gather(input=_pred, dim=1, index=sample_)
            _pred = _pred.resize_as(_sample)

            reg_loss = compareloss(_pred, 0.2)
            reg_loss.requires_grad_(True)
            reg_loss.backward()

            optimizer.step()
            train_loss_accum += float(reg_loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def train(self, dataname, graph_list, folder_path, batch_size=10, lr=0.01, decay=0.00001, epochs=100):

        loader = DataLoader(graph_list, batch_size=len(graph_list), shuffle=False, num_workers=1)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=decay, amsgrad=True)


        for step, data in enumerate(loader):
            sample = findsample(data)

        self.model.to(self.device)

        train_loss_min = 1000000
        record_loss = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = self.train_graph_sim(self.model, loader, optimizer, sample)
            logger.info("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))

            record_loss.append(train_loss)

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                torch.save(self.model.state_dict(),
                           "{}/pre_trained_gnn/{}.{}.pth".format(folder_path, dataname, self.gnn_type))
                logger.info("+++model saved ! {}.{}.pth".format(dataname, self.gnn_type))

        return record_loss


if __name__ == '__main__':
    pass
