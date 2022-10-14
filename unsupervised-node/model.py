import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from layers import *


class NodeClassificationModel(torch.nn.Module):
    def __init__(self, args):
        super(NodeClassificationModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_layers = args.num_layers

        self.gan = GAN(args)

        self.proj_head = Sequential(Linear(self.nhid, self.nhid), ReLU(inplace=True), Linear(self.nhid, self.nhid))

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Base model with GCNConv
        for i in range(self.num_layers):
            if i:
                conv = GCNConv(self.nhid, self.nhid)
            else:
                conv = GCNConv(self.num_features, self.nhid)
            bn = BatchNorm1d(self.nhid)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)

        x_pos = self.gan(x.detach(), edge_index, batch)

        x = self.proj_head(x)
        x_pos = self.proj_head(x_pos)

        loss = self.loss_cl(x, x_pos.detach())

        return loss

    def inference(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        xs = []

        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)

        return x
    
    def loss_cl(self, x1, x2):
        T = self.args.temperature
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix_a = torch.exp(sim_matrix_a / T)
        pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
        loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
        loss_a = - torch.log(loss_a).mean()

        sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
        sim_matrix_b = torch.exp(sim_matrix_b / T)
        pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
        loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
        loss_b = - torch.log(loss_b).mean()

        loss = (loss_a + loss_b) / 2
        return loss

    
    