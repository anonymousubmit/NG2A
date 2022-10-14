import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from layers import *


class GraphClassificationModel(torch.nn.Module):
    def __init__(self, args):
        super(GraphClassificationModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_layers = args.num_layers

        self.gan = GAN(args)

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.embedding_dim = self.nhid * args.num_layers
        self.proj_head = Sequential(Linear(self.embedding_dim, self.embedding_dim), ReLU(inplace=True), Linear(self.embedding_dim, self.embedding_dim))

        # Base model with GINConv
        for i in range(self.num_layers):
            if i:
                nn = Sequential(Linear(self.nhid, self.nhid))
            else:
                nn = Sequential(Linear(self.num_features, self.nhid))
            conv = GINConv(nn, train_eps=True)
            bn = BatchNorm1d(self.nhid)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        xs = []
        xpos = []

        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

            x_pos = self.gan(x.detach(), edge_index, batch)
            xpos.append(x_pos)
        
        x_pool = [global_add_pool(x, batch) for x in xs]
        xpos_pool = [global_add_pool(x, batch) for x in xpos]

        x = torch.cat(x_pool, dim=1)
        x = self.proj_head(x)
        x_pos = torch.cat(xpos_pool, dim=1)
        x_pos = self.proj_head(x_pos)

        loss = self.loss_cl(x, x_pos.detach())

        return loss

    def inference(self, data):
        with torch.no_grad():
            x, edge_index, batch = data.x, data.edge_index, data.batch
            edge_attr = None

            xs = []

            for i in range(self.num_layers):
                x = F.relu(self.convs[i](x, edge_index))
                x = self.bns[i](x)
                xs.append(x)

            x_pool = [global_add_pool(x, batch) for x in xs]
            x = torch.cat(x_pool, dim=1)

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
