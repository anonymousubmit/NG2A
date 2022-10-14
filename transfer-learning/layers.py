import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, Size, OptTensor
from torch_scatter import scatter
from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.nn import global_add_pool, global_max_pool


num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 


class NeighborPropagate(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(NeighborPropagate, self).__init__()

        self.emb_dim = emb_dim
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class Generator(torch.nn.Module):
    def __init__(self, nhid):
        super(Generator, self).__init__()
        self.nhid = nhid

        self.fc1 = torch.nn.Linear(self.nhid, self.nhid)
        self.fc2 = torch.nn.Linear(self.nhid, self.nhid)

        self.bilinear = nn.Parameter(torch.Tensor(1, self.nhid))
        nn.init.xavier_uniform_(self.bilinear.data)
    
    def forward(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))
    
    def discriminate(self, x, z):
        value = ((x * z) * self.bilinear).sum(dim=1)
        return torch.sigmoid(value)


class GAN(torch.nn.Module):
    def __init__(self, nhid):
        super(GAN, self).__init__()
        self.nhid = nhid

        self.neighborprop = NeighborPropagate(nhid)

        self.generator = Generator(nhid)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001)
    
    def forward(self, x, edge_index, batch, edge_attr):
        # calculate neighborhood representation
        self.optimizer_G.zero_grad()
        x_neighbor = self.neighborprop(x, edge_index, edge_attr)
        gaussian_noise = x_neighbor + torch.randn((x.size(0), self.nhid), device=x.device)
        x_gen = self.generator(gaussian_noise)

        x_gen_pool = global_add_pool(x_gen, batch)
        x_pool = global_add_pool(x, batch)
        x_neg_pool = x_pool[torch.randperm(x_pool.size(0))]

        x_neg = x[torch.randperm(x.size(0))]

        pos_loss = - torch.log(self.generator.discriminate(x_gen_pool, x_pool) + 1e-15).mean()
        neg_loss = - torch.log(1 - self.generator.discriminate(x_gen_pool, x_neg_pool) + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        self.optimizer_G.step()

        return x_gen
