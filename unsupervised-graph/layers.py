import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, Size, OptTensor
from torch_scatter import scatter
from torch_geometric.nn import global_add_pool


class NeighborPropagate(MessagePassing):
    def __init__(self, aggr: str = 'mean', **kwargs,):
        kwargs['aggr'] = aggr if aggr != 'lstm' else None
        super().__init__(**kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def aggregate(self, x: Tensor, index: Tensor, ptr: Optional[Tensor] = None, dim_size: Optional[int] = None) -> Tensor:
        return scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)


class Generator(torch.nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.nhid = args.nhid

        self.fc1 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.fc2 = torch.nn.Linear(self.nhid // 2, self.nhid)

        self.bilinear = nn.Parameter(torch.Tensor(1, args.nhid))
        nn.init.xavier_uniform_(self.bilinear.data)
    
    def forward(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))
    
    def discriminate(self, x, z):
        value = ((x * z) * self.bilinear).sum(dim=1)
        return torch.sigmoid(value)


class GAN(torch.nn.Module):
    def __init__(self, args):
        super(GAN, self).__init__()
        self.args = args
        self.nhid = args.nhid

        self.neighborprop = NeighborPropagate().to(args.device)

        self.generator = Generator(self.args).to(args.device)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001)
    
    def forward(self, x, edge_index, batch):
        # calculate neighborhood representation
        self.optimizer_G.zero_grad()
        x_neighbor = self.neighborprop(x, edge_index)
        gaussian_noise = x_neighbor + torch.randn((x.size(0), self.nhid), device=self.args.device)
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
