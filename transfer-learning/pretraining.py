import argparse
from functools import partial

from loader import MoleculeDataset
from torch_geometric.loader import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter


def train(args, model, device, dataset, optimizer):
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=True)

    model.train()

    train_loss_accum = 0

    for step, data in enumerate(tqdm(loader, desc='Iteration')):
        data = data.to(device)

        optimizer.zero_grad()
        loss = model.forward_cl(data)
        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
    
    return train_loss_accum/(step+1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0.0, help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last", help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default = 'init_weights/', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset_name = args.dataset
    dataset = MoleculeDataset("dataset/" + dataset_name, dataset=dataset_name)

    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))

        train_loss = train(args, model, device, dataset, optimizer)
        print(train_loss)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), args.output_model_file + "model_" + str(epoch) + ".pth")

if __name__ == "__main__":
    main()
