import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Constant
from utils import *
from model import *


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--num_layers', type=int, default=5, help='number of conv layers')
parser.add_argument('--dataset', type=str, default='MUTAG', help='DD/PROTEINS/NCI1/MUTAG/COLLAB/IMDB-BINARY/REDDIT-BINARY/REDDIT-MULTI-5K')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature in info-nce loss')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

if args.dataset in {'DD', 'PROTEINS', 'NCI1', 'MUTAG'}:
    dataset = TUDataset(os.path.join('./data', args.dataset), name=args.dataset, use_node_attr=True)
else:
    dataset = TUDataset(os.path.join('./data', args.dataset), name=args.dataset)
    dataset.transform = Constant(value=1.0, cat=False)

args.num_features = dataset.num_features

print(args)

loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

model = GraphClassificationModel(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(model, optimizer, loader):
    min_loss = 1e10
    patience_cnt = 0
    loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        for i, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            loss = model(data)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.8f}'.format(loss_train), 'time: {:.6f}s'.format(time.time() - t))

        loss_values.append(loss_train)
        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        if loss_values[-1] < min_loss:
            min_loss = loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch


def gen_representations(model, loader):
    model.eval()
    
    cnt = 0
    representations = None
    labels = None
    
    for data in loader:
        cnt += 1
        data = data.to(args.device)
        embeddings = model.inference(data)

        if cnt == 1:
            labels = data.y.cpu().detach().numpy()
            representations = embeddings.cpu().detach().numpy()
        else:
            y = data.y.cpu().detach().numpy()
            labels = np.concatenate((labels, y))
            embeddings = embeddings.cpu().detach().numpy()
            representations = np.concatenate((representations, embeddings))
    
    return representations, labels


if __name__ == '__main__':
    best_model = train(model, optimizer, loader)
    model.load_state_dict(torch.load('{}.pth'.format(best_model)))
    representations, labels = gen_representations(model, loader)
    print('\nK-fold cross validation results:')
    k_fold_cross_val(representations, labels)
