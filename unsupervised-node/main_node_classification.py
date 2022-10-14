import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from torch_geometric.datasets import WikiCS, Amazon, Coauthor
from utils import *
from model import *


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=1024, help='hidden size')
parser.add_argument('--num_layers', type=int, default=1, help='number of conv layers')
parser.add_argument('--dataset', type=str, default='WikiCS', help='WikiCS/Computers/Photo/CS/Physics')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature in info-nce loss')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

if args.dataset in {'Computers', 'Photo'}:
    dataset = Amazon(os.path.join('./data', args.dataset), name=args.dataset)
    data = create_masks(dataset.data)
elif args.dataset in {'CS', 'Physics'}:
    dataset = Coauthor(os.path.join('./data', args.dataset), name=args.dataset)
    data = create_masks(dataset.data)
else:
    dataset = WikiCS(os.path.join('./data', args.dataset))
    data = create_masks(dataset.data)

args.num_features = data.x.size(1)
args.num_nodes = data.x.size(0)

print(args)

data = data.to(args.device)

model = NodeClassificationModel(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train(model, optimizer):
    min_loss = 1e10
    patience_cnt = 0
    loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train), 'time: {:.6f}s'.format(time.time() - t))

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
    print('\nOptimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))
    
    return best_epoch


def compute_representation(model, data):
    model.eval()
    
    data = data.to(args.device)
    embeddings = model.inference(data)
    embeddings = embeddings.cpu().detach().numpy()

    return embeddings


if __name__ == '__main__':
    best_model = train(model, optimizer)
    model.load_state_dict(torch.load('{}.pth'.format(best_model)))

    embeddings = compute_representation(model, data)
    evaluate_node_classification_LR_model(embeddings, data, args.dataset)
