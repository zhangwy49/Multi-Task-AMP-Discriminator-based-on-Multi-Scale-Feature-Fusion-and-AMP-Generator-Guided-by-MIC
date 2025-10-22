import warnings
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import trange
from torch_geometric.utils import k_hop_subgraph, index_to_mask
from datasets import DataLoader
from models import *
import argparse
import numpy as np
import pandas as pd
import random
import itertools
from Bio import SeqIO
from config import Config, seed_everything


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, out

@torch.no_grad()
def valid_reg(data):
    model.eval()
    out = model(data)
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    return val_loss

@torch.no_grad()
def valid_class(data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    val_correct = pred[data.val_mask] == data.y[data.val_mask]  # Check against ground-truth labels.
    val_acc = int(val_correct.sum()) / int(data.val_mask.sum())  # Derive ratio of correct predictions.
    return val_acc

@torch.no_grad()
def test_reg(data):
    model.eval()
    out = model(data)
    loss = criterion(out[data.test_mask], data.y[data.test_mask])
    return loss

@torch.no_grad()
def test_class(data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  

    return test_acc

@torch.no_grad()
def run_full_data_reg(data, forcing=0):
    model.eval() 
    mask = data.train_mask.detach().view(-1, 1)
    out = model(data)  
    
    if forcing:
        print('use forcing...')
        pred = data.y.detach().view(-1, 1) * mask + out * ~mask  
    else:
        pred = out  

    return pred

@torch.no_grad()
def run_full_data_class(data, forcing=0):
    model.eval()
    mask = data.train_mask.detach().view(-1, 1)
    out = model(data)
    pred = out.argmax(dim=1, keepdim=True)  # Use the class with highest probability.
    if forcing:
        print('use forcing...')
        pred = data.y.detach().view(-1, 1) * mask + pred * ~mask
    onehot = torch.zeros(out.shape, device=Config.device)#out.shape
    onehot.scatter_(1, pred, 1)

    return onehot


def cal_nei_index(ei, k, num_nodes, include_self=1):
    if not os.path.exists('index'):
        os.makedirs('index')
    if include_self:
        path_name = f'index/{args.dataset}_hop{k}.npy'
    else:
        path_name = f'index/{args.dataset}_hop{k}_noself.npy'
    if os.path.exists(path_name):
        neigh_dict = np.load(path_name, allow_pickle=True).item()
    else:
        neigh_dict = {}
        for id in trange(num_nodes):
            # neigh = k_hop_subgraph(id, k, ei)[0]
            # exclude self
            if include_self:
                neigh = k_hop_subgraph(id, k, ei)[0]
            else:
                neigh = k_hop_subgraph(id, k, ei)[0][1:]
            neigh_dict[id] = neigh
        np.save(path_name, neigh_dict)
    return neigh_dict


def cal_nc_class(nei_dict, y, thres=2., use_tensor=True):
    if not use_tensor:
        y = y.numpy()
    nc = np.empty(y.shape[0])
    for i, neigh in nei_dict.items():
        if use_tensor:
            labels = torch.index_select(y, 0, neigh.to(y.device))
            if len(labels):
                nc[i] = len(labels) / torch.max(torch.sum(labels, dim=0)).item()
            else:
                nc[i] = 1.0
        else:
            labels = y[neigh].reshape(len(neigh))
            if len(labels):
                nc[i] = len(labels) / max(np.bincount(labels))
            else:
                nc[i] = 1.0

    # low_cc: 1 ; high_cc: 0
    mask = np.where(nc <= thres, 1., 0.)
    return torch.from_numpy(mask).float().to(Config.device)

def cal_nc_reg(nei_dict, y, thres=0.5, use_tensor=True):
    if not use_tensor:
        y = y.numpy()
    
    nc = np.empty(y.shape[0])  

    for i, neigh in nei_dict.items():
        if use_tensor:
            labels = torch.index_select(y, 0, neigh.to(y.device))  
            if len(labels) > 0:
                mean_neighbor = torch.mean(labels).item()  
                nc[i] = abs(y[i].item() - mean_neighbor)  
            else:
                nc[i] = 0.0  
        else:
            labels = y[neigh].reshape(len(neigh))
            if len(labels) > 0:
                mean_neighbor = np.mean(labels)  
                nc[i] = abs(y[i] - mean_neighbor)  
            else:
                nc[i] = 0.0  

    # nc > T set as low_cc
    mask = np.where(nc > thres, 1., 0.)
    
    return torch.from_numpy(mask).float().to(Config.device)


def compute_threshold(y, args_threshold):
    print(y)
    np_y = y.detach().cpu().numpy()
    sigma_y = np.std(np_y)  
    if sigma_y == 0:  
        sigma_y = 1e-6  
    threshold = 2 ** (args_threshold / 10 * np.log2(sigma_y))
    return threshold


def compute_splits(start, end):
    n = end - start
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_idx = torch.arange(start, start + n_train)
    val_idx = torch.arange(start + n_train, start + n_train + n_val)
    test_idx = torch.arange(start + n_train + n_val, end)
    return train_idx, val_idx, test_idx


def gpr_splits(data, num_classes, pos_count=3280):
    num_nodes = data.num_nodes

    pos_start, pos_end = 0, pos_count           # [0, pos_count-1]
    neg_start, neg_end = pos_count, pos_count * 2 # [pos_count, pos_count*2-1]

    

    pos_train, pos_val, pos_test = compute_splits(pos_start, pos_end)
    neg_train, neg_val, neg_test = compute_splits(neg_start, neg_end)

    train_index = torch.cat([pos_train, neg_train])
    val_index = torch.cat([pos_val, neg_val])
    test_index = torch.cat([pos_test, neg_test])

    data.train_mask = index_to_mask(train_index, size=num_nodes)
    data.val_mask = index_to_mask(val_index, size=num_nodes)
    data.test_mask = index_to_mask(test_index, size=num_nodes)

    return data




if __name__ == "__main__":
    # PARSER BLOCK
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, default='AMP_class')
    parser.add_argument('--baseseed', '-S', type=int, default=42)
    parser.add_argument('--hidden', '-H', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--dp1', type=float, default=0.5)
    parser.add_argument('--dp2', type=float, default=0.5)
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--hops', type=int, default=1)
    parser.add_argument('--forcing', type=int, default=1, choices=[0, 1])
    parser.add_argument('--addself', '-A', type=int, default=1, choices=[0, 1])
    parser.add_argument('--model', '-M', type=str, default='NCGCN')
    parser.add_argument('--threshold', '-T', type=float, default=3)

    args = parser.parse_args()
    dataset, data = DataLoader(args.dataset) 
    print(f"load {args.dataset} dataset successfully!")

    warnings.filterwarnings("ignore")

    args_dict = vars(args)
    args = argparse.Namespace(**args_dict)

    # for classification
    args.threshold = 2 ** (args.threshold / 10 * np.log2(dataset.num_classes))
    # for regression 
    #args.threshold = compute_threshold(data.y, args.threshold)
    data.y = torch.tensor(data.y).to('cuda')
    print(args)

    train_rate = 0.8
    val_rate = 0.1
    # dataset split
    num_nodes = dataset.num_nodes
    
    if torch.cuda.is_available():
        model = NCGCN(dataset.num_features, dataset.num_classes, args).to('cuda')
    
    data.edge_index = data.edge_index.to('cuda')
    data.x = SparseTensor.from_dense(data.x.to('cuda'))  

    neigh_dict = cal_nei_index(data.edge_index, args.hops, dataset.num_nodes)

    print('indexing finished')
    # training settings
    data.cc_mask = torch.ones(dataset.num_nodes).float()
    print(data.cc_mask.shape)
        
    if torch.cuda.is_available():
        data = gpr_splits(data, dataset.num_classes,pos_count=3280).to('cuda')

    model.reset_parameters()
    #criterion = torch.nn.MSELoss() # regerssion task
    criterion = torch.nn.CrossEntropyLoss() #classification

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    data.update_cc = True

    best_metric = 0. #for classification best_loss = 0.
    final_test_metric = 0.
    es_count = patience = 100
    for epoch in range(20):
        loss, out = train(data)
        data.update_cc = False
        val_metric = valid_class(data)
        test_metric = test_class(data)
        #if val_metric < best_metric: # regression task
        if val_metric >= best_metric: # classification task
            es_count = patience
            best_metric = val_metric
            final_test_metric = test_metric
            predict = run_full_data_class(data, args.forcing)
            #data.cc_mask = cal_nc_reg(neigh_dict, predict.detach().cpu(), args.threshold)  #regression task
            data.cc_mask = cal_nc_class(neigh_dict, predict.detach().cpu(), args.threshold) # classification task
            data.update_cc = True
        else:
            es_count -= 1
        if es_count <= 0:
            break
        print(best_metric)
    val_metric = torch.tensor(best_metric)
    test_metric = torch.tensor(final_test_metric)
    print(f'{args.dataset} valid_metric: {100 * val_metric.item():.2f}')
    print(f'{args.dataset} test_metric: {100 * test_metric.item():.2f}')

    # get embedding
    model.eval()
    all_embeddings = model.get_embeddings(data).cpu().numpy()  
    
    # dataset split 
    train_embeddings = all_embeddings[data.train_mask.cpu().numpy()]
    val_embeddings = all_embeddings[data.val_mask.cpu().numpy()]
    test_embeddings = all_embeddings[data.test_mask.cpu().numpy()]

    np.save("../AMP_dataset/embedding/NCGCN_embedding_train.npy", train_embeddings)
    np.save("../AMP_dataset/embedding/NCGCN_embedding_test.npy", test_embeddings)
    np.save("../AMP_dataset/embedding/NCGCN_embedding_val.npy", val_embeddings)

    