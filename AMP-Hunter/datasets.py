#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import os.path as osp
import torch_geometric.transforms as T

from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import remove_self_loops


class AMP_Loader(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        
        super(AMP_Loader, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        node_features_path = '../AMP_dataset/processed/node_feature_label_class.txt'
        edge_features_path = '../AMP_dataset/processed/graph_edges.txt'

        
        with open(node_features_path, 'r') as f:
            x = []
            y = []
            for line in f:
                fields = line.split('\t')
                y.append(int(fields[2]))
                split_values = fields[1].split(',')
                sequences = [float(v) for v in split_values]
                x.append(sequences)
                
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.long)
            #y = torch.tensor(y, dtype=torch.float)

        with open(edge_features_path, 'r') as f:
            data = []
            for line in f:
                fields = line.split('\t')
                edge = [int(v) for v in fields]
                data.append(edge)
                
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = remove_self_loops(edge_index)
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

def DataLoader(name):
    dataset = AMP_Loader(root='../data/', name=name, pre_transform=T.NormalizeFeatures())
    
    dataset.num_nodes = len(dataset[0].y)
    return dataset, dataset[0]



