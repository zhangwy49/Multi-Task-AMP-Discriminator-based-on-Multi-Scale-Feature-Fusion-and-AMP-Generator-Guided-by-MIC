# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class CNN(nn.Module):
    def __init__(self, batch_size=128, embedding_size=20, num_tokens=100, num_filters=100, filter_sizes=(2, 3, 4), num_classes=1, num_heads=4):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.hidden1 = 20
        self.hidden2 = 60
        self.hidden3 = 20
        self.dropout = 0.3
        self.fc1 = nn.Linear(self.embedding_size, self.hidden1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_filters, (k, self.hidden1)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.num_filters, self.hidden2)
        self.new_fc3 = nn.Linear(self.hidden2, self.num_classes)
        self.softmax = nn.functional.softmax
        self.fc = nn.Linear(98 * 3, 1)

    def conv_and_pool(self, x, conv):  # [128,1,100,20]
        # print(x.shape) #[128, 1, 100, 20])
        x = F.relu(conv(x)).squeeze(3)  # [128 100 [31 30 29]=90 ]
        return x

    def forward(self, x):
        # tuple 2 [128,32]
        # [N,100,26]
        out = self.fc1(x)  # in[128,100,20]  out[128,100,20]
        #  out: [128 32 300]
        # [N,100,20]
        out = out.unsqueeze(1)  # in[128,100,20] out[128,1,100,20]
        # out: [128 1 32 300]   32 2 3 4     31 30 29=
        # [N, 1, 100, 20]
        # print(out.shape)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 2)
        # print(out.shape)
        out = self.fc(out).squeeze(2)

        # out: [128 768]
        # [N, 300]
        out = self.dropout(out)
        out = self.relu(out)
        # out: [128 100]
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.new_fc3(out)
        # out:[128 10]
        return out

