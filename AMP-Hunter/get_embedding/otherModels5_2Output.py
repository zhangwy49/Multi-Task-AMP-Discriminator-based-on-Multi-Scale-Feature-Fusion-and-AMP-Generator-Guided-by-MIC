import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""cnn"""
class CNN(nn.Module):
    def __init__(self, batch_size=128, embedding_size=20, num_tokens=100, num_filters=100, filter_sizes=(2, 3, 4), num_classes=2, num_heads=4):
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
        self.fc3 = nn.Linear(self.hidden2, self.num_classes)
        self.softmax = nn.functional.softmax
        self.fc = nn.Linear(98 * 3, 1)

    def conv_and_pool(self, x, conv):

        x = F.relu(conv(x)).squeeze(3)

        return x

    def forward(self, x):

        out = self.fc1(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 2)

        out = self.fc(out).squeeze(2)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out


