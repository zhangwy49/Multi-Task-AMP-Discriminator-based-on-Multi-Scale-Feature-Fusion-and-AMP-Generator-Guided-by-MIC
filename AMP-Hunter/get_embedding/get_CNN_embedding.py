import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import math
import argparse
from Bio import SeqIO

# Data paths
trainPath = "../AMP_dataset/raw/train.txt"
valPath = "../AMP_dataset/raw/val.txt"
testPath = "../AMP_dataset/raw/test.txt"


batch_size = 6560
modelPath = "../models/CNN_class.pth" #classification
#modelPath = "../models/CNN_reg.pth" #regression

# Sequence to numerical mapping
mydict = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19}

#softmax = nn.functional.softmax


def dataProcessPipeline(seq):
    testest = seq
    num_seq = [mydict[character.upper()] for character in seq]

    seq = np.array(num_seq, dtype=int)
    len = seq.shape[0]
    torch_seq = torch.tensor(seq)

    if torch.sum(torch_seq[torch_seq < 0]) != 0:
        print(torch_seq[torch_seq < 0])
        print("wrong seq:", seq)
        print(testest)

    onehotSeq = torch.nn.functional.one_hot(torch_seq, num_classes=20)
    # Pad the sequence to a length of 100
    pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, 100 - len))
    mask = np.zeros(100, dtype=int)
    mask[len:] = 1
    mask = torch.tensor(mask)
    pad_seq = pad(onehotSeq)

    return pad_seq, mask


class Dataset(Dataset):
    def __init__(self, txt_path):
        self.seqs = []
        self.labels = []
        self.values = []
        self._load_from_txt(txt_path)

    def _load_from_txt(self, txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  
                
                parts = line.split()  
                if len(parts) < 3:
                    continue  

                first_col = parts[0]  
                seq = parts[1]   
                mic = float(parts[2])   
                
                if first_col.startswith("AMP"):
                    label = 1
                elif first_col.startswith("NAMP"):
                    label = 0
                else:
                    continue 
                
                self.seqs.append(seq)
                self.labels.append(label)
                self.values.append(mic)
                #print(self.values.type)

    def __getitem__(self, index):
        seq = self.seqs[index]
        num_seq, mask = dataProcessPipeline(seq)
        label = self.labels[index] #classification
        #label = torch.tensor(self.values[index], dtype=torch.float32) #regression
        return num_seq, mask, label

    def __len__(self):
        return len(self.seqs)


trainData = Dataset(txt_path=trainPath)
valData = Dataset(txt_path=valPath)
testData = Dataset(txt_path=testPath)

test_loader = DataLoader(dataset=testData, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=valData, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(dataset=trainData, batch_size=batch_size, shuffle=False)


def save_input_hook(module, input, output):
    
    input_data = input[0]
    global input_to_fc3
    input_to_fc3 = input_data.detach().cpu().to('cuda')

def hook_embeddings(modelPath,data_loader,task):
    model = torch.load(modelPath)
    model.cuda()
    model.eval()
    for data in data_loader:
        inputs, masks, labels = data
        inputs = inputs.float()
        masks = masks.float()
        #regression
        labels = labels.float()
        #classification
        #inputs, masks, labels = Variable(inputs), Variable(masks), Variable(labels)

        inputs = inputs.cuda()
        masks = masks.cuda()
        
        hook_handle = model.fc3.register_forward_hook(save_input_hook)  #classification task
        #hook_handle = model.new_fc3.register_forward_hook(save_input_hook) #regression task
        out = model(inputs)
        hook_handle.remove()

        if task == "train":
            np.save("../AMP_dataset/embedding/CNN_embedding_train.npy", input_to_fc3.cpu().numpy())
        elif task == "val":
            np.save("../AMP_dataset/embedding/CNN_embedding_val.npy", input_to_fc3.cpu().numpy())
        else:
            np.save("../AMP_dataset/embedding/CNN_embedding_test.npy", input_to_fc3.cpu().numpy())

        hook_handle.remove()
            

hook_embeddings(modelPath, train_loader, "train")
hook_embeddings(modelPath, val_loader, "val")
hook_embeddings(modelPath, test_loader, "test")
