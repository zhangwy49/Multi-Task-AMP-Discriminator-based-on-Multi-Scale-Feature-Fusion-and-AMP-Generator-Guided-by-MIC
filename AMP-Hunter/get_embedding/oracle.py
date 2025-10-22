import warnings
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import itertools
import math
import argparse
from Bio import SeqIO
import os



# Configuration parameters

batch_size = 656
MAX_MIC = math.log10(8192)
My_MAX_MIC = math.log10(600)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")


base_dir = os.path.dirname(os.path.abspath(__file__))



mydict = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19}
myInvDict = dict([val, key] for key, val in mydict.items())



def dataProcessPipeline(seq):

    # this function first transform peptide sequences into numerical sequence,
    # transformer it into onehot vector and padding them into a fix length
    # returning the padding vector and mask

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
    # onehotSeq = torch.nn.functional.one_hot(c
    pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, 100 - len))
    mask = np.zeros(100, dtype=int)
    mask[len:] = 1
    mask = torch.tensor(mask)

    pad_seq = pad(onehotSeq)

    return pad_seq, mask


class ListDataset(Dataset):
    def __init__(self, sequence_list):
        self.sequences = sequence_list

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        num_seq, mask = dataProcessPipeline(seq)

        return num_seq, mask, seq


    def __len__(self):
        return len(self.sequences)


class Oracle:
    def __init__(self):
        print("Current working directory:", os.path.dirname(os.path.abspath(__file__)))
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_path = os.path.join(script_dir, "../models/CNN_reg.pth")
        model_path = os.path.normpath(model_path)
        
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)


    def predict_from_sequences(self, sequence_list,device=device,batch_size=batch_size):
        cleaned_sequences = [seq.replace("X", "") for seq in sequence_list]
        dataset = ListDataset(cleaned_sequences)
        return self._predict(dataset,batch_size)

    def _predict(self, dataset,batch_size=batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        fitness_scores = []

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, masks, seqs = data
                inputs = inputs.float().cuda()

                out = self.model(inputs)
                out_ori = torch.squeeze(out,dim=1)
                out_numpy = out_ori.cpu().numpy().tolist()
                fitness_scores.extend([round(v, 3) for v in out_numpy])

        return fitness_scores



