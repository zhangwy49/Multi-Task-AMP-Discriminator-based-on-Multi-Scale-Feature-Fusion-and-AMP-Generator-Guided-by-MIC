import torch
import os
import numpy as np
import random


class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
