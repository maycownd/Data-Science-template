# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # TensorBoard support
from ..preprocessing.fmnist import load_fmnist_torch

# calculate train time, writing train data to files etc.
import time
import pandas as pd
import json
from IPython.display import clear_output

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True) # On by default, leave it here for clarity



def main():
    train_set, test_set = load_fmnist_torch()
    
