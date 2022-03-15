import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence, pad_sequence
import pandas as pd
import numpy as np
# from models import VanillaLSTM, VanillaGRU, VanillaRNN
from Dyck_Generator_Suzgun_Batch import DyckLanguage
from Dyck1_Datasets import NextTokenPredictionTrainDataset, NextTokenPredictionLongTestDataset, \
    NextTokenPredictionShortTestDataset, NextTokenPredictionValidationDataset
from torch.utils.data import Dataset, DataLoader
# import Dyck_Generator_Suzgun
# from Dyck_Generator_Suzgun import DyckLanguage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# vocab = ['PAD','(', ')']
vocab = {'PAD':0, '(':1,')':2}
tags = {'':0, '0':1, '1':2}
n_letters= len(vocab)-1
n_tags = len(tags)-1
num_bracket_pairs = 25
length_bracket_pairs = 50