import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, pairs):
        """Initialization"""
        self.pairs = pairs

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.pairs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        sample = self.pairs[index]

        subarea1 = sample[0]
        subarea2 = sample[1]
        hs = sample[2]
        tp = sample[3]
        direction = sample[4]

        subarea1 = torch.from_numpy(subarea1).double()
        subarea2 = torch.from_numpy(subarea2).double()
        hs = torch.from_numpy(hs).double()
        tp = torch.from_numpy(tp).double()
        direction = torch.from_numpy(direction).double()

        return subarea1, subarea2, hs, tp, direction
