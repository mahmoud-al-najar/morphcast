import torch


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

        x = sample[0]
        y = sample[1]
        x = torch.from_numpy(x).double()
        y = torch.from_numpy(y).double()
        return x, y
