import os
import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, list_ids, labels):
        """Initialization"""
        self.labels = labels
        self.list_IDs = list_ids

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        sample_id = self.list_IDs[index]

        data_dir_path = '/media/mn/WD4TB/topo/survey_dems/xyz_data/FULL'
        x = np.load(os.path.join(data_dir_path, f'{sample_id}_FULL.npy'))
        y = np.load(os.path.join(data_dir_path, f'{self.labels[sample_id]}_FULL.npy'))

        return x, y
