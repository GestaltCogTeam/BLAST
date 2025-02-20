import pickle

import numpy as np
from torch.utils.data import Dataset


class BLASTDataset(Dataset):
    """Only used for pretraining"""

    def __init__(self, mode: str, num_valid_samples: int = None, **kwargs) -> None:
        super().__init__()
        assert mode in ['train', 'valid', 'test', 'val']
        if mode == 'val': mode = 'valid'

        self.mode = mode
        self.num_valid_samples = num_valid_samples

        shape = np.load(f"datasets/{self.mode}/shape.npy")
        self.memmap_data = np.memmap(f'datasets/{self.mode}/data.dat', dtype=np.float32, shape=tuple(shape), mode='r')

        if self.mode == 'valid' and self.num_valid_samples is not None:
            print(f"Using {self.num_valid_samples} samples for {self.mode} dataset")
            x = self.num_valid_samples
            y = self.memmap_data.shape[0]
            _p = (y - 1) / (x - 1)
            idx_list = list(range(self.num_valid_samples))
            idx_list = [int(_p * i) for i in idx_list]
            self.memmap_data = self.memmap_data[idx_list]

        print(f"Loaded {self.mode} dataset with shape {self.memmap_data.shape}")

    def __getitem__(self, idx: int) -> tuple:
        # get raw data
        seq = self.memmap_data[idx].astype(np.float32)

        # get puts, labels, and mask
        seq = np.nan_to_num(seq, nan=np.nan, posinf=np.nan, neginf=np.nan)
        mask = (~np.isnan(seq)).astype(np.int32)
        
        # normalize data
        mean = np.nanmean(seq)
        std = np.nanstd(seq)
        std = std if std != 0 else 1
        seq = (seq - mean) / std
        
        # get inputs and labels
        seq = np.nan_to_num(seq, nan=0.) # avoid nan in inputs
        inputs = seq[:-1]
        labels = seq[1:]
        mask = mask[1:]
        return {'inputs': inputs, 'labels': labels, 'mask': mask}

    def __len__(self):
        return self.memmap_data.shape[0]
