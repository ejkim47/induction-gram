from torch.utils.data import Dataset
import torch
from os.path import join
import os
import numpy as np


class NextWordDataset(Dataset):
    '''This class is used to create a dataset for the next word prediction task.
    It returns tensors of token indexes (numbers) that can be decoded and inspected with the tokenizer.
    '''

    def __init__(self, data_dir: str, context_length: int, stride: int, split: str = 'train'):
        self.context_length = context_length
        self.stride = stride
        self.split = split
        self.tokens_ = np.memmap(join(data_dir, split, f'tokenized.0'), dtype=np.uint16, mode='r')
        self._get_selected_indices(data_dir,  split)

    def _get_selected_indices(self, data_dir, split):
        if os.path.exists(join(data_dir, f'{split}_selected_indices_{self.context_length + 1}-s{self.stride}.npy')):
            selected_indices = np.load(join(data_dir, f'{split}_selected_indices_{self.context_length + 1}-s{self.stride}.npy'))
        else:
            selected_indices = np.array([False] * (len(self.tokens_) - self.context_length))
            offsets = (np.memmap(join(data_dir, split, 'offset.0'), dtype=np.uint64, mode='r') // 2).astype(np.int64)
            mask = (offsets[1:] - self.context_length - 1) - offsets[:-1] > 0
            for a, b in zip(offsets[:-1][mask], offsets[1:][mask]):
                selected_indices[int(a)+1:int(b) - self.context_length:self.stride] = True
            if len(self.tokens_) - self.context_length - 1 - offsets[-1] > 0:
                selected_indices[int(offsets[-1]) + 1: int(len(self.tokens_) - self.context_length):self.stride] = True
            np.save(join(data_dir, f'{split}_selected_indices_{self.context_length + 1}-s{self.stride}.npy'), selected_indices)
        self.idx2newidx = selected_indices.nonzero()[0]

    def __len__(self):
        return len(self.idx2newidx)

    def __getitem__(self, idx):
        new_idx = self.idx2newidx[idx]
        tokens = torch.tensor(self.tokens_[new_idx: new_idx + self.context_length].astype(np.int32)).long()
        next_token = torch.tensor(self.tokens_[new_idx + self.context_length].astype(np.int32)).long()
        return tokens, next_token
