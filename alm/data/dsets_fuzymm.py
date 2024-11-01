from torch.utils.data import Dataset
import torch
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class SentenceDataset(Dataset):
    '''This class is used to create a dataset for the next word prediction task.
    It returns tensors of token indexes (numbers) that can be decoded and inspected with the tokenizer.
    '''

    def __init__(self, data_dir: str, max_n_tokens: int, min_n_tokens: int, split: str = 'train', pad_token_id: int = 50256):
        self.max_n_tokens = max_n_tokens
        self.split = split
        self.pad_token_id = pad_token_id
        self.tokens_ = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
        if os.path.exists(os.path.join(data_dir, f'{split}_selected_indices_{min_n_tokens}-{max_n_tokens}.npy')):
            selected_indices = np.load(os.path.join(data_dir, f'{split}_selected_indices_{min_n_tokens}-{max_n_tokens}.npy'))
        else:
            pad_index = (self.tokens_ == self.pad_token_id)
            window = sliding_window_view(pad_index, self.max_n_tokens)
            selected_indices = ~window[:, :min_n_tokens].any(1)
            np.save(os.path.join(data_dir, f'{split}_selected_indices_{min_n_tokens}-{max_n_tokens}.npy'), selected_indices)
        self.idx2newidx = selected_indices.nonzero()[0]

    def __len__(self):
        return len(self.idx2newidx)

    def __getitem__(self, idx):
        new_idx = self.idx2newidx[idx]
        tokens = torch.tensor(self.tokens_[new_idx: new_idx + self.max_n_tokens].astype(np.int32)).long()
        if tokens.max() == self.pad_token_id:
            tokens[tokens.argmax():] = self.pad_token_id
        return new_idx, tokens


class SentenceDatasetfromSuffixArray(Dataset):
    '''This class is used to create a dataset for the next word prediction task.
    It returns tensors of token indexes (numbers) that can be decoded and inspected with the tokenizer.
    '''

    def __init__(self, data_dir: str, max_n_tokens: int, min_n_tokens: int, pad_token_id: int = 50256):
        self.max_n_tokens = max_n_tokens
        self.min_n_tokens = min_n_tokens
        self.pad_token_id = pad_token_id
        self.tokens_ = np.memmap(os.path.join(data_dir, f'tokenized.0'), dtype=np.uint16, mode='r')
        self._get_selected_indices(data_dir)

    def _get_selected_indices(self, data_dir):
        if os.path.exists(os.path.join(data_dir, f'selected_indices_{self.min_n_tokens}-{self.max_n_tokens}.npy')):
            selected_indices = np.load(os.path.join(data_dir, f'selected_indices_{self.min_n_tokens}-{self.max_n_tokens}.npy'))
        else:
            selected_indices = np.array([False] * (len(self.tokens_) - self.max_n_tokens + 1))
            offsets = np.memmap(os.path.join(data_dir, f'offset.0'), dtype=np.uint64, mode='r') // 2
            mask = (offsets[1:] - self.min_n_tokens) - offsets[:-1] > 0
            for a, b in zip(offsets[:-1][mask], offsets[1:][mask]):
                selected_indices[int(a)+1:int(b)+1-self.min_n_tokens] = True
            if len(self.tokens_) - self.min_n_tokens - offsets[-1] > 0:
                selected_indices[int(offsets[-1]) + 1: int(len(self.tokens_) - self.min_n_tokens) + 1] = True
            np.save(os.path.join(data_dir, f'selected_indices_{self.min_n_tokens}-{self.max_n_tokens}.npy'), selected_indices)
        self.idx2newidx = selected_indices.nonzero()[0]

    def __len__(self):
        return len(self.idx2newidx)

    def __getitem__(self, idx):
        new_idx = int(self.idx2newidx[idx])
        tokens = torch.tensor(self.tokens_[new_idx: new_idx + self.max_n_tokens].astype(np.int32)).long()
        if tokens.max() == self.tokens_[0]:
            tokens[tokens.argmax():] = self.pad_token_id
        if len(tokens) < self.max_n_tokens:
            tokens = torch.cat([tokens, torch.ones(self.max_n_tokens - len(tokens)).long() * self.pad_token_id])
        return new_idx, tokens

