# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict, List

import torch


class BatchBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size: int = buffer_size
        self.batches: Dict[str, List]  = {}

    def add(self, batch: dict):
        # if it is not the first push
        if self.batches != {}:
            for k, v in batch.items():
                self.batches[k].append(v)

        # if it is the first push
        else:
            for k, v in batch.items():
                self.batches[k] = []
                self.batches[k].append(v)

    def build_dataset(self, duplicates=1):
        total_data = {}
        for k, v in self.batches.items():
            data = torch.cat(v)
            # if len(data) > self.buffer_size:
            #     print(f'WARNING: data ({k}) length exceeds maximum buffer size')
            total_data[k] = data[-self.buffer_size:]
            self.batches[k] = [total_data[k],] # handle off-policy samples
        return BufferDataset(total_data, duplicates)

    def __len__(self):
        keys = [k for k in self.batches.keys()]
        if len(keys) == 0:
            return 0
        return sum([len(batch) for batch in self.batches[keys[0]]])

    def state_dict(self) -> None:
        return {"batches": self.batches}

    def load_state_dict(self, state_dict: dict) -> None:
        self.batches = state_dict["batches"]


class BufferDataset(torch.utils.data.Dataset):
    def __init__(self, total_data: dict, duplicates: int):
        super().__init__()
        # has more than 1 key(s) & all values have same length
        assert len(total_data.keys()) > 0

        keys = list(total_data.keys())
        self.len = len(total_data[keys[0]])
        for v in total_data.values():
            assert len(v) == self.len

        self.total_data = total_data
        self.duplicates = duplicates # expand factor

    def __getitem__(self, idx):
        return {k: v[idx % self.len] for k, v in self.total_data.items()}

    def __len__(self):
        return self.len * self.duplicates


class DeviceBatchLoader:
    """ Same batches as `DataLoader(BufferDataset, shuffle=True)` (one pass over
        `len * duplicates` shuffled indices, last batch may be smaller),
        but the data lives on `device` and a batch is a single indexing op
        instead of `batch_size` python-level `__getitem__` calls + collate.
    """
    def __init__(self, dataset: BufferDataset, batch_size: int, device):
        self.data = {k: v.to(device) for k, v in dataset.total_data.items()}
        self.len = dataset.len
        self.n = len(dataset)
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return -(-self.n // self.batch_size)

    def __iter__(self):
        perm = torch.randperm(self.n, device=self.device)
        for i in range(0, self.n, self.batch_size):
            idx = perm[i:i + self.batch_size] % self.len
            yield {k: v[idx] for k, v in self.data.items()}
