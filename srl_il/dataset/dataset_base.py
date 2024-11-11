import abc
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence
import h5py
import numpy as np
import torch
import torch.utils
from torch.utils.data import Dataset, Subset, Dataset
import pathlib
import struct
import torch.nn.functional as F
from glob import glob


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories. Each data sample is a trajectory.
    TrajectoryDataset[i] returns two dicts. 
        traj_dict:
            Key: the name of the data. e.g. camera, action, etc
            Value: The data. The first dimention is the full trajectory length.
        global_dict:
            Key: the name of the data. e.g. goal camera, natural language description, etc
            Value: The data. The first dimention is the full trajectory length.
    The data in these dicts can be hdf5 pointers(the hdf5.Dataset object that hasn't been loaded into memory) or numpy arrays.
    The actual data loading is done in `load` method.
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        """
        Returns the number of trajectories.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def load(self, data, key, is_global=False):
        """
        Load the data into memory and convert to torch array.
        args: 
        data: The data to be loaded, should be the same type as the elements returned by __getitem__
        key: The key of the data to be loaded
        is_global: If True, the data is global data. Otherwise, the data is trajectory data.
        """
        raise NotImplementedError


class TrajectorySubset(TrajectoryDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def __len__(self):
        return Subset.__len__(self)
    
    def load(self, data, key, is_global=False):
        return self.dataset.load(data, key, is_global)


def random_split_trajectory_dataset(dataset:TrajectoryDataset, N_elems:Sequence[int], random_seed:int=42):
    Ntotal = sum(N_elems)
    assert len(dataset) == Ntotal
    indices = torch.randperm(Ntotal).tolist()
    subsets = []
    start = 0
    for length in N_elems:
        end = start + length
        subsets.append(TrajectorySubset(dataset, indices[start:end]))
        start = end
    return subsets

class SequenceDataset(Dataset):
    """
    Slices trajectories from a TrajectoryDataset into windows of a fixed length.
    TrajectoryDataset[i] returns three tuples. 
        traj_tuple:
            Tuple of the trajectory data. Each element is a tensor of shape [window_size, ...]
        traj_masks:
            Tuple of the valid masks. True means valid, False means padded. Each element is a tensor of shape [window_size].
        global_tuple:
            Tuple of the global data. Each element is a tensor

    Args:
        keys_traj: list of tuples: each element represents a key in the dataset
            [keyname, srcname, start, end] The start and end are the indices of the window. Start and end can be none for the full sequence.
        keys_global: list of strings: each element represents a key in the global dataset
        pad_before: If True, pads the sequence before the start index.
        pad_after: If True, pads the sequence after the end index.
        pad_type: The type of padding. Can be 'zero', 'near'. Default is 'zero'.
    """
    def __init__(
        self,
        dataset: TrajectoryDataset,
        window_size: int,
        keys_traj: Sequence[tuple[str, str, Optional[int], Optional[int]]],
        keys_global: Sequence[str],
        pad_before: False,
        pad_after: True,
        pad_type: str = "zero",
    ):
        self._dataset = dataset
        self._window_size = window_size
        self._keys_traj = keys_traj
        self._keys_global = keys_global
        self._pad_before = pad_before
        self._pad_after = pad_after
        self._pad_type = pad_type

        # [start: end] will be loaded from the dataset[idx]
        self._idx_to_slice = [] # list of tuples: (idx, start, end, pad_before, pad_after)

        for i in range(len(self.dataset)):  # type: ignore
            seq_len = self.dataset.get_seq_length(i)  # avoid reading actual seq (slow)
            for j in range(-window_size+1, seq_len): # the logical start of the window
                start = max(0, j)
                end = min(seq_len, j + window_size)
                pad_before = max(0, -j)
                pad_after = max(0, j + window_size - seq_len)

                if (not self._pad_before) and pad_before > 0:
                    continue
                if (not self._pad_after) and pad_after > 0:
                    continue
                self._idx_to_slice.append((i, start, end, pad_before, pad_after))

        # check the keys
        data_sample_traj, data_sample_global = self.dataset[0]
        all_names = []
        for key, src, start, end in self._keys_traj:
            assert src in data_sample_traj, f"Key {key} is from {src}, which is not found in the dataset"
            start = 0 if start is None else start
            end = self._window_size if end is None else end
            assert 0<= start <= end <= self._window_size, "start must be >= 0"
            all_names.append(key)
        for key in self._keys_global:
            assert key in data_sample_global, f"Key {key} not found in the dataset"
            all_names.append(key)
        assert len(all_names) == len(set(all_names)), f"Duplicate keys found in {all_names}"
        assert self._pad_type in ["zero", "near"], f"Unknown pad_type {self._pad_type}"


    @property
    def dataset(self):
        return self._dataset
    
    @property
    def window_size(self):
        return self._window_size
    
    @property
    def keys_traj(self):
        return self._keys_traj
    
    @property
    def keys_global(self):
        return self._keys_global
    
    @property
    def pad_before(self):
        return self._pad_before
    
    @property
    def pad_after(self):
        return self._pad_after
    
    @property
    def pad_type(self):
        return self._pad_type
    

    def get_seq_length(self, idx: int) -> int:
        return self._window_size

    def __len__(self):
        return len(self._idx_to_slice)

    def __getitem__(self, idx):
        i, start, end, pad_before, pad_after = self._idx_to_slice[idx]
        traj_masks = {}
        ret_dict = {}
        data_traj, data_global = self.dataset[i]
        for key in self._keys_global:
            # global_tuple += (data_global[key],)
            ret_dict[key] = self.dataset.load(data_global[key], key, is_global=True)
        for key, src, key_start, key_end in self._keys_traj:
            key_start = 0 if key_start is None else key_start # idx in the window
            key_end = self._window_size if key_end is None else key_end # idx in the window

            slice = data_traj[src][start:end]

            key_pad_before = max(0, pad_before - key_start)
            key_start = max(0, key_start - pad_before) # idx in slice
            key_pad_after = max(0, pad_after - (self._window_size-key_end))
            key_end = min(end-start, end-start - (self._window_size - key_end - pad_after)) # idx in slice
            # key_pad_before + (key_end - key_start) + key_pad_after == self._window_size

            traj_parts = []
            mask_parts = []
            if key_pad_before > 0:
                if self._pad_type == "zero":
                    traj_parts.append(torch.zeros(key_pad_before, *slice.shape[1:], dtype=slice.dtype))
                elif self._pad_type == "near":
                    traj_parts.append(torch.repeat_interleave(self.dataset.load(slice[:1], src, is_global=False),
                                key_pad_before, dim=0))
                mask_parts.append(torch.zeros(key_pad_before, dtype=torch.bool))

            if key_end >0 and key_start < end-start:
                traj_parts.append(self.dataset.load(slice[key_start:key_end], src, is_global=False))
                mask_parts.append(torch.ones(key_end - key_start, dtype=torch.bool))

            if key_pad_after > 0:
                if self._pad_type == "zero":
                    traj_parts.append(torch.zeros(key_pad_after, *slice.shape[1:], dtype=slice.dtype))
                elif self._pad_type == "near":
                    traj_parts.append(torch.repeat_interleave(self.dataset.load(slice[-1:], src, is_global=False), 
                                    key_pad_after, dim=0))
                mask_parts.append(torch.zeros(key_pad_after, dtype=torch.bool))
            
            # traj_tuple += (torch.cat(traj_parts, dim=0),)
            # traj_masks += (torch.cat(mask_parts, dim=0),)
            ret_dict[key] = torch.cat(traj_parts, dim=0)
            traj_masks[key] = torch.cat(mask_parts, dim=0)

        return ret_dict, traj_masks


def get_train_val_test_seq_datasets(
    traj_dataset: TrajectoryDataset,
    test_fraction: float,
    val_fraction:float,
    window_size_train: int,
    window_size_test: int,
    keys_traj: Sequence[tuple[str, Optional[int], Optional[int]]],
    keys_global: Sequence[str],
    pad_before: bool,
    pad_after: bool,
    pad_type: str,
    random_seed: int = 42,
):
    """
    Splits a TrajectoryDataset into train, validation, and test sets. And build the SequenceDataset for each set.
    The definition of the train, val, test are different from the standard split.
        Train set is used for training, 
        Validation set is sampled from the same trajectories as the train set
        Test set is sampled from the remaining trajectories.
    """
    N_trajs = len(traj_dataset)
    if test_fraction == 0:
        train_traj_dataset = traj_dataset
        test_traj_dataset = None
    else:
        N_test = max(1, int(test_fraction * N_trajs))
        train_traj_dataset, test_traj_dataset = random_split_trajectory_dataset(
            traj_dataset, 
            [N_trajs - N_test, N_test], 
            random_seed=random_seed
        )

    seq_ds_kwargs = {
        "keys_traj": keys_traj,
        "keys_global": keys_global,
        "pad_before": pad_before,
        "pad_after": pad_after,
        "pad_type": pad_type,
    }
    train_seq_ds = SequenceDataset(train_traj_dataset, window_size=window_size_train, **seq_ds_kwargs)

    N_train = len(train_seq_ds)
    N_val = int(val_fraction * N_train)
    train_ds, val_ds = torch.utils.data.random_split(
        train_seq_ds, 
        [N_train - N_val, N_val], 
        generator= torch.Generator().manual_seed(random_seed)
    )

    if test_fraction == 0:
        test_ds = None
    else:
        test_ds = SequenceDataset(test_traj_dataset, window_size=window_size_test,  **seq_ds_kwargs)
    return train_ds, val_ds, test_ds
