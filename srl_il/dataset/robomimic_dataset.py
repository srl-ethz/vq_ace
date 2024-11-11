from .dataset_base import TrajectoryDataset, get_train_val_test_seq_datasets
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
import tqdm


class RobomimicTrajectorySequenceDatasetPreloaded(TrajectoryDataset):
    def __init__(self, data_path, rgb_processing_keys):
        data_path = Path(data_path)
        self.data = []
        self.seq_lengths = []        
        with h5py.File(data_path, "r") as h5_file:
            print(f"Loading data from {data_path} Into memory")
            for demo in tqdm.tqdm(h5_file["data"].keys()):
                data_dict = {
                    k:torch.tensor(np.array(h5_file["data"][demo][k])).float()
                    for k in ['actions', 'dones', 'rewards', 'states']
                }
                data_dict.update({
                        k : torch.tensor(np.array(v)).float()
                        for k,v in h5_file["data"][demo]["obs"].items()
                    }
                )
                for k in rgb_processing_keys:
                    data_dict[k] = (data_dict[k].permute(0, 3, 1, 2)/255.0)

                self.data.append(data_dict)
                # we are dropping the next_obs as they can be derived from obs array
                self.seq_lengths.append(
                    h5_file["data"][demo].attrs['num_samples']
                )

    def get_seq_length(self, idx):
        return int(self.seq_lengths[idx])

    def __getitem__(self, idx):
        return (self.data[idx],
                None)

    def __len__(self):
        return len(self.seq_lengths)

    def load(self, data, key, is_global=False):
        return data
    

class RobomimicTrajectorySequenceDataset(TrajectoryDataset):
    def __init__(self, data_path, rgb_processing_keys):
        data_path = Path(data_path)
        self.data = []
        self.seq_lengths = []        
        # with h5py.File(data_path, "r") as h5_file:
        self._h5_file = h5py.File(data_path, "r")
        self._rgb_processing_keys = rgb_processing_keys
        for demo in self._h5_file["data"].keys():
            data_dict = {
                k:self._h5_file["data"][demo][k]
                for k in ['actions', 'dones', 'rewards', 'states']
            }
            data_dict.update({
                    k : v
                    for k,v in self._h5_file["data"][demo]["obs"].items()
                }
            )
            self.data.append(data_dict)
            # we are dropping the next_obs as they can be derived from obs array
            self.seq_lengths.append(
                self._h5_file["data"][demo].attrs['num_samples']
            )

    def get_seq_length(self, idx):
        return int(self.seq_lengths[idx])

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        return (data_dict,
                None)

    def __len__(self):
        return len(self.seq_lengths)

    def __del__(self):
        try:
            self._h5_file.close()
        except:
            pass # it's okay

    def load(self, data, key, is_global=False):
        data = torch.tensor(np.array(data)).float()
        if key in self._rgb_processing_keys:
            data = (data.permute(0, 3, 1, 2)/255.0)
        return data
    

class robomimic_train_val_test:

    def __init__(self,          
        data_path,
        rgb_processing_keys,
        preloaded,
        test_fraction,
        val_fraction,
        window_size_train,
        window_size_test,
        keys_traj,
        keys_global,
        pad_before,
        pad_after,
        pad_type,
        random_seed,
    ):
        if preloaded:
            self.sequence_dataset = RobomimicTrajectorySequenceDatasetPreloaded(
                data_path,
                rgb_processing_keys
            )
        else:
            self.sequence_dataset = RobomimicTrajectorySequenceDataset(
                data_path,
                rgb_processing_keys
            )
        self.train_data, self.val_data, self.test_data =  get_train_val_test_seq_datasets(
            self.sequence_dataset,
            test_fraction = test_fraction,
            val_fraction = val_fraction,
            window_size_train = window_size_train,
            window_size_test = window_size_test,
            keys_traj = keys_traj,
            keys_global = keys_global,
            pad_before = pad_before,
            pad_after = pad_after,
            pad_type = pad_type,
            random_seed = random_seed
        )
