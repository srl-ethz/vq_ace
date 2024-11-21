from .dataset_base import TrajectoryDataset, get_train_val_test_seq_datasets
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob

class FaiveTrajectorySequenceDataset(TrajectoryDataset):
    def __init__(self, data_directory, onehot_goals=False):
        data_directory = Path(data_directory)
        qpos_hand = []
        qpos_franka = []
        oakd_front_view_images = []
        actions_hand = []
        actions_franka = []
        seqlengths = []
        for f in glob(str(data_directory / "*.h5")):
            with h5py.File(f, "r") as h5_file:
                actions_hand.append(torch.tensor(np.array(h5_file['actions_hand'])).float())
                actions_franka.append(torch.tensor(np.array(h5_file['actions_franka'])).float())
                qpos_hand.append(torch.tensor(np.array(h5_file['observations/qpos_hand'])).float())
                qpos_franka.append(torch.tensor(np.array(h5_file['observations/qpos_franka'])).float())
                oakd_front_view_images.append(torch.tensor(np.array(h5_file['observations/images/oakd_front_view/color'])).float().permute(0, 3, 1, 2)/255.0)
                seq_length = actions_hand[-1].shape[0]
                # assert seq_length == top_view_color_images.shape[0] == wrist_view_color_images.shape[0] == action.shape[0]
                seqlengths.append(seq_length)
        self.actions_hand = actions_hand
        self.actions_franka = actions_franka
        self.qpos_hand = qpos_hand
        self.qpos_franka = qpos_franka
        self.oakd_front_view_images = oakd_front_view_images
        self.seqlengths = seqlengths

    def get_seq_length(self, idx):
        return int(self.seqlengths[idx])

    def __getitem__(self, idx):
        return ({"actions_hand":self.actions_hand[idx],
                 "actions_franka":self.actions_franka[idx],
                 "qpos_hand":self.qpos_hand[idx],
                 "qpos_franka":self.qpos_franka[idx],
                 "oakd_front_view_images":self.oakd_front_view_images[idx]}, 
                None)

    def __len__(self):
        return len(self.actions_franka)

    def load(self, data, key, is_global=False):
        return data


class faive_train_val_test:
    def __init__(self,          
            data_directory,
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
        self.sequence_dataset = FaiveTrajectorySequenceDataset(data_directory)
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
