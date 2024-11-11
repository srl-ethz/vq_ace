from .dataset_base import TrajectoryDataset, get_train_val_test_seq_datasets
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob

class FaiveTrajectorySequenceDataset(TrajectoryDataset):
    def __init__(self, data_directory, device="cuda", onehot_goals=False):
        data_directory = Path(data_directory)
        observations = []
        actions = []
        seqlengths = []
        print("Warning: the observation is termporarily set to action")
        for f in glob(str(data_directory / "*.hdf5")):
            with h5py.File(f, "r") as h5_file:
                front_view_images = h5_file['observations']['images']['front_view_color_image']
                top_view_color_images = h5_file['observations']['images']['top_view_color_image']
                wrist_view_color_images = h5_file['observations']['images']['wrist_view_color_image']
                action = torch.tensor(np.array(h5_file['action']))
                observations.append(action.to(device).float()) # This is temporary
                actions.append(action.to(device).float())
                seq_length = action.shape[0]
                # assert seq_length == top_view_color_images.shape[0] == wrist_view_color_images.shape[0] == action.shape[0]
                seqlengths.append(seq_length)
        self.observations = observations
        self.actions = actions
        self.seqlengths = seqlengths

    def get_seq_length(self, idx):
        return int(self.seqlengths[idx])

    def __getitem__(self, idx):
        return ({"action":self.actions[idx]}, 
                None)

    def __len__(self):
        return len(self.actions)

    def load(self, data, key, is_global=False):
        return data


def get_faive_train_val_test(
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
    return get_train_val_test_seq_datasets(
        FaiveTrajectorySequenceDataset(
            data_directory
        ),
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
