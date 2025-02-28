from .dataset_base import TrajectoryDataset, get_train_val_test_seq_datasets
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob

class RokokoTrajectorySequenceDataset(TrajectoryDataset):
    def __init__(self, data_directory, device="cuda"):
        data_directory = Path(data_directory)
        joint_angles = []
        seqlengths = []
        rolling_concat_factors = torch.tensor([1, 0.5, 0.5,  0.5, 0.5,  0.5, 0.5,  0.5, 0.5,  0.5, 0.5])
        for f in glob(str(data_directory / "*.hdf")):
            with h5py.File(f, "r") as h5_file:
                joints = torch.tensor(np.array(h5_file['joint_angles']))
                seq_length = joints.shape[0]
                seqlengths.append(seq_length)
                # change the scale from degree to radian
                joints = joints * np.pi / 180
                joints *= rolling_concat_factors.unsqueeze(0)
                joint_angles.append(joints.to(device).float())


        self.joint_angles = joint_angles
        self.seqlengths = seqlengths

    def get_seq_length(self, idx):
        return int(self.seqlengths[idx])

    def __getitem__(self, idx):
        return ({"joints":self.joint_angles[idx], 
                },
                None)

    def __len__(self):
        return len(self.joint_angles)


class rokoko_train_val_test:

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
        self.sequence_dataset = RokokoTrajectorySequenceDataset(
            data_directory
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
