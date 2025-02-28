from .dataset_base import TrajectoryDataset, get_train_val_test_seq_datasets
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob

class EmbedFaiveTrajectorySequenceDataset(TrajectoryDataset):
    def __init__(self, data_directory, actuator_names, qpos_joint_names, device="cuda"):
        data_directory = Path(data_directory)
        actions = []
        q_poses = []
        seqlengths = []
        for f in glob(str(data_directory / "*.h5")):
            with h5py.File(f, "r") as h5_file:
                action = torch.tensor(np.array(h5_file['data/ctrl']))
                actions.append(action.to(device).float())
                seq_length = action.shape[0]
                seqlengths.append(seq_length)
                data_actuator_names = [n.decode("utf-8") if  isinstance(n, bytes) else n for n in list(h5_file['info/actuator_names'])]
                for i in range(len(actuator_names)):
                    assert data_actuator_names[i] == actuator_names[i], f"actuator names do not match: {data_actuator_names[i]} != {actuator_names[i]}"
                
                q_pos = torch.tensor(np.array(h5_file['data/q_pos']))
                data_joint_names = [n.decode("utf-8") if  isinstance(n, bytes) else n for n in list(h5_file['info/joint_names'])]
                jnt_qposadr = list(h5_file['info/jnt_qposadr']) + [q_pos.shape[1],]

                q_pos_indexes = []
                for target_name in qpos_joint_names:
                    i = data_joint_names.index(target_name)
                    q_pos_indexes.extend(range( jnt_qposadr[i]  , jnt_qposadr[i+1])) 
                
                q_pos = q_pos[:, q_pos_indexes]
                q_poses.append(q_pos.to(device).float())
                

        self.q_poses = q_poses
        self.actions = actions
        self.seqlengths = seqlengths

    def get_seq_length(self, idx):
        return int(self.seqlengths[idx])

    def __getitem__(self, idx):
        return ({"action":self.actions[idx], 
                 "qpos":self.q_poses[idx], 
                },
                None)

    def __len__(self):
        return len(self.actions)


class embed_faive_train_val_test:

    def __init__(self,          
        data_directory,
        actuator_names,
        qpos_joint_names,
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
        self.sequence_dataset = EmbedFaiveTrajectorySequenceDataset(
            data_directory,
            actuator_names,
            qpos_joint_names
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
