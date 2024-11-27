from .dataset_base import TrajectoryDataset, get_train_val_test_seq_datasets
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
# from sentence_transformers import SentenceTransformer

class FaiveTrajectorySequenceDataset(TrajectoryDataset):
    def __init__(self, data_directory):
        data_directory = Path(data_directory)
        self.traj_data = None # the data in sequence
        self.global_data = None # the information for the whole sequence
        self.seqlengths = []
        self._h5_files = [] 
        # sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Iterate over all the files to find that keys in all files
        for f in glob(str(data_directory / "*.h5")):
            h5_file =  h5py.File(f, "r")
            try:
                traj_data = {
                    key: h5_file[key] for key in ["actions_franka", "actions_hand"] if key in h5_file
                }
                # encode the task description
                global_data = {
                    # key: sentence_model.encode([h5_file[key][()]])  for key in ["task_description"] if key in h5_file
                }

                obs_group = h5_file["observations"]
                traj_data.update({
                    key:  obs_group[key] for key in obs_group if key != "images"
                })
                for img_name, img_data in obs_group.get("images", {}).items():
                    traj_data[f"{img_name}/color"] = img_data["color"]
                    if "extrinsics" in img_data:
                        global_data[f"{img_name}/extrinsics"] = img_data["extrinsics"]
                    if "intrinsics" in img_data:
                        global_data[f"{img_name}/intrinsics"] = img_data["intrinsics"]
                    if "projection" in img_data:
                        global_data[f"{img_name}/projection"] = img_data["projection"]
                self._h5_files.append(h5_file)
            except Exception as e:
                print(f"Warning: skipping {f}")
                print(e)
                continue
            
            if self.traj_data is None:
                self.traj_data = {key: [data] for key, data in traj_data.items()}
                self.global_data = {key: [data] for key, data in global_data.items()}

            else: # find the intersection of keys
                for key in list(self.traj_data.keys()):
                    if key in traj_data:
                        self.traj_data[key].append(traj_data[key])
                    else:
                        self.traj_data.pop(key)
                        print(f"Warning: key {key} not found in {f}")
                for key in list(self.global_data.keys()):
                    if key in global_data:
                        self.global_data[key].append(global_data[key])
                    else:
                        self.global_data.pop(key)
                        print(f"Warning: key {key} not found in {f}")
            self.seqlengths.append(len(list(traj_data.values())[0]))
        
    def get_seq_length(self, idx):
        return int(self.seqlengths[idx])

    def __getitem__(self, idx):
        return {k:v[idx] for k, v in self.traj_data.items()}, {k:v[idx] for k, v in self.global_data.items()}

    def __len__(self):
        return len(self.seqlengths)

    def __del__(self):
        try:
            for h5_file in self._h5_files:
                h5_file.close()
        except:
            pass # it's okay

    def load(self, data, key, is_global=False):
        if key == "task_description":
            return data
        data = torch.tensor(np.array(data)).float()
        if "color" in key:
            data = data.permute(0, 3, 1, 2) / 256.0
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
