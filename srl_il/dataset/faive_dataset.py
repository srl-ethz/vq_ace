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
    """
    NOTE: this dataset implementation is ugly, but it is efficient.
    When working with multiple data workers, make sure these workers do not share the same file handles.
    I.e. open the files in __getitem__() instead of __init__
    """
    def __init__(self, data_directory, cache_data_size=1e6):
        """
        cache_data_size: the number of the data (numel) of the key (whole trajectory) that is placed in memory instead of in the hdf file
        """
        data_directory = Path(data_directory)
        self.traj_data = None # the data in sequence
        self.global_data = None # the information for the whole sequence
        self.seqlengths = []
        self._h5_files = {} # file handlers
        self.cache_data_size = cache_data_size
        # sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Iterate over all the files to find that keys in all files
        for f in glob(str(data_directory / "*.h5")):
            with h5py.File(f, "r", swmr=True, libver='latest') as h5_file:
                # check what datasets are there in this file
                try:
                    traj_data = {
                        key: (h5_file[key], f, h5_file[key].name) for key in ["actions_franka", "actions_hand"] if key in h5_file
                    }
                    # encode the task description
                    global_data = {
                        # key: sentence_model.encode([h5_file[key][()]])  for key in ["task_description"] if key in h5_file
                    }

                    obs_group = h5_file["observations"]
                    traj_data.update({
                        key:  (obs_group[key], f, obs_group[key].name) for key in obs_group if key != "images"
                    })
                    for img_name, img_data in obs_group.get("images", {}).items():
                        traj_data[f"{img_name}/color"] = (img_data["color"], f, img_data["color"].name)
                        if "extrinsics" in img_data:
                            global_data[f"{img_name}/extrinsics"] = (img_data["extrinsics"], f, img_data["extrinsics"].name)
                        if "intrinsics" in img_data:
                            global_data[f"{img_name}/intrinsics"] = (img_data["intrinsics"], f, img_data["intrinsics"].name)
                        if "projection" in img_data:
                            global_data[f"{img_name}/projection"] = (img_data["projection"], f, img_data["projection"].name)
                except Exception as e:
                    print(f"Warning: skipping {f}")
                    print(e)
                    continue
            
                self.seqlengths.append(len(list(traj_data.values())[0][0]))
                # populate self.traj_data, it's either cached data or a file path and dataset path
                if self.traj_data is None:
                    self.traj_data = {key: [self._write_cache_or_not(data)]
                        for key, data in traj_data.items()}
                    self.global_data = {key: [self._write_cache_or_not(data)]
                        for key, data in global_data.items()}

                else: # find the intersection of keys
                    for key in list(self.traj_data.keys()):
                        if key in traj_data:
                            self.traj_data[key].append(self._write_cache_or_not(traj_data[key]))
                        else:
                            self.traj_data.pop(key)
                            print(f"Warning: key {key} not found in {f}")
                    for key in list(self.global_data.keys()):
                        if key in global_data:
                            self.global_data[key].append(self._write_cache_or_not(global_data[key]))
                        else:
                            self.global_data.pop(key)
                            print(f"Warning: key {key} not found in {f}")
    
    def _write_cache_or_not(self, data_and_name):
        data, file_name, data_name = data_and_name
        
        if data.size < self.cache_data_size:
            return data[()] # return the data loaded into memory
        else:
            return (file_name, data_name) # return the path to the h5 file and the data name
    
    def _read_cache_or_not(self, data_or_name):
        if type(data_or_name)==tuple:
            file_name, data_name = data_or_name
            if file_name not in self._h5_files:
                self._h5_files[file_name] = h5py.File(file_name, "r", swmr=True, libver='latest')
            return self._h5_files[file_name][data_name] # read the reference from h5
        else:
            return np.array(data_or_name)
            
    def get_seq_length(self, idx):
        return int(self.seqlengths[idx])

    def __getitem__(self, idx):
        return ({k:self._read_cache_or_not(v[idx]) for k, v in self.traj_data.items()}, 
                {k:self._read_cache_or_not(v[idx]) for k, v in self.global_data.items()})

    def __len__(self):
        return len(self.seqlengths)

    def __del__(self):
        try:
            for f in list(self._h5_files.keys()):
                h5file = self._h5_files.pop(f)
                h5file.close()
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
