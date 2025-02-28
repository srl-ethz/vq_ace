from .dataset_base import TrajectoryDataset, get_train_val_test_seq_datasets
import torch
from torch.utils.data import Dataset
import ujson
import os
import pandas as pd
import numpy as np
import random
import time 
import h5py
import json
from typing import List


def get_numbers(values):
    numbers = []
    for sequence in values:
        for frame in sequence:
            frame = np.array(frame)
            numbers.append(frame)
    return numbers

        
class DexycbDatasetTemporal(Dataset):
    def __init__(self, json_folder_path, reduce_eval=False, normalize=True, augment=False, only_mano=True, print_stats=True):
        self.data = []
        self.path = json_folder_path
        self.lengths = []
        
        for path in os.scandir(self.path):
            if path.is_file():
                with h5py.File(path, "r") as file:
                    sequences_group = file["sequences"]

                    mean_group_mano = file['mean_mano']
                    self.mean_mano = mean_group_mano['data'][:]
                    std_group_mano = file['std_mano']   
                    self.std_mano = std_group_mano['data'][:]
                    mean_group_obj = file['mean_obj']
                    self.mean_obj = mean_group_obj['data'][:]
                    std_group_obj = file['std_obj']
                    self.std_obj = std_group_obj['data'][:]      
                    n = reduce_eval*5000 +1
                    for sequence_name in sequences_group:
                        demo = sequences_group[sequence_name]
                        features_array = []
                        if demo == []:
                            print('empty demo found')
                            continue
                        mano_sequences = demo['mano'][:].tolist()
                        self.lengths.append(len(mano_sequences))
                        object_sequences = demo['object'][:].tolist()
                        # ycb = demo['ycb_idx'][()]
                        # # One hot ecodinf of ycb_id
                        # ycb = np.eye(22)[ycb]
                        # ycb = torch.tensor(ycb, dtype=torch.float).unsqueeze(0).repeat(15,1)

                        for j, (mano, object) in enumerate(zip(mano_sequences, object_sequences)):
                            mano_array = np.array(mano)
                            object_array = np.array(object)
                            if normalize:
                                mano_array = (mano_array - self.mean_mano) / self.std_mano
                                object_array = (object_array - self.mean_obj) / self.std_obj
                            if mano_array.shape[0] < 15:
                                continue
                            if only_mano:
                                features = mano_array
                            else:
                                features = np.concatenate((mano_array,object_array), axis=1)
                            features_array.append(np.float32(features))
                        tuples = self.sample_tuples(features_array, D=5)
                        self.data = self.data + tuples
                    if print_stats:
                        print('average length of demos: ', np.mean(self.lengths))
                        print('max length of demos: ', np.max(self.lengths))
                        print('min length of demos: ', np.min(self.lengths))
                        print('std length of demos: ', np.std(self.lengths))
                        print('number of tuples: ', len(self.data))
                        print('number of demos: ', len(sequences_group))
                        array = [demo[0][0] for demo in self.data]
                        numbers = get_numbers(array)
                        print('mean of normalized mano: ', np.mean(np.mean(numbers[:67], axis=0)))
                        print('std of normalized mano: ', np.mean(np.std(numbers[:67], axis=0)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return ({"data": self.data[index]}, None)

    def sample_tuples(self, array : List, D=5):
        n = len(array)
        indices = list(range(n))
        random.shuffle(indices)
        
        tuples = []
        visited = set()
        for running_idx,i in enumerate(indices):
            if i not in visited:
                visited.add(i)
                current_tuple = [torch.tensor(array[i], dtype=torch.float)]
                for j in indices[running_idx+1:]:
                    if j not in visited and np.abs(j-i) <= D:
                        visited.add(j)
                        current_tuple.append(torch.tensor(array[j], dtype=torch.float))
                        tuples.append([tuple(current_tuple),torch.tensor(np.abs(j-i))])
                        break            
                
        return tuples
    

class DexycbDataset(Dataset):
    def __init__(self, json_folder_path, reduce_eval=False, normalize=False, augment=False, only_mano=True, print_stats=True, remove_outliers=False):
        self.data = []
        self.ycb_idxs = []  
        self.path = json_folder_path
        self.lengths = []

        for path in os.scandir(self.path):
            if path.is_file():
                with h5py.File(path, "r") as file:
                    sequences_group = file["sequences"]

                    mean_group_mano = file['mean_mano']
                    self.mean_mano = mean_group_mano['data'][:]
                    std_group_mano = file['std_mano']   
                    self.std_mano = std_group_mano['data'][:]
                    mean_group_obj = file['mean_obj']
                    self.mean_obj = mean_group_obj['data'][:]
                    std_group_obj = file['std_obj']
                    self.std_obj = std_group_obj['data'][:]      
                    for i,sequence_name in enumerate(sequences_group):
                        if reduce_eval == True and i%10 != 0:
                            continue 
                        demo = sequences_group[sequence_name]
                        features_array = []
                        if demo == []:
                            print('empty demo found')
                            continue
                        mano_sequences = demo['mano'][:].tolist()
                        self.lengths.append(len(mano_sequences))
                        object_sequences = demo['object'][:].tolist()
                        # ycb = demo['ycb_idx'][()]
                        # # One hot ecodinf of ycb_id
                        # ycb_onehot = np.eye(22)[ycb.copy()]
                        # ycb_onehot = torch.tensor(ycb_onehot, dtype=torch.float).unsqueeze(0).repeat(15,1)

                        for j, (mano, object) in enumerate(zip(mano_sequences, object_sequences)):
                            mano_array = np.array(mano)
                            object_array = np.array(object)
                            if normalize:
                                # Normalize just the last 9 components of mano (position + 6d rotation), keep the rest as is to avoid too high dependency on the retargeter parameters
                                mano_array[:,-9:] = (mano_array[:,-9:] - self.mean_mano[-9:]) / self.std_mano[-9:]
                                # mano_array = (mano_array - self.mean_mano) / self.std_mano
                                object_array = (object_array - self.mean_obj) / self.std_obj
                                if remove_outliers and np.any(np.abs(mano_array) > 2):
                                    continue
                            if mano_array.shape[0] < 15:
                                print('short demo found')
                                continue
                            # features = np.concatenate((mano_array,object_array,ycb_onehot), axis=1)
                            features = np.concatenate((mano_array,object_array), axis=1)
                            if only_mano:
                                features = mano_array
                            self.data.append(torch.tensor(np.array(features), dtype=torch.float))
                            # self.ycb_idxs.append(torch.tensor(np.array(ycb), dtype=torch.float))
                            if augment:
                                features = np.flip(features, axis=0).copy()
                                self.data.append(torch.tensor(np.array(features), dtype=torch.float))


                    if print_stats:
                        print('average length of demos: ', np.mean(self.lengths))
                        print('max length of demos: ', np.max(self.lengths))
                        print('min length of demos: ', np.min(self.lengths))
                        print('std length of demos: ', np.std(self.lengths))
                        print('number of tuples: ', len(self.data))
                        print('number of demos: ', len(sequences_group))

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return ({"data": self.data[index], 
                # "ybc_idx":self.ycb_idxs[index]
                }, None)

def get_dexycb_train_val_test(
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

