from ..common.autoinit_class import AutoInit
from abc import ABC, abstractmethod
import hydra
from omegaconf import OmegaConf
import wandb
import torch
import random
import numpy as np
import os


def set_seed(seed: int):
    """
    Functions setting the seed for reproducibility on ``random``, ``numpy``,
    and ``torch``

    Args:

        seed (int): The seed to be applied
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Pipeline(AutoInit, cfgname_and_funcs=((None, "_init_workspace"),)):
    """
    Base pipeline class that all other pipelines subclass. 
    A pipeline is like a workspace where there is algo, data, and other components such as visualizer and virtual environment.
    Only the pipeline and its subclasses should define the run method.
    """
    
    def _init_workspace(self, seed=0, output_dir=None, debugrun=False, **kwargs):
        """
        Initialize the workspace configuration.
        """
        self.debugrun = debugrun
        self.resume = "resume_path" in kwargs.keys()
        self.output_dir = output_dir
        self._cfg = kwargs
        # if not self.resume:
        #     set_seed(seed)

    @abstractmethod
    def run(self):
        """
        Run the pipeline.
        """
        raise NotImplementedError()

class AlgoMixin(AutoInit, cfgname_and_funcs=(("algo_cfg", "_init_algo"),)):
    # create the algo
    def _init_algo(self, **algo_cfg):
        self.algo = hydra.utils.instantiate(algo_cfg,  _recursive_=False)


class WandbMixin(AutoInit, cfgname_and_funcs=((None, "_init_wandb"),)):
    def _init_wandb(self, **cfg):
        cfg = OmegaConf.create(cfg)
        self.wandb_project_name = cfg.wandb_cfg.project
        self.wandb_run_name = cfg.wandb_cfg.run_name
        tags = cfg.wandb_cfg.get("tags", None)
        mode = cfg.wandb_cfg.get("mode", "online")
        hydra_dir = self.output_dir
        if self.output_dir is None:
            mode = "disabled"
            hydra_dir = None
        self.wandb_run = wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            config = OmegaConf.to_container(cfg, resolve=True),
            dir=hydra_dir,
            tags = tags,
            mode = mode
        )
        self.wandb_run.log_code(".")


class DatasetMixin(AutoInit, cfgname_and_funcs=(("dataset_cfg", "_init_dataset"),)):
    """
    Base dataset mixin class that all other dataset mixins subclass. 
    A dataset mixin is a class that provides the dataset to the pipeline.
    """
    
    def _init_dataset(self, data, batch_size, pin_memory, num_workers):
        """
        Initialize the dataset configuration.
        """
        self.datasets = hydra.utils.instantiate(data)
        train_data = self.datasets.train_data
        val_data = self.datasets.val_data
        test_data = self.datasets.test_data
        if self.debugrun:
            train_data = torch.utils.data.Subset(train_data, range(len(train_data) // 10))
            val_data = torch.utils.data.Subset(val_data, range(len(val_data) // 10))
            test_data = torch.utils.data.Subset(test_data, range(len(test_data) // 10)) if test_data is not None else None
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers
        )
        self.eval_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers
        ) 
        self.test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers
        ) if test_data is not None else None
    
    def dataset_name_to_id_in_traj_batch(self, name):
        """
        Get the index of the dataset in the trajectory batch.
        """
        keys_traj = self.train_dataset.keys_traj
        for i, (k, k_start, k_end) in enumerate(keys_traj):
            if name == k:
                return i
        return None

    def dataset_name_to_id_in_glob_batch(self, name):
        """
        Get the index of the dataset in the trajectory batch.
        """
        keys_global = self.train_dataset.keys_global
        for i, (k, k_start, k_end) in enumerate(keys_global):
            if name == k:
                return i
        return None

class Lr_SchedulerMixin(AutoInit, cfgname_and_funcs=(("lr_scheduler_cfg", "_create_schedulers"),)):
    """
    Base lr_scheduler mixin class that all other lr_scheduler mixins subclass. 
    A lr_scheduler mixin is a class that provides the lr_scheduler to the pipeline.
    """
    
    def _create_schedulers(self, **lr_scheduler_cfg):
        self._lr_schedulers = {}
        self._lr_scheduler_cfg = lr_scheduler_cfg
        for model_name, model_scheduler_cfg in lr_scheduler_cfg.items():
            scheduler_cls = hydra.utils.get_class(model_scheduler_cfg.scheduler_cls)
            self._lr_schedulers[model_name] = scheduler_cls(
                self.algo._optimizers[model_name], **model_scheduler_cfg.get("params", {})
            )
        
    def _step_schedulers(self, metrics_dict):
        for k, scheduler in self._lr_schedulers.items():
            cfg = self._lr_scheduler_cfg[k]
            if scheduler is None:
                pass
            if cfg.get("step_with_metrics", False):
                metrics = metrics_dict[cfg["metrics_name"]]
                scheduler.step(metrics)
            else:
                scheduler.step()


#####
# Normlization Mixins
#####
class NormalizationMixin(AutoInit, cfgname_and_funcs=(("normalizer_cfg", "_init_normalizer"),)):
    """
    Normlization mixin class that all other normlization mixins subclass. 
    A normlization mixin set the normalizers of Algo
    Supported normalizatio types:
        - hardcode: hardcode the mean and std
        - dataset_minmax: normalize the data to [0, 1], can have a quantile parameter 0~1
        - dataset_standard: standardize the data to have mean 0 and std 1
    """
    
    def _hardcode_normalizer(self, type, mean=0.0, std=1.0, **kwargs):
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        for k in kwargs.keys():
            print(f"Warning: key {k} is not used in hardcode_normalizer")
        return mean, std

    def _dataset_minmax_normalizer(self, type, dataname, is_traj_key=True, quantile=1, **kwargs):
        """
        norm_cfg: configuration of the normalizations, with the following format:
            key_name: {type: , **kwargs}
        """
        seq_dataset = self.datasets.sequence_dataset # the dataset of all raw sequences
        datas = []
        for i in range(len(seq_dataset)):
            data_traj, data_glob = seq_dataset[i]
            if is_traj_key:
                data = data_traj[dataname]
            else:
                data = data_glob[dataname]
            datas.append(data)
        datas = torch.cat(datas, dim=0)
        if quantile == 1:
            min_val = torch.min(datas, dim=0).values
            max_val = torch.max(datas, dim=0).values
        else:
            min_val = torch.quantile(datas, 1-quantile, dim=0)
            max_val = torch.quantile(datas, quantile, dim=0)
        mean = (min_val + max_val) / 2
        std = (max_val - min_val) / 2
        for k in kwargs.keys():
            print(f"Warning: key {k} is not used in dataset_minmax_normalizer")
        return mean, std

    def _init_normalizer(self, **norm_cfg):
        """
        norm_cfg: configuration of the normalizations, with the following format:
            key_name: {type: , **kwargs}
        """
        if self.resume: # do not need to reinitialize the normalizers
            return

        for k,v in norm_cfg.items():
            if v["type"] == "hardcode":
                mean, std = self._hardcode_normalizer(**v)
            elif v["type"] == "dataset_minmax":
                mean, std = self._dataset_minmax_normalizer(**v)
            else:
                raise NotImplementedError(f"Normalization type {v['type']} is not implemented")
            self.algo._normalizer_means[k] = mean
            self.algo._normalizer_stds[k] = std
