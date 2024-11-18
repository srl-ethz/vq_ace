import datetime

import numpy as np

from pathlib import Path
import os
import hydra
from copy import deepcopy
from typing import Any, Dict, List, Optional
from ..algo.base_algo import Algo, TrainerMixin
from .pipeline_base import Pipeline, AlgoMixin, WandbMixin, DatasetMixin, Lr_SchedulerMixin, NormalizationMixin
from .vis_mixins.mujoco_visulizer import MujocoVisualizerMixin
from .data_augmentation import DataAugmentationMixin

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


class TrainPipeline(Pipeline, AlgoMixin, DatasetMixin, Lr_SchedulerMixin, WandbMixin, MujocoVisualizerMixin, DataAugmentationMixin, NormalizationMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.algo, Algo), "algo should be an instance of Algo"
        assert isinstance(self.algo, TrainerMixin), "algo should be an instance of TrainerMixin"
        if self.resume and self.resume_path is not None:
            self.load_checkpoint(self.resume_path)

    def _init_workspace(self, **cfg):
        super()._init_workspace(**cfg)
        self.training_config = cfg["training_cfg"]
        # setting up the training
        # set best losses for early stopping
        self.epoch = 1
        self.best_train_loss = 1e10
        self.best_eval_loss = 1e10


    def save_checkpoint(self, filepath):
        """Saves a checkpoint alowing to restart training from here
        """
        algo_state = self.algo._get_model_and_optimizer_states()
        checkpoint = {
            'algo_state': algo_state,
            'lr_schedulers': {k: v.state_dict() for k, v in self._lr_schedulers.items()},
            'epoch': self.epoch,
            'best_train_loss': self.best_train_loss,
            'best_eval_loss': self.best_eval_loss,
        }
        torch.save(checkpoint, filepath)

    def save_model(self, filepath):
        """Saves the final model
        """
        checkpoint = self.algo.serialize()
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """Loads a checkpoint to restart training from a previous point
        """
        checkpoint = torch.load(filepath)
        self.algo._load_model_and_optimizer_states(checkpoint['algo_state'])
        self.epoch = checkpoint['epoch']
        self.best_train_loss = checkpoint['best_train_loss']
        self.best_eval_loss = checkpoint['best_eval_loss']
        for k, v in self._lr_schedulers.items():
            v.load_state_dict(checkpoint['lr_schedulers'][k])

    def run(self):
        """This function is the main training function
        """

        print(
            f"Training params:\n - max_epochs: {self.training_config.num_epochs}\n"
            f" - checkpoint saving every: {self.training_config.steps_saving}\n"
        )
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)

        for epoch in range(1, self.training_config.num_epochs + 1):
            self.epoch = epoch
            metrics = {}
            
            self.algo.train_epoch_begin(epoch)
            print("training epoch", epoch)
            for inputs in tqdm(self.train_loader):
                batch, mask_batch = inputs
                batch, mask_batch = self.data_augmentation_train(batch, mask_batch)
                self.algo.train_step((batch, mask_batch), epoch)
            train_metrics = self.algo.train_epoch_end(epoch)
            metrics.update(train_metrics)

            if self.eval_loader is not None:
                self.algo.eval_epoch_begin(epoch)
                print("eval epoch", epoch)
                for batch, mask_batch in tqdm(self.eval_loader):
                    batch, mask_batch = self.data_augmentation_eval(batch, mask_batch)
                    self.algo.eval_step((batch, mask_batch), epoch)
                eval_metrics = self.algo.eval_epoch_end(epoch)
                metrics.update(eval_metrics)
            
            epoch_train_loss = metrics["train_epoch_loss"]
            epoch_eval_loss = metrics.get("eval_epoch_loss", epoch_train_loss)
            self._step_schedulers(metrics)

            wandb.log(metrics)
            
            update_best_model = False
            save_checkpoint = False

            if (
                epoch_eval_loss < self.best_eval_loss
                and self.eval_loader is not None
            ):
                self.best_eval_loss = epoch_eval_loss
                update_best_model = True

            elif (
                epoch_train_loss < self.best_train_loss
                and  self.eval_loader is None
            ):
                self.best_train_loss = epoch_train_loss
                update_best_model = True

            # save checkpoints
            if (
                self.training_config.steps_saving is not None
                and epoch % self.training_config.steps_saving == 0
            ):
                save_checkpoint = True

            if update_best_model:
                self.save_model(os.path.join(self.output_dir, "checkpoints", "best_model.pth"))
                print(f"Best model saved in {self.output_dir}.\n")
            if save_checkpoint:
                self.save_checkpoint(os.path.join(self.output_dir, "checkpoints", f"checkpoint_{epoch}.pth"))
                print(f"Checkpoint saved in {self.output_dir}.\n")

            if epoch % self.training_config.steps_visualize == 0:
                os.makedirs(os.path.join(self.output_dir, "visualizations", f"epoch_{epoch}"), exist_ok=True)
                batch_traj, batch_mask = next(iter(self.eval_loader))
                batch_traj, batch_mask = self.data_augmentation_eval(batch_traj, batch_mask)
                batch_traj = {k: v[:3] for k,v in batch_traj.items()}
                batch_mask = {k: v[:3] for k,v in batch_mask.items()}
                vis_list = []
                for i, joint_sequence in enumerate(batch_traj[self.mj_visualizer_key]):
                    vis_list.append({"origins": joint_sequence})
                for j in range(1):
                    recon_traj = self.algo.reconstruct(batch_traj, batch_mask)[self.mj_visualizer_key]
                    for i, joint_sequence in enumerate(recon_traj):
                        vis_list[i][f"recon_{j}"] = joint_sequence
                print("Visualizing")
                for i, vis_dict in enumerate(tqdm(vis_list)):
                    self.draw_gif_from_joint_sequence(vis_dict, 
                        output_path=os.path.join(self.output_dir, "visualizations", f"epoch_{epoch}", f"{i}.gif"),
                        wandb_log_name = f"test/vis/sample_{i}"
                    )
                
                    