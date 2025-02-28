"""
This file contains base classes that other algorithm classes subclass.
"""
import textwrap
from copy import deepcopy
from collections import OrderedDict
from ..common.autoinit_class import AutoInit
import torch.nn as nn
import torch
from abc import ABC, abstractmethod
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import hydra

class Algo(AutoInit, cfgname_and_funcs=(("algo_cfg", "_init_algo"),)):
    """
    Base algorithm class that all other algorithms subclass. Defines several
    functions that should be overriden by subclasses, in order to provide
    a standard API.
    It only contains the model and the device where the model should live.
    """

    def _init_algo(self, device):
        """
        Initialize the algorithm.
        """
        self._models = nn.ModuleDict()
        self._normalizer_means= {}
        self._normalizer_stds= {}
        self.set_device(device)

    @property
    def models(self):
        return self._models

    def set_device(self, device=None):
        """
        Set the device where the model should live.
        """
        device = self.device if device is None else device
        self.device = device
        self._models.to(device)
        
        self._normalizer_means = {k:v.to(device) for k,v in self._normalizer_means.items()}
        self._normalizer_stds= {k:v.to(device) for k,v in self._normalizer_stds.items()}

    def set_eval(self):
        """
        Prepare networks for evaluation.
        """
        self._models.eval()

    def set_train(self):
        """
        Prepare networks for training.
        """
        self._models.train()

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "model": self._models.state_dict(),
            "normalizer_means": self._normalizer_means,
            "normalizer_stds": self._normalizer_stds
        }

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self._models.load_state_dict(model_dict['model'])
        self._normalizer_means = model_dict['normalizer_means']
        self._normalizer_stds = model_dict['normalizer_stds']

    def __repr__(self):
        """
        Pretty print algorithm and network description.
        """
        return "{} (\n".format(self.__class__.__name__) + \
               textwrap.indent(self._models.__repr__(), '  ') + "\n)"


class TrainerMixin(ABC, AutoInit, cfgname_and_funcs=[("trainer_cfg", "_init_trainer")]):

    def _init_trainer(self, optimizer_cfg):
        """
        Initialize the trainer.
        """
        self._optimizers = OrderedDict()
        self._create_optimizers(optimizer_cfg)

    @property
    def optimizers(self):
        return self._optimizers

    @staticmethod
    def _create_optimizer(model_optm_cfg, model):
        """
        Creates an optimizer
        """
        optimizer_cls = hydra.utils.get_class(model_optm_cfg.optm_cls)
        optimizer_params = model_optm_cfg.get("optimizer_params", {})
        if optimizer_params is not None:
            optimizer = optimizer_cls(
                model.parameters(),
                lr=model_optm_cfg.lr,
                **optimizer_params,
            )
        return optimizer

    def _create_optimizers(self, optimizer_cfg):
        """
        Creates optimizers 
        """
        for model_name, model_optm_cfg in optimizer_cfg.items():
            self._optimizers[model_name] = self._create_optimizer(model_optm_cfg, self.models[model_name])

    def _optms_zero_grad(self):
        for optimizer in self._optimizers.values():
            optimizer.zero_grad()

    def _step_optimizers(self):
        for optimizer in self._optimizers.values():
            optimizer.step()

    def _get_model_and_optimizer_states(self):
        """
        Get the state of the model and optimizer.
        """
        model_state = self.serialize()
        optimizer_state = {k: v.state_dict() for k, v in self._optimizers.items()}
        return {'model': model_state, 'optimizer': optimizer_state}

    def _load_model_and_optimizer_states(self, states):
        """
        Load the state of the model and optimizer.
        """
        self.deserialize(states['model'])
        for k, v in self._optimizers.items():
            if k in states['optimizer']:
                v.load_state_dict(states['optimizer'][k])
            else:
                print(f"Warning: optimizer {k} not found in checkpoint")

    @abstractmethod
    def train_epoch_begin(self, epoch):
        """
        Prepare for a new training epoch.
        """
        pass

    @abstractmethod
    def train_epoch_end(self, epoch):
        """
        End the current training epoch.
        Returns a dictionary of metrics.
        """
        pass

    @abstractmethod
    def train_step(self, inputs, epoch=None):
        """
        One step of training
        """
        pass

    @abstractmethod
    def eval_epoch_begin(self, epoch):
        """
        Prepare for a new validation epoch.
        """
        pass


    @abstractmethod
    def eval_epoch_end(self, epoch):
        """
        End the current validation epoch.
        Returns a dictionary of metrics.
        """
        pass

    @abstractmethod
    def eval_step(self, inputs, epoch=None):
        """
        One step of validation
        """
        pass


class PolicyMixin(ABC, AutoInit, cfgname_and_funcs=[("policy_cfg", "_init_policy")]):
    """
    Base Policy class that makes an Algo class a policy.
        A policy object is used for controlling the robot action given the current observation.
    """

    def _init_policy(self):
        pass

    @abstractmethod
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        pass

    @abstractmethod
    def step(self, obs_dict):
        """
        Compute the action to take in the current state.
        """
        pass
