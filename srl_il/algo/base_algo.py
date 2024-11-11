"""
This file contains base classes that other algorithm classes subclass.
"""
import textwrap
from copy import deepcopy
from collections import OrderedDict
from ..common.autoinit_class import AutoInit
from ..models.common.linear_normalizer import LinearNormalizer
import torch.nn as nn
import torch
from abc import ABC, abstractmethod
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from ..common.chunk_buffer import ChunkBufferBatch
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
        self._normalizers = nn.ModuleDict()
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
        self._normalizers.to(device)

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
            "normalizer": self._normalizers.state_dict()
        }

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self._models.load_state_dict(model_dict['model'])
        # self._normalizers.load_state_dict(model_dict['normalizer'])
        normalizer_keys = set()
        for full_key in model_dict['normalizer'].keys():
            parts = full_key.split('.')
            normalizer_keys.add(parts[0])
        for key in normalizer_keys:
            # Instantiate a new LinearNormalizer
            self._normalizers[key] = LinearNormalizer(
                model_dict['normalizer'][f'{key}.params_dict.offset'],
                model_dict['normalizer'][f'{key}.params_dict.scale'],
            )


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

    def train_epoch_begin(self, epoch):
        """
        Prepare for a new training epoch.
        """
        self.train_epoch_loss = 0
        self.train_epoch_loss_dict = {}
        self.train_epoch_count = 0
        self.set_device(self.device)
        self.set_train()

    def train_epoch_end(self, epoch):
        """
        End the current training epoch.
        Returns a dictionary of metrics.
        """
        epoch_loss = self.train_epoch_loss / self.train_epoch_count
        self.train_epoch_loss_dict = {k: v / self.train_epoch_count for k, v in self.train_epoch_loss_dict.items()}

        metric_dict = {'lr/'+k: v.param_groups[0]['lr'] for k, v in self._optimizers.items()}
        metric_dict['train_epoch_loss'] = epoch_loss
        metric_dict.update({'train_'+k: v for k, v in self.train_epoch_loss_dict.items()})
        return metric_dict

    @abstractmethod
    def train_step(self, inputs, epoch=None):
        """
        One step of training
        """
        pass

    def eval_epoch_begin(self, epoch):
        """
        Prepare for a new validation epoch.
        """
        self.eval_epoch_loss = 0
        self.eval_epoch_loss_dict = {}
        self.val_epoch_count = 0
    
    def eval_epoch_end(self, epoch):
        """
        End the current validation epoch.
        Returns a dictionary of metrics.
        """
        epoch_loss = self.eval_epoch_loss / self.val_epoch_count
        metric_dict =  {
            'eval_epoch_loss': epoch_loss
        }
        metric_dict.update({'eval_'+k: v / self.val_epoch_count for k, v in self.eval_epoch_loss_dict.items()})
        return metric_dict
    
    @abstractmethod
    def eval_step(self, inputs, epoch=None):
        """
        One step of validation
        """
        pass


class PolicyTranslatorBase(ABC):
    """
    Base class for translating the model output to the actions to execute.
    """
    @abstractmethod
    def action(self, model_output, batch=None):
        """
        Translate the model output to the actions
        """
        pass

class PolicyTranslatorDirect(PolicyTranslatorBase):
    """
    Used in the case where the action name is directly included in the model output.
    """
    def __init__(self, action_name):
        self.action_names = action_name
    def action(self, model_output, batch=None):
        return model_output[self.action_names]

class PolicyTranslator_6Drotation2Ang(PolicyTranslatorBase):
    """
    This is the action translator in diffusion policy code. 
    Source action: ee_pos(3) ee_rot(6) ee_gripper(1)
    Output action: ee_pos(3) ee_rot(3) ee_gripper(1)
    """
    def __init__(self, action_name):
        self.action_name = action_name
        from srl_il.common.rotation_transformer import RotationTransformer
        self.policy_rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

    def action(self, model_output, batch=None):
        action = model_output[self.action_name]
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.policy_rotation_transformer.inverse(rot)
        uaction = torch.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)
        return uaction


class PolicyMixin(ABC, AutoInit, cfgname_and_funcs=[("policy_cfg", "_init_policy")]):
    """
    Base Policy class that makes an Algo class a policy.
        A policy object is used for controlling the robot action given the current observation.
    The lifecycle of a policy in rollout is as follows:
        1. reset_policy() is called at the beginning of a rollout.
        2. predict_action() is called at each step of the rollout to get the action to take.
    The shapes of the policies contains the batch dimension.
    """

    def _init_policy(self, policy_bs, policy_obs_list, policy_translator):
        """
        policy_bs: batch size of the policy
        policy_obs_list: list of observation spaces, used for wrapping the history buffer.
            tuple of (name, length) where name is the name of the observation and length is the length of the history buffer.
        """
        self._policy_bs = policy_bs
        self._policy_obs_list = policy_obs_list
        self._policy_history_buffer = {}
        self._policy_translator = hydra.utils.instantiate(policy_translator)

    def _get_policy_observation(self, obs_dict):
        """
        Update the observation buffer.
        """
        obs_dict_return = {}
        obs_mask = {}
        for k, l in self._policy_obs_list:
            if k not in obs_dict:
                raise ValueError(f"Observation {k} not found in observation dictionary.")
            if k not in self._policy_history_buffer:
                self._policy_history_buffer[k] = ChunkBufferBatch(
                    batch_size=self._policy_bs,
                    data_shape=obs_dict[k].shape[1:],
                    chunk_length=l,
                    device=self.device
                )
            self._policy_history_buffer[k].append(obs_dict[k])
            obs, mask = self._policy_history_buffer[k].get_top()
            obs_dict_return[k] = obs
            obs_mask[k] = mask+1 # when dataloader's padding before is false, the first mask should never be false. Otherwise batch normalizer gives an nan
        return obs_dict_return, obs_mask
    
    def _translate_policy_output(self, model_output, batch=None):
        """
        Translate the model output to the actions
        """
        if self._policy_translator is not None:
            return self._policy_translator.action(model_output, batch)
        assert len(model_output) == 1, "Policy output should be a dictionary with a single key."
        return list(model_output.values())[0]
    
    def reset_policy(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._policy_history_buffer = {}

    def reset_policy_idx(self, idx):
        """
        Reset algo state to prepare for environment rollouts.
        """
        for k in self._policy_history_buffer:
            self._policy_history_buffer[k].reset_idx(idx)

    @abstractmethod
    def predict_action(self, obs_dict):
        """
        Compute the action to take in the current state.
        """
        pass
