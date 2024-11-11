from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_algo import Algo, TrainerMixin, PolicyMixin
from .mixin_obs import ObsEncoderMixin
from ..models.common.transformer import MIMO_Transformer
from ..models.common.position_encoding import PositionEncodingSineTable, SinusoidalTimeStepEmb, PositionEmbeddingSineSeq 
from torch.autograd import Variable
from ..models.common.vector_quantizer import VectorQuantizerEMA
from ..common.chunk_buffer import TemporalAggregationBuffer
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class Diffusion(Algo, ObsEncoderMixin):
    """
    A normal diffusion model that reconstructs the target
    """

    def _init_algo(self, device, target_dims, T_target,
                   network_is_causal, network_group_keys,
                   network_cfg, scheduler_cfg):
        """
        Initialize the algorithm.
        network_group_keys: List of observation group names that pass into the diffusion network (the conditional)
        """
        super()._init_algo(device)
        self.target_dims = target_dims
        self.T_target = T_target # action chunk size to predict
        self.network_group_keys = network_group_keys
        self.network_is_causal = network_is_causal
        self.network_cfg = network_cfg
        self.targets_idx_mapping = [0]
        for key, dim in self.target_dims.items():
            self.targets_idx_mapping.append(self.targets_idx_mapping[-1] + dim)


        if self.network_is_causal:
            self.target_tidx = list(np.array(range(T_target))) 
        else:
            self.target_tidx = [0] * T_target

        # this implementation currently follows the `encoder only BERT` branch in diffusion policy
        self._models["network"] = MIMO_Transformer(
            extra_output_shapes=[], # no output shape, as the de-noised target is read from the raw corresponding to the input
            **network_cfg
        )
        self._noise_scheduler_cfg = scheduler_cfg
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps = scheduler_cfg["num_train_timesteps"],
            beta_schedule = scheduler_cfg["beta_schedule"],
            clip_sample = scheduler_cfg["clip_sample"],
            beta_start = scheduler_cfg["beta_start"],
            variance_type = scheduler_cfg["variance_type"],
            beta_end = scheduler_cfg["beta_end"]
        )

        ##NOTE## self._models["obs_encoder"] will be constructed by the ObsEncoderMixin
        d_model = network_cfg['d_model']
        self._models_embeds = nn.ModuleDict()
        self._models_embeds["timestep_embed"] = SinusoidalTimeStepEmb(d_model)
        # self._models_embeds["target_pos_embed"] = nn.Embedding(T_target, d_model)
        self._models_embeds["timestep_pos_embed"] = nn.Embedding(1, d_model)
        self._models_embeds["whole_pos_embed"] = PositionEmbeddingSineSeq(temperature=10000, num_pos_feats=d_model)
        self._models["embeds"] = self._models_embeds

        # TODO: check if a non-linear projection is needed
        self._models_projs = nn.ModuleDict()
        a_dim = self.targets_idx_mapping[-1]
        self._models_projs["targets_to_token_proj"] = nn.Linear(a_dim, d_model) # project to transformer's dimension
        self._models_projs["token_to_target_proj"] = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, a_dim)) # project from transformer's dimension (diffusion policy have a layer norm before the output, so do I)
        self._models["projs"] = self._models_projs
        self.set_device(device)
        n_parameters = sum(p.numel() for p in self._models.parameters() if p.requires_grad)
        print("number of parameters: %.2fM" % (n_parameters/1e6,))



    def denoise_step(self, t, noisy_target, cond, target_mask=None):
        """
        args:
            t: The time index of the target
            noisy_target: (batch_size, T_target, a_dim) # assume has been normalized
            batch: A dict of observations. Trajectory and Global keys from dataset
            mask_batch: A dict of masks for trajectory keys. True means valid, False means padded.
            target_mask: (batch_size, T_target) mask for the target. True means valid, False means padded.
        return mu and log_var of the latent space
        inputs: (batch_size, seq_len, input_size)
        """
        bs = noisy_target.shape[0]
        # 1. time
        timesteps = t
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=noisy_target.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(noisy_target.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(noisy_target.shape[0])
        time_emb = self._models_embeds["timestep_embed"](timesteps).unsqueeze(1)
        timestep_pos_embed = self._models_embeds["timestep_pos_embed"].weight.unsqueeze(0).expand((bs, -1, -1)) # (batch, 1, d_model)

        # 2. target
        target_tokens = self._models_projs["targets_to_token_proj"](noisy_target)
        # target_pos_embed = self._models_embeds["target_pos_embed"].weight.unsqueeze(0).expand((bs, -1, -1))  # (batch, T_target, d_model)
        target_mask = target_mask if target_mask is not None else torch.ones_like(target_tokens[:, :, 0]) # (batch, tokens)

        # 3. condition
        obs_embeds, obs_posembs, obs_masks = cond

        target_begin_dim = obs_embeds.shape[1] + 1
        src = torch.cat([obs_embeds, time_emb+timestep_pos_embed, target_tokens], dim=1)
        # pos_embeds = torch.cat([obs_posembs, torch.zeros_like(time_emb), target_pos_embed], dim=1)     # (batch, tokens, d_model)
        pos_embeds = self._models_embeds["whole_pos_embed"](src)
        masks = torch.cat([obs_masks, torch.ones_like(time_emb[:,:,0]), target_mask], dim=1)

        if self.network_is_causal:
            src_tidx = list(np.array(range(src.shape[1]))) 
        else:
            src_tidx = None
        out, _ = self._models["network"](src, pos_embeds, masks, src_tidx) # (batch, T_total, z_dim) key: mu, logvar

        return self._models_projs["token_to_target_proj"](out[:, target_begin_dim:, :])
    

    def denoise(self, batch, mask_batch, target_mask=None):
        bs = list(mask_batch.values())[0].shape[0]
        cond = self.encode_obs(batch, mask_batch, self.network_group_keys) #(obs_embeds, obs_posembs, obs_masks) # (batch, tokens, d_model)

        target = torch.randn(
            size=[bs, self.T_target, self.targets_idx_mapping[-1]], 
            dtype=torch.float32,
            device=self.device
        )
        self.noise_scheduler.set_timesteps(self._noise_scheduler_cfg["num_inference_steps"])
        for t in self.noise_scheduler.timesteps:
            model_output = self.denoise_step(t, target, cond, target_mask=target_mask)
            
            target = self.noise_scheduler.step(model_output, t, target).prev_sample
        
        target_dict = {}
        for i, (key, dim) in enumerate(self.target_dims.items()):
            target_dict[key] = self._normalizers[key].denormalize(
                target[:, :, self.targets_idx_mapping[i]:self.targets_idx_mapping[i+1]]
            )

        return target_dict

class DiffusionPolicy(Diffusion, PolicyMixin):
    def _init_policy(self, policy_bs, policy_obs_list, update_freq, policy_translator=None):
        super()._init_policy(policy_bs, policy_obs_list, policy_translator)
        self._poilcy_update_freq = update_freq

    def reset_policy(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        super().reset_policy()
        self._policy_step_cnt = self._poilcy_update_freq + 1
        self._policy_action = None
        
    def predict_action(self, obs_dict):
        if self._policy_step_cnt >=self._poilcy_update_freq:
            with torch.no_grad():
                obs_dict, mask_batch = self._get_policy_observation(obs_dict)
                policy_output = self.denoise(obs_dict, mask_batch)
                self._policy_action = self._translate_policy_output(policy_output, obs_dict)
            self._policy_step_cnt = 0
        ret = self._policy_action[:, self._policy_step_cnt] 
        self._policy_step_cnt += 1
        return ret


class DiffusionPolicyTrainer(DiffusionPolicy, TrainerMixin):
    def _init_trainer(self, loss_params, optimizer_cfg):
        super()._init_trainer(optimizer_cfg)
        self._loss_params = loss_params


    def _compute_losses(self, batch):

        batch, mask_batch = batch # mask_batch: True means valid, False means padded
        batch = {k: v.to(self.device) for k, v in batch.items()}
        mask_batch = {k: v.to(self.device) for k, v in mask_batch.items()}

        target_mask = mask_batch[list(self.target_dims.keys())[0]] #(batch_size, T_target) 1 means valid, 0 means padded
        for key in list(self.target_dims.keys())[1:]:
            target_mask = target_mask * mask_batch[key]

        bs = target_mask.shape[0]
        # concatenae and normalize the target
        target = torch.zeros((bs, self.T_target, self.targets_idx_mapping[-1]), device=self.device)
        for i, (key, dim) in enumerate(self.target_dims.items()):
            target[:, :, self.targets_idx_mapping[i]:self.targets_idx_mapping[i+1]] = self._normalizers[key].normalize(batch[key]) 

        obs_cond = self.encode_obs(batch, mask_batch, self.network_group_keys) # (batch, tokens, d_model)

        # add noise
        noise = torch.randn_like(target)
        timesteps = torch.randint(0, self._noise_scheduler_cfg["num_train_timesteps"], (bs,), device=self.device).long()
        noisy_target = self.noise_scheduler.add_noise(
            target, noise, timesteps)

        pred = self.denoise_step(timesteps, noisy_target, obs_cond, target_mask)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            gt = noise
        elif pred_type == "sample":
            gt = target
        else:
            raise ValueError(f"Unknown prediction type {pred_type}")

        loss_dict = dict()
        # all_l1 = F.l1_loss(pred, gt, reduction='none')
        # l1 = (all_l1 * mask.unsqueeze(-1)).mean()
        # loss_dict['loss_l1'] = l1
        all_l2 = F.mse_loss(pred, gt, reduction='none')
        l2 = (all_l2 * target_mask.unsqueeze(-1)).mean()
        loss_dict['loss_l2'] = l2
        total_loss = l2
        return total_loss, loss_dict

    def train_step(self, batch, epoch=None):
        """
        One step of training
        """
        # parse the batch from the dataloader
        loss, loss_dict = self._compute_losses(batch)
        if loss != loss:
            raise ArithmeticError("NaN detected in train loss")

        self._optms_zero_grad()
        loss.backward()
        self._step_optimizers()

        # summarise the train step
        batch, mask_batch = batch # mask_batch: True means valid, False means padded
        bs = batch[list(self.target_dims.keys())[0]].shape[0]
        self.train_epoch_loss += loss.item() * bs
        for k,v in loss_dict.items():
            self.train_epoch_loss_dict[k] = self.train_epoch_loss_dict.get(k, 0) + v.item() * bs
        self.train_epoch_count += bs

    def eval_step(self, batch, epoch=None):
        """
        One step of validation
        """
        self.set_eval()
        # parse the batch from the dataloader
        loss, loss_dict = self._compute_losses(batch)
        if loss != loss:
            raise ArithmeticError("NaN detected in train loss")
        
        batch, mask_batch = batch # mask_batch: True means valid, False means padded
        bs = batch[list(self.target_dims.keys())[0]].shape[0]
        self.eval_epoch_loss += loss.item() * bs
        for k,v in loss_dict.items():
            self.eval_epoch_loss_dict[k] = self.eval_epoch_loss_dict.get(k, 0) + v.item() * bs
        self.val_epoch_count += bs

