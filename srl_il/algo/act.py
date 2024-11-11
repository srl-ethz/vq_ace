"""
Implementation of ACT: Action Chunking with Transformers
"""
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_algo import Algo, TrainerMixin, PolicyMixin
from .mixin_obs import ObsEncoderMixin
from ..models.common.transformer import MIMO_Transformer
from ..models.common.position_encoding import PositionEncodingSineTable
import uuid
import os
from torch.autograd import Variable
from ..models.common.vector_quantizer import VectorQuantizerEMA
from ..common.chunk_buffer import TemporalAggregationBuffer

class ActionChunkingPolicyMixin(PolicyMixin):

    def _init_policy(self, policy_bs, policy_obs_list, k=0.01, policy_translator=None):
        """
        policy_bs: batch size of the policy
        policy_obs_list: list of observation spaces, used for wrapping the history buffer.
            tuple of (name, length) where name is the name of the observation and length is the length of the history buffer.
        k: the exponential decay factor for the temporal aggregation buffer
        """
        super()._init_policy(policy_bs, policy_obs_list, policy_translator)
        self._policy_temporal_aggregation_k = k

    def reset_policy(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        super().reset_policy()
        self._policy_temporal_aggregation_buffer = None

    def reset_policy_idx(self, idx):
        """
        Reset algo state to prepare for environment rollouts.
        """
        super().reset_policy_idx(idx)
        self._policy_temporal_aggregation_buffer.reset_idx(idx)

    def predict_action(self, obs_dict):
        """
        Compute the action to take in the current state.
        """
        with torch.no_grad():
            obs_dict, mask_batch = self._get_policy_observation(obs_dict)
            z = torch.zeros((self._policy_bs, self.T_z, self.z_dim), device=self.device)
            output =  self.decode(z, obs_dict, mask_batch)
            action_chunk = self._translate_policy_output(output)

            if self._policy_temporal_aggregation_buffer is None:
                self._policy_temporal_aggregation_buffer = TemporalAggregationBuffer(
                    self._policy_bs,
                    (action_chunk.shape[2],),
                    self.T_target,
                    max_timesteps=200,
                    device=self.device
                )
            self._policy_temporal_aggregation_buffer.append(action_chunk)
            actions_for_curr_step, mask = self._policy_temporal_aggregation_buffer.get_top() # batch, T, a_dim
            k = self._policy_temporal_aggregation_k
            n_preds = mask.shape[1]
            exp_weights = torch.exp(-k * torch.arange(n_preds, 0, -1)).unsqueeze(dim=0).repeat(self._policy_bs, 1).to(self.device)
            exp_weights = exp_weights * mask
            exp_weights = exp_weights / exp_weights.sum(dim=1, keepdim=True)
            exp_weights = exp_weights.unsqueeze(dim=2)
            action = (actions_for_curr_step * exp_weights).sum(dim=1)
        return action

class ACT(Algo, ObsEncoderMixin):
    """
    Normal ACT training.
    """

    def _init_algo(self, device, target_dims, z_dim, T_target, T_z, 
                   encoder_is_causal, decoder_is_causal,
                   encoder_group_keys, decoder_group_keys,
                   encoder_cfg, decoder_cfg):
        """
        Initialize the algorithm.
        encoder_group_keys: List of observation group names that pass into the act_encoder (the prior)
        decoder_group_keys: List of observation group names that pass into the act_decoder (the conditional)
        target_dims: A dict of target dimensions. Key is the name of the target, value is the dimension of the target. Currentl implementation is to concatenate the target
        """
        super()._init_algo(device)
        self.z_dim = z_dim # latent dimension
        self.target_dims = OrderedDict(target_dims)
        self.T_target = T_target # action chunk size to predict
        self.T_z = T_z # number of tokens in the latent space
        self.encoder_group_keys = encoder_group_keys
        self.decoder_group_keys = decoder_group_keys
        self.encoder_is_causal = encoder_is_causal
        self.decoder_is_causal = decoder_is_causal

        # for key, dim in self.target_dims.items():
        self.targets_idx_mapping = [0]
        for key, dim in self.target_dims.items():
            self.targets_idx_mapping.append(self.targets_idx_mapping[-1] + dim)

        if self.encoder_is_causal:
            self.z_tidx = list((np.array(range(T_z)) + 1) / T_z * (T_target-1))
        else:
            self.z_tidx = [0] * T_z
        self._models["act_encoder"] = MIMO_Transformer(
            extra_output_shapes=self._act_encoder_output_shapes(),
            **encoder_cfg
        )

        if self.decoder_is_causal:
            self.target_tidx = list(np.array(range(T_target))) 
        else:
            self.target_tidx = [0] * T_target
        self._models["act_decoder"] = MIMO_Transformer(
            extra_output_shapes=[(self.target_tidx, self.target_dims)],
            **decoder_cfg
        )

        # self._models["obs_encoder"] will be constructed by the ObsEncoderMixin
        self._models_embeds = nn.ModuleDict()
        self._models_projs = nn.ModuleDict()
        self._models_embeds["target_pos_embed"] = PositionEncodingSineTable(T_target, encoder_cfg['d_model'], freeze=True)
        self._models_projs["targets_to_token_proj"] = nn.Linear(self.targets_idx_mapping[-1], encoder_cfg['d_model'])
        self._models_projs["embed_to_token_proj"] = nn.Linear(z_dim, decoder_cfg['d_model'])
        self._models_embeds["z_embedding"] = nn.Embedding(T_z, decoder_cfg['d_model'])
        self._models["embeds"] = self._models_embeds
        self._models["projs"] = self._models_projs
        self.set_device(device)
        n_parameters = sum(p.numel() for p in self._models.parameters() if p.requires_grad)
        print("number of parameters: %.2fM" % (n_parameters/1e6,))


    def _act_encoder_output_shapes(self):
        return [(self.z_tidx, {"mu": self.z_dim, "logvar": self.z_dim})]
    
    def encode(self, batch, mask_batch, target_mask=None):
        """
        args:
            batch: A dict of observations. Trajectory and Global keys from dataset (Target is also inside the batch)
            mask_batch: A dict of masks for trajectory keys. True means valid, False means padded.
            target_mask: (batch_size, T_target) mask for the target. True means valid, False means padded.
        return mu and log_var of the latent space
        inputs: (batch_size, seq_len, input_size)
        """
        bs = batch[list(self.target_dims.keys())[0]].shape[0]
        target = torch.zeros((bs, self.T_target, self.targets_idx_mapping[-1]), device=self.device)
        for i, (key, dim) in enumerate(self.target_dims.items()):
            target[:, :, self.targets_idx_mapping[i]:self.targets_idx_mapping[i+1]] = self._normalizers[key].normalize(batch[key])

        target_tokens = self._models_projs["targets_to_token_proj"](target)
        target_pos_embed = self._models_embeds["target_pos_embed"](target) # (batch, T_target, d_model)
        target_mask = target_mask if target_mask is not None else torch.ones_like(target_tokens[:, :, 0]) # (batch, tokens)

        obs_embeds, obs_posembs, obs_masks = self.encode_obs(batch, mask_batch, self.encoder_group_keys) #(batch, tokens, d_model)

        src = torch.cat([obs_embeds, target_tokens], dim=1)
        pos_embeds = torch.cat([obs_posembs, target_pos_embed], dim=1)     # (batch, tokens, d_model)
        masks = torch.cat([obs_masks, target_mask], dim=1)

        if self.encoder_is_causal:
            src_tidx = [-1]*obs_embeds.shape[1] + self.target_tidx
        else:
            src_tidx = None
        _, out = self._models["act_encoder"](src, pos_embeds, masks, src_tidx) # (batch, T_z, z_dim) key: mu, logvar
        return out

    def decode(self, z: torch.Tensor, batch, mask_batch):
        """
        Decode the latent space, conditioned on the batch observation
        args:
            z: (batch_size, T_z, latent_dim)
            batch: A dict of observations. Trajectory and Global keys from dataset
            mask_batch: A dict of masks for trajectory keys. True means valid, False means padded.
        """
        bs = z.shape[0]
        batch["z"] = z
        obs_embeds, obs_posembs, obs_masks = self.encode_obs(batch, mask_batch, self.decoder_group_keys) #(batch, tokens, d_model)
        z = self._models_projs["embed_to_token_proj"](z)
        z_emb = self._models_embeds["z_embedding"].weight.unsqueeze(0).repeat(bs, 1, 1) # (batch, T_z, d_model)
        z_mask = torch.ones_like(z[:, :, 0])
        src = torch.cat([obs_embeds, z], dim=1)
        pos_embeds = torch.cat([obs_posembs, z_emb], dim=1)
        masks = torch.cat([obs_masks, z_mask], dim=1)

        if self.decoder_is_causal:
            src_tidx = [-1]*obs_embeds.shape[1] + self.z_tidx
        else:
            src_tidx = None
        _, act_decoder_out = self._models["act_decoder"](src, pos_embeds, masks, src_tidx)
        for key, v in act_decoder_out.items():
            act_decoder_out[key] = self._normalizers[key].denormalize(v)
        return act_decoder_out

    def reconstruct(self, batch, mask_batch, target_mask=None):
        """
        Reconstruct the target from the latent space
        """
        act_encoder_out = self.encode(batch, mask_batch, target_mask)
        mu = act_encoder_out['mu'] # (batch, T_z, z_dim)
        logvar = act_encoder_out['logvar'] # (batch, T_z, z_dim)
        z = reparametrize(mu, logvar)
        return self.decode(z, batch, mask_batch)

    def export_onnx(self, batch, batch_mask, datakeys, f):
        """
        This function is used to export the decoder to ONNX
        Although it's better to use dynamo_export, it do not support all of the operations
        So we use torch script based onnx exporter https://pytorch.org/docs/stable/onnx_torchscript.html
           Directly calling export will does the equivalent of torch.jit.trace() which records the static graph of the model. 
           So we need to test the model to makesure the static graph is correct.
        """
        import torch.onnx
        old_device = self.device
        self.set_eval()
        self.set_device("cpu") # prevent some gpu related errors
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch_mask = {k: v.to(self.device) for k, v in batch_mask.items()}
        bs = list(batch.values())[0].shape[0]
        z = torch.randn((bs, self.T_z, self.z_dim), device=self.device)

        input_names = ["z"]
        input_tuple = (z,)
        for key in datakeys:
            input_names.append(key)
            input_tuple += (batch[key],)
            if key in batch_mask.keys():
                input_names.append(key+"_mask")
                input_tuple += (batch_mask[key],)
        
        class dummyDecoder(nn.Module):
            def __init__(innerself, models):
                super().__init__()
                innerself._models = models
                
            def forward(innerself, z, *args):
                batch = {k: v for k, v in zip(input_names[1:], args) if not k.endswith("_mask")}
                batch_mask = {k[:-5]: v for k, v in zip(input_names[1:], args) if k.endswith("_mask")}
                return self.decode(z, batch, batch_mask)
        
        dummy = dummyDecoder(self._models)
        dynamic_axes = {k: {0: "batch"} for k in input_names}
        dynamic_axes.update({k: {0: "batch"} for k in self.target_dims.keys()})
        torch.onnx.export(dummy, input_tuple, f, verbose=False,
                          input_names=input_names, output_names=list(self.target_dims.keys()),
                          dynamic_axes=dynamic_axes,
                          export_params=True, opset_version=12, do_constant_folding=True,
                        )
        self.set_eval() # export_onnx operation affects the model's state
        self.set_device(old_device)


# The vector quantized version of ACT
class ACT_VQ(ACT):

    def _init_algo(self, device, target_dims, z_dim, T_target, T_z, 
                encoder_is_causal, decoder_is_causal,
                encoder_group_keys, decoder_group_keys,
                encoder_cfg, decoder_cfg, vq_cfg
        ):

        ACT._init_algo(self, device, target_dims, z_dim, T_target, T_z,
                encoder_is_causal, decoder_is_causal,
                encoder_group_keys, decoder_group_keys,
                encoder_cfg, decoder_cfg
        )

        self._models["vq"] = VectorQuantizerEMA(
            num_embeddings=vq_cfg['num_embeddings'],
            embedding_dim=z_dim,
            decay=vq_cfg['decay'],
            epsilon=1e-5
        )
        self.set_device(device)

    def _act_encoder_output_shapes(self):
        """
        This method is used in ACT._init_algo to define the output shapes of the act_encoder
        """
        return [(self.z_tidx, {"mu": self.z_dim})]

    def quantize(self, inputs):
        """
        Please note that the stop-gradient operation happens inside of the VectorQuantizerEMA
        args:
            inputs: (batch_size, ... , z_dim)
        return:
            quantized: (batch_size, ... , z_dim)
            loss: scalar
            perplexity: scalar
        """

        loss, quantized, perplexity, encodings = self._models["vq"](inputs)
        return quantized, loss, perplexity
    
    def reconstruct(self, batch, mask_batch, target_mask=None):
        """
        Reconstruct the target from the latent space
        """
        act_encoder_out = self.encode(batch, mask_batch, target_mask)
        mu = act_encoder_out['mu'] # (batch, T_z, z_dim)
        z, _, _ = self.quantize(mu)
        return self.decode(z, batch, mask_batch)

    def export_onnx(self, batch, batch_mask, datakeys, f):
        """
        This function is used to export the decoder to ONNX
        Although it's better to use dynamo_export, it do not support all of the operations
        So we use torch script based onnx exporter https://pytorch.org/docs/stable/onnx_torchscript.html
           Directly calling export will does the equivalent of torch.jit.trace() which records the static graph of the model. 
           So we need to test the model to makesure the static graph is correct.
        """
        import torch.onnx
        old_device = self.device
        self.set_eval()
        self.set_device("cpu") # prevent some gpu related errors
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch_mask = {k: v.to(self.device) for k, v in batch_mask.items()}
        bs = list(batch.values())[0].shape[0]
        embed_inds = torch.randint(0, self._models["vq"]._num_embeddings, (bs, self.T_z), device=self.device)

        input_names = ["embed_inds"]
        input_tuple = (embed_inds,)
        for key in datakeys:
            input_names.append(key)
            input_tuple += (batch[key],)
            if key in batch_mask.keys():
                input_names.append(key+"_mask")
                input_tuple += (batch_mask[key],)
        
        class dummyDecoder(nn.Module):
            def __init__(innerself, models):
                super().__init__()
                innerself._models = models
                
            def forward(innerself, embed_ind, *args):
                z = self._models["vq"]._embedding(embed_ind)
                batch = {k: v for k, v in zip(input_names[1:], args) if not k.endswith("_mask")}
                batch_mask = {k[:-5]: v for k, v in zip(input_names[1:], args) if k.endswith("_mask")}
                return self.decode(z, batch, batch_mask)
        
        dummy = dummyDecoder(self._models)
        dynamic_axes = {k: {0: "batch"} for k in input_names}
        dynamic_axes.update({k: {0: "batch"} for k in self.target_dims.keys()})
        torch.onnx.export(dummy, input_tuple, f, verbose=False,
                          input_names=input_names, output_names=list(self.target_dims.keys()),
                          dynamic_axes=dynamic_axes,
                          export_params=True, opset_version=12, do_constant_folding=True,
                        )
        self.set_eval() # export_onnx operation affects the model's state
        self.set_device(old_device)

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps

class ACTPolicy(ACT, ActionChunkingPolicyMixin):
    pass

class ACT_VQPolicy(ACT_VQ, ActionChunkingPolicyMixin):
    pass


class ACTTrainerMixin(TrainerMixin):
    def _init_trainer(self, loss_params, optimizer_cfg):
        super()._init_trainer(optimizer_cfg)
        self._loss_params = loss_params

    def _compute_losses(self, batch, mask_batch):
        # set the target mask to true where all the target is valid
        target_mask = mask_batch[list(self.target_dims.keys())[0]] #(batch_size, T_target) 1 means valid, 0 means padded
        for key in list(self.target_dims.keys())[1:]:
            target_mask = target_mask * mask_batch[key]
        act_encoder_out = self.encode(batch, mask_batch, target_mask)
        mu = act_encoder_out['mu'] # (batch, T_z, z_dim)
        logvar = act_encoder_out['logvar'] # (batch, T_z, z_dim)
        latent_sample = reparametrize(mu, logvar)

        reconstruct = self.decode(latent_sample, batch, mask_batch)

        loss_dict = dict()
        # the loss is averaged over the batch
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl =  klds.sum(-1).mean()
        loss_dict['loss_kl'] = kl
        total_loss = kl * self._loss_params['kl_weight']

        for key in self.target_dims.keys():
            all_l1 = F.l1_loss(batch[key], reconstruct[key], reduction='none')
            l1 = (all_l1 * target_mask.unsqueeze(-1)).mean()
            loss_dict['loss_l1_'+key] = l1
            all_l2 = F.mse_loss(batch[key], reconstruct[key], reduction='none')
            l2 = (all_l2 * target_mask.unsqueeze(-1)).mean()
            loss_dict['loss_l2_'+key] = l2
            total_loss += l1
        
        return total_loss, loss_dict

    def train_step(self, batch, epoch=None):
        """
        One step of training
        """
        # parse the batch from the dataloader
        batch, mask_batch = batch # mask_batch: True means valid, False means padded
        batch = {k: v.to(self.device) for k, v in batch.items()}
        mask_batch = {k: v.to(self.device) for k, v in mask_batch.items()}

        loss, loss_dict = self._compute_losses(batch, mask_batch)
        if loss != loss:
            raise ArithmeticError("NaN detected in train loss")

        self._optms_zero_grad()
        loss.backward()
        self._step_optimizers()

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
        batch, mask_batch = batch
        batch = {k: v.to(self.device) for k, v in batch.items()}
        mask_batch = {k: v.to(self.device) for k, v in mask_batch.items()}

        loss, loss_dict = self._compute_losses(batch, mask_batch)
        if loss != loss:
            raise ArithmeticError("NaN detected in train loss")

        bs = batch[list(self.target_dims.keys())[0]].shape[0]
        self.eval_epoch_loss += loss.item() * bs
        for k,v in loss_dict.items():
            self.eval_epoch_loss_dict[k] = self.eval_epoch_loss_dict.get(k, 0) + v.item() * bs
        self.val_epoch_count += bs


class ACTTrainer(ACT, ACTTrainerMixin):
    pass

class ACT_VQTrainer(ACT_VQ, ACTTrainerMixin):

    def train_step(self, batch, epoch=None):
        """
        One step of training
        """
        # parse the batch from the dataloader
        batch, mask_batch = batch # mask_batch: True means valid, False means padded
        batch = {k: v.to(self.device) for k, v in batch.items()}
        mask_batch = {k: v.to(self.device) for k, v in mask_batch.items()}

        loss, loss_dict = self._compute_losses(batch, mask_batch)
        if loss != loss:
            raise ArithmeticError("NaN detected in train loss")

        self._optms_zero_grad()
        loss.backward()
        self._step_optimizers()

        bs = batch[list(self.target_dims.keys())[0]].shape[0]
        self.train_epoch_loss += loss.item() * bs
        for k,v in loss_dict.items():
            self.train_epoch_loss_dict[k] = self.train_epoch_loss_dict.get(k, 0) + v.item() * bs
        self.train_epoch_count += bs

    def _compute_losses(self, batch, mask_batch):
        target_mask = mask_batch[list(self.target_dims.keys())[0]]
        for key in list(self.target_dims.keys())[1:]:
            target_mask = target_mask * mask_batch[key]

        act_encoder_out = self.encode(batch, mask_batch, target_mask)
        mu = act_encoder_out['mu'] # (batch, T_z, z_dim)
        latent_quantized, vq_loss, perplexity = self.quantize(mu)

        reconstruct = self.decode(latent_quantized, batch, mask_batch)

        # the loss is averaged over the batch
        loss_dict = dict()
        total_loss = 0
        for key in self.target_dims.keys():
            all_l1 = F.l1_loss(batch[key], reconstruct[key], reduction='none')
            l1 = (all_l1 * target_mask.unsqueeze(-1)).mean()
            loss_dict['loss_l1_'+key] = l1
            all_l2 = F.mse_loss(batch[key], reconstruct[key], reduction='none')
            l2 = (all_l2 * target_mask.unsqueeze(-1)).mean()
            loss_dict['loss_l2_'+key] = l2
            total_loss += l1

        commitment_loss = vq_loss["commitment_loss"]
        loss_dict['loss_vq_commitment'] = commitment_loss
        total_loss += commitment_loss * self._loss_params['vq_commitment_weight']
        loss_dict['perplexity'] = perplexity.mean()
        loss_dict['perplexity_over_cbsize'] = perplexity.mean() / self._models["vq"]._num_embeddings
        return total_loss, loss_dict
    
    def eval_step(self, batch, epoch=None):
        """
        One step of validation
        """
        self.set_eval()
        # parse the batch from the dataloader
        batch, mask_batch = batch
        batch = {k: v.to(self.device) for k, v in batch.items()}
        mask_batch = {k: v.to(self.device) for k, v in mask_batch.items()}

        loss, loss_dict = self._compute_losses(batch, mask_batch)
        if loss != loss:
            raise ArithmeticError("NaN detected in train loss")

        bs = batch[list(self.target_dims.keys())[0]].shape[0]
        self.eval_epoch_loss += loss.item() * bs
        for k,v in loss_dict.items():
            self.eval_epoch_loss_dict[k] = self.eval_epoch_loss_dict.get(k, 0) + v.item() * bs
        self.val_epoch_count += bs

class ACTPolicyTrainer(ACTPolicy, ACTTrainerMixin):
    pass
