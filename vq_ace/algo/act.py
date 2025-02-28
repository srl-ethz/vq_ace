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
from ..models.common.transformer import MIMO_TRANSENCODER
from ..models.common.position_encoding import PositionEncodingSine1D
import uuid
import os
from torch.autograd import Variable
from ..models.common.vector_quantizer import VectorQuantizerEMA

class ACT(Algo, ObsEncoderMixin):
    """
    Normal ACT training.
    """

    def _init_algo(self, device, a_dim, z_dim, T_target, T_z, 
                   encoder_is_causal, decoder_is_causal,
                   encoder_group_keys, decoder_group_keys,
                   encoder_cfg, decoder_cfg):
        """
        Initialize the algorithm.
        encoder_group_keys: List of observation group names that pass into the act_encoder (the prior)
        decoder_group_keys: List of observation group names that pass into the act_decoder (the conditional)
        """
        super()._init_algo(device)
        self.z_dim = z_dim # latent dimension
        self.a_dim = a_dim # action dimension
        self.T_target = T_target # action chunk size to predict
        self.T_z = T_z # number of tokens in the latent space
        self.encoder_group_keys = encoder_group_keys
        self.decoder_group_keys = decoder_group_keys
        self.encoder_is_causal = encoder_is_causal
        self.decoder_is_causal = decoder_is_causal
        if self.encoder_is_causal:
            self.z_tidx = list((np.array(range(T_z)) + 1) / T_z * (T_target-1))
        else:
            self.z_tidx = [0] * T_z
        self._models["act_encoder"] = MIMO_TRANSENCODER(
            output_shapes=self._act_encoder_output_shapes(),
            **encoder_cfg
        )

        if self.decoder_is_causal:
            self.target_tidx = list(np.array(range(T_target))) 
        else:
            self.target_tidx = [0] * T_target
        self._models["act_decoder"] = MIMO_TRANSENCODER(
            output_shapes=[(self.target_tidx, {"target": a_dim})],
            **decoder_cfg
        )

        # self._models["obs_encoder"] will be constructed by the ObsEncoderMixin
        self._models["target_pos_embed"] = PositionEncodingSine1D(T_target, encoder_cfg['d_model'], freeze=True)
        self._models["target_to_token_proj"] = nn.Linear(a_dim, encoder_cfg['d_model']) # project to act_encoder's dimension
        self._models["token_to_target_proj"] = nn.Linear(encoder_cfg['d_model'], a_dim) # project from act_encoder's dimension
        self._models["embed_to_token_proj"] = nn.Linear(z_dim, decoder_cfg['d_model'])
        self._models["z_embedding"] = nn.Embedding(T_z, decoder_cfg['d_model'])
        self.set_device(device)

    def _act_encoder_output_shapes(self):
        return [(self.z_tidx, {"mu": self.z_dim, "logvar": self.z_dim})]
    
    def encode(self, target, batch, mask_batch, target_mask=None):
        """
        args:
            target: (batch_size, T_target, a_dim)
            batch: A dict of observations. Trajectory and Global keys from dataset
            mask_batch: A dict of masks for trajectory keys. True means valid, False means padded.
            target_mask: (batch_size, T_target) mask for the target. True means valid, False means padded.
        return mu and log_var of the latent space
        inputs: (batch_size, seq_len, input_size)
        """
        bs = target.shape[0]
        means = self._normalizer_means['target']
        stds = self._normalizer_stds['target']
        target = (target - means)/stds
        target_tokens = self._models["target_to_token_proj"](target)
        target_pos_embed = self._models["target_pos_embed"](target) # (T_target, d_model)
        target_mask = target_mask if target_mask is not None else torch.ones_like(target_tokens[:, :, 0]) # (batch, tokens)

        obs_embeds, obs_posembs, obs_masks = self.encode_obs(batch, mask_batch, self.encoder_group_keys) #(batch, tokens, d_model)

        src = torch.cat([obs_embeds, target_tokens], dim=1)
        pos_embeds = torch.cat([obs_posembs, target_pos_embed], dim=0)     # (tokens, d_model)
        masks = torch.cat([obs_masks, target_mask], dim=1)

        if self.encoder_is_causal:
            src_tidx = [-1]*obs_embeds.shape[1] + self.target_tidx
        else:
            src_tidx = None
        return self._models["act_encoder"](src, pos_embeds, masks, src_tidx) # (batch, T_z, z_dim) key: mu, logvar
    

    def decode(self, z: torch.Tensor, batch, mask_batch):
        """
        Decode the latent space, conditioned on the batch observation
        args:
            z: (batch_size, 1, latent_dim)
            batch: A dict of observations. Trajectory and Global keys from dataset
            mask_batch: A dict of masks for trajectory keys. True means valid, False means padded.
        """
        bs = z.shape[0]
        batch["z"] = z
        obs_embeds, obs_posembs, obs_masks = self.encode_obs(batch, mask_batch, self.decoder_group_keys) #(batch, tokens, d_model)
        z = self._models["embed_to_token_proj"](z)
        z_emb = self._models["z_embedding"].weight
        z_mask = torch.ones_like(z[:, :, 0])
        src = torch.cat([obs_embeds, z], dim=1)
        pos_embeds = torch.cat([obs_posembs, z_emb], dim=0)
        masks = torch.cat([obs_masks, z_mask], dim=1)

        if self.decoder_is_causal:
            src_tidx = [-1]*obs_embeds.shape[1] + self.z_tidx
        else:
            src_tidx = None
        act_decoder_out = self._models["act_decoder"](src, pos_embeds, masks, src_tidx)
        target = act_decoder_out['target']

        means = self._normalizer_means.get('target', 0.0)
        stds = self._normalizer_stds.get('target', 1.0)
        target = target * stds + means

        return target

    def reconstruct(self, target, batch, mask_batch, target_mask=None):
        """
        Reconstruct the target from the latent space
        """
        act_encoder_out = self.encode(target, batch, mask_batch, target_mask)
        mu = act_encoder_out['mu'] # (batch, T_z, z_dim)
        logvar = act_encoder_out['logvar'] # (batch, T_z, z_dim)
        z = reparametrize(mu, logvar)
        print(z)
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
        dynamic_axes["output"] = {0: "batch"}
        torch.onnx.export(dummy, input_tuple, f, verbose=False,
                          input_names=input_names, output_names=["output"],
                          dynamic_axes=dynamic_axes,
                          export_params=True, opset_version=12, do_constant_folding=True,
                        )
        self.set_eval() # export_onnx operation affects the model's state
        self.set_device(old_device)


# The vector quantized version of ACT
class ACT_VQ(ACT):

    def _init_algo(self, device, a_dim, z_dim, T_target, T_z, 
                encoder_is_causal, decoder_is_causal,
                encoder_group_keys, decoder_group_keys,
                encoder_cfg, decoder_cfg, vq_cfg
        ):

        ACT._init_algo(self, device, a_dim, z_dim, T_target, T_z,
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
    
    def reconstruct(self, target, batch, mask_batch, target_mask=None):
        """
        Reconstruct the target from the latent space
        """
        act_encoder_out = self.encode(target, batch, mask_batch, target_mask)
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
        dynamic_axes["output"] = {0: "batch"}
        torch.onnx.export(dummy, input_tuple, f, verbose=False,
                          input_names=input_names, output_names=["output"],
                          dynamic_axes=dynamic_axes,
                          export_params=True, opset_version=12, do_constant_folding=True,
                        )
        self.set_eval() # export_onnx operation affects the model's state
        self.set_device(old_device)

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


class ACTTrainerMixin(TrainerMixin):
    def _init_trainer(self, action_name_in_batch, loss_params, optimizer_cfg):
        super()._init_trainer(optimizer_cfg)
        self._action_name_in_batch = action_name_in_batch
        self._loss_params = loss_params

    def train_epoch_begin(self, epoch):
        self.train_epoch_loss = 0
        self.train_epoch_loss_dict = {}
        self.train_epoch_count = 0
        self.set_device(self.device)
        self.set_train()

    def train_epoch_end(self, epoch):
        epoch_loss = self.train_epoch_loss / self.train_epoch_count
        self.train_epoch_loss_dict = {k: v / self.train_epoch_count for k, v in self.train_epoch_loss_dict.items()}

        metric_dict = {'lr/'+k: v.param_groups[0]['lr'] for k, v in self._optimizers.items()}
        metric_dict['train_epoch_loss'] = epoch_loss
        metric_dict.update({'train_'+k: v for k, v in self.train_epoch_loss_dict.items()})
        return metric_dict
    
    def _compute_losses(self, mu, logvar, target, reconstructions, target_mask):
        # the loss is averaged over the batch
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl =  klds.sum(-1).mean()

        loss_dict = dict()
        all_l1 = F.l1_loss(target, reconstructions, reduction='none')
        l1 = (all_l1 * target_mask.unsqueeze(-1)).mean()
        loss_dict['loss_l1'] = l1
        all_l2 = F.l2_loss(target, reconstructions, reduction='none')
        # l2 = (all_l2 * target_mask.unsqueeze(-1)).mean()
        # loss_dict['loss_l2'] = l2
        # loss_dict['loss_kl'] = kl
        total_loss = l1 + kl * self._loss_params['kl_weight']
        return total_loss, loss_dict

    def train_step(self, batch, epoch=None):
        """
        One step of training
        """
        # parse the batch from the dataloader
        batch, mask_batch = batch # mask_batch: True means valid, False means padded
        batch = {k: v.to(self.device) for k, v in batch.items()}
        mask_batch = {k: v.to(self.device) for k, v in mask_batch.items()}

        target = batch[self._action_name_in_batch] #(batch_size, T_target, a_dim)
        target_mask = mask_batch[self._action_name_in_batch] #(batch_size, T_target) 1 means valid, 0 means padded
        act_encoder_out = self.encode(target, batch, mask_batch, target_mask)
        mu = act_encoder_out['mu'] # (batch, T_z, z_dim)
        logvar = act_encoder_out['logvar'] # (batch, T_z, z_dim)
        latent_sample = reparametrize(mu, logvar)

        reconstruct = self.decode(latent_sample, batch, mask_batch)

        loss, loss_dict = self._compute_losses(mu, logvar, target, reconstruct, target_mask)
        if loss != loss:
            raise ArithmeticError("NaN detected in train loss")

        self._optms_zero_grad()
        loss.backward()
        self._step_optimizers()

        bs = target.shape[0]
        self.train_epoch_loss += loss.item() * bs
        for k,v in loss_dict.items():
            self.train_epoch_loss_dict[k] = self.train_epoch_loss_dict.get(k, 0) + v.item() * bs
        self.train_epoch_count += bs

    def eval_epoch_begin(self, epoch):
        self.eval_epoch_loss = 0
        self.eval_epoch_loss_dict = {}
        self.val_epoch_count = 0
    
    def eval_epoch_end(self, epoch):
        epoch_loss = self.eval_epoch_loss / self.val_epoch_count
        metric_dict =  {
            'eval_epoch_loss': epoch_loss
        }
        metric_dict.update({'eval_'+k: v / self.val_epoch_count for k, v in self.eval_epoch_loss_dict.items()})
        return metric_dict

    def eval_step(self, batch, epoch=None):
        """
        One step of validation
        """
        self.set_eval()
        # parse the batch from the dataloader
        batch, mask_batch = batch
        batch = {k: v.to(self.device) for k, v in batch.items()}
        mask_batch = {k: v.to(self.device) for k, v in mask_batch.items()}

        target = batch[self._action_name_in_batch] #(batch_size, T_target, a_dim)
        target_mask = mask_batch[self._action_name_in_batch] #(batch_size, T_target)
        act_encoder_out = self.encode(target, batch, mask_batch, target_mask)
        mu = act_encoder_out['mu'] # (batch, T_z, z_dim)
        logvar = act_encoder_out['logvar'] # (batch, T_z, z_dim)
        latent_sample = reparametrize(mu, logvar)

        reconstruct = self.decode(latent_sample, batch, mask_batch)

        loss, loss_dict = self._compute_losses(mu, logvar, target, reconstruct, target_mask)
        if loss != loss:
            raise ArithmeticError("NaN detected in train loss")

        bs = target.shape[0]
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

        target = batch[self._action_name_in_batch] #(batch_size, T_target, a_dim)
        target_mask = mask_batch[self._action_name_in_batch] #(batch_size, T_target) 1 means valid, 0 means padded
        act_encoder_out = self.encode(target, batch, mask_batch, target_mask)
        mu = act_encoder_out['mu'] # (batch, T_z, z_dim)
        latent_quantized, vq_loss, perplexity = self.quantize(mu)

        reconstruct = self.decode(latent_quantized, batch, mask_batch)

        loss, loss_dict = self._compute_losses(vq_loss, perplexity, target, reconstruct, target_mask)
        if loss != loss:
            raise ArithmeticError("NaN detected in train loss")

        self._optms_zero_grad()
        loss.backward()
        self._step_optimizers()

        bs = target.shape[0]
        self.train_epoch_loss += loss.item() * bs
        for k,v in loss_dict.items():
            self.train_epoch_loss_dict[k] = self.train_epoch_loss_dict.get(k, 0) + v.item() * bs
        self.train_epoch_count += bs

    def _compute_losses(self, vq_loss, perplexity, target, reconstructions, target_mask):
        info_dict = dict()
        all_l1 = F.l1_loss(target, reconstructions, reduction='none')
        l1 = (all_l1 * target_mask.unsqueeze(-1)).mean()
        all_l2 = F.l1_loss(target, reconstructions, reduction='none')
        l2 = (all_l2 * target_mask.unsqueeze(-1)).mean()
        commitment_loss = vq_loss["commitment_loss"]
        # info_dict['loss_l1'] = l1
        info_dict['loss_l2'] = l2
        info_dict['loss_vq_commitment'] = commitment_loss
        total_loss = l2 + commitment_loss * self._loss_params['vq_commitment_weight']
        info_dict['perplexity'] = perplexity.mean()
        info_dict['perplexity_over_cbsize'] = perplexity.mean() / self._models["vq"]._num_embeddings
        return total_loss, info_dict
    
    def eval_step(self, batch, epoch=None):
        """
        One step of validation
        """
        self.set_eval()
        # parse the batch from the dataloader
        batch, mask_batch = batch
        batch = {k: v.to(self.device) for k, v in batch.items()}
        mask_batch = {k: v.to(self.device) for k, v in mask_batch.items()}

        target = batch[self._action_name_in_batch] #(batch_size, T_target, a_dim)
        target_mask = mask_batch[self._action_name_in_batch] #(batch_size, T_target)
        act_encoder_out = self.encode(target, batch, mask_batch, target_mask)
        mu = act_encoder_out['mu'] # (batch, T_z, z_dim)
        latent_quantized, vq_loss, perplexity = self.quantize(mu)

        reconstruct = self.decode(latent_quantized, batch, mask_batch)

        loss, loss_dict = self._compute_losses(vq_loss, perplexity,  target, reconstruct, target_mask)
        if loss != loss:
            raise ArithmeticError("NaN detected in train loss")

        bs = target.shape[0]
        self.eval_epoch_loss += loss.item() * bs
        for k,v in loss_dict.items():
            self.eval_epoch_loss_dict[k] = self.eval_epoch_loss_dict.get(k, 0) + v.item() * bs
        self.val_epoch_count += bs
