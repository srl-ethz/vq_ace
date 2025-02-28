from ..common.autoinit_class import AutoInit
import torch.nn as nn
import torch
import hydra
import numpy as np
from ..models.common.position_encoding import PositionEncodingSine1D

"""
An observation encoder is a neural network that encodes the observation into some unified representations. E.g. tokens
"""

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)

class ObsEncoder:
    def __init__(self, output_dim, obs_groups_cfg):
        self.backbones_mapping = {}
        self.nets = nn.ModuleDict()
        self.obs_groups_cfg = obs_groups_cfg
        self.output_dim = output_dim
        self.build_networks()

    def build_encoder(self, group_name, type, datakeys, backbone_instance_name=None, **cfg):
        """
        The data to the encoder have the shape: [batch, traj_horizon, channels ... ]
        Encoding is to encode along the horizon dimension, i.e. output shape is [batch, token_horizon, out_channels ...]
        """
        if backbone_instance_name is not None:
            if backbone_instance_name in self._obs_encoder_backbones_mapping.keys():
                return self.nets[self._obs_encoder_backbones_mapping[backbone_instance_name][0]]
            else:
                net = self.build_encoder(type, datakeys, **cfg)
                self._obs_encoder_backbones_mapping[backbone_instance_name] = [group_name]                
                return net
        if type=="torch":
            assert len(datakeys) == 1, "Only one datakey is supported for now"
            return hydra.utils.instantiate(cfg)
        elif type=="none":
            return None

    def build_posemb(self, group_name, type, **cfg):
        if type=="seq":
            embed = PositionEncodingSine1D(cfg['seq_len'], self.output_dim, freeze=True)
            return embed
        else:
            return 0.0

    def build_networks(self):
        obs_groups_cfg = self.obs_groups_cfg

        for group_name, cfg in obs_groups_cfg.items():
            encoder_cfg = cfg.get("encoder_cfg", {})
            posemb_cfg = cfg.get("posemb_cfg", {})
            datakeys = cfg['datakeys']
            self.nets[f'{group_name}_proj'] = self.build_encoder(group_name, datakeys=datakeys, **encoder_cfg)
            self.nets[f'{group_name}_posemb'] = self.build_posemb(group_name, **posemb_cfg)
        
    def encode_obs(self, obs_batch, obs_masks, group_names=None):
        """
        args:
            obs_batch: is the dictionary of observations, no masks
            obs_masks: is the dictionary of masks, with the same keys as obs_batch. True means valid, False means padded.
        
        The output should be (embed, pos_embeds, masks) of shape (batch, horizon, channels, ...)
        group_names is a list of group names to encode, if None, encode all groups
        """
        if group_names is None:
            group_names = self.obs_groups_cfg.keys()

        if len(group_names) == 0:
            return None, None, None
        encoder_res = []
        posembs = []
        res_masks = []
        for gn in group_names:
            res = self.nets[f'{gn}_proj'](*[obs_batch[k] for k in self.obs_groups_cfg[gn]["datakeys"]])
            posemb = self.nets[f'{gn}_posemb'](res)
            encoder_res.append(res)
            posembs.append(posemb)
            
            first_key_in_group = self.obs_groups_cfg[gn]["datakeys"][0]
            res_mask = obs_masks[first_key_in_group] if first_key_in_group in obs_masks.keys() else torch.ones_like(res[:, :, 0])
            res_masks.append(res_mask)
        encoder_res = torch.cat(encoder_res, dim=1) # (batch, tokens, channels, ...)
        posembs = torch.cat(posembs, dim=0) # (tokens, channels, ...)
        res_mask = torch.cat(res_masks, dim=1)
        return encoder_res, posembs, res_mask.to(encoder_res.device)


class ObsEncoderMixin(AutoInit, cfgname_and_funcs=(("obs_encoder_cfg", "_init_obs_encoder"),)):
    """
    Mixin class for algorithms that require an observation encoder.
        neural networks are saved in self._models["obs_encoder"]
        offers the following functions:
            encode_obs(obs_batch): 
    """
    def _init_obs_encoder(self, output_dim, obs_groups_cfg):
        """
        Initialize the observation encoder according to the config
        """
        self._obs_encoder = ObsEncoder(output_dim, obs_groups_cfg)
        self._models["obs_encoder"] = self._obs_encoder.nets
        self.set_device() # ensure set device after constructing the networks

    def normalize_obs(self, traj_dict, group_names=None):
        """
        Normalize the observation batch (NOT INPLACE but to return a new one)
        For a key in traj_dict named `k`, the std and mean stored in self._normalizer_stds[obs_{k}] and self._normalizer_means[obs_{k}] are required
        """

        group_names = group_names if group_names is not None else self._obs_encoder.obs_groups_cfg.keys()
        keys_for_encoders = set([k for gn in group_names for k in self._obs_encoder.obs_groups_cfg[gn]["datakeys"]])

        return_dict = {}
        for k in keys_for_encoders:
            obs_k = f"obs_{k}"
            if obs_k in self._normalizer_stds.keys():
                return_dict[k] = (traj_dict[k] - self._normalizer_means[obs_k]) / self._normalizer_stds[obs_k]
            else:
                raise ValueError(f"Normalizer for {obs_k} is not found, please set _normalizer_stds and _normalizer_means, with proper modules")
        return return_dict

    def encode_obs(self, batch, mask_batch, group_names=None):
        """
        Encode the observation batch
        """
        obs_embeds, obs_posembs, obs_masks = self._obs_encoder.encode_obs(self.normalize_obs(batch, group_names), mask_batch, group_names)
        if obs_embeds is None:
            a = next(iter(batch.values()))
            bs = a.shape[0]
            obs_embeds = torch.zeros(bs, 0, self._obs_encoder.output_dim, device=self.device)
            obs_posembs = torch.zeros(0, self._obs_encoder.output_dim, device=self.device)
            obs_masks = torch.zeros(bs, 0, device=self.device)
        return obs_embeds, obs_posembs, obs_masks
         
