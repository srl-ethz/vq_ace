from ..common.autoinit_class import AutoInit
import torch.nn as nn
import torch
import hydra
import numpy as np
from ..models.common.position_encoding import PositionEmbeddingSineGrid, PositionEmbeddingSineSeq, PositionEncodingSineTable
from torchvision import models as vision_models
import torchvision
import functools
from ..models.robomimic_utils.crop_randomizer import CropRandomizer


class LowdimConcat(nn.Module):
    def __init__(self, datakeys, input_dim, output_dim):
        super(LowdimConcat, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.datakeys = datakeys
    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        return self.linear(x)

class ResNet18(nn.Module):
    """
    A ResNet18 block that can be used to process input images.
    Adapted from robomimic's ResNet18_BWHC without shape inference
    """
    def __init__(
        self,
        output_dim,
        input_channel=3,
        pretrained=False,
        flatten=False,
        input_shape = None
        # TODO: implement input_coord_conv
    ):
        """
        Args:
            output_dim (int): output dimension of the network. (token size)
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            flatten (bool): if True, flatten the output of the network into B, T, <ouput_dim>. 
                            If False, the output of the network is B, T, <ouput_dim>, H, W.
            input_shape (tuple): shape of the input image. Is used when flatten is True to infer the shape of the output.
        """
        super(ResNet18, self).__init__()
        net = vision_models.resnet18(pretrained=pretrained)
        if input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._output_dim = output_dim
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))
        self.flatten = flatten
        if self.flatten:
            assert input_shape is not None, "input_shape is required when flatten is True"
            self._input_shape = input_shape
            dummy_input = torch.zeros((1, input_channel, *input_shape))
            dummy_output = self.nets(dummy_input)
            output_shape = dummy_output.shape[2:]
            n_output_channels = output_shape[0] * output_shape[1] * 512
        else:
            n_output_channels = 512
        self.output_proj = nn.Linear(n_output_channels, output_dim) # the last block of resnet18 has 512 channels

    def forward(self, inputs):
        # inputs: (batch, T, C, H, W)
        input_shape = inputs.shape
        inputs = inputs.view(-1, *inputs.shape[2:]) # (batch*T, C, H, W)
        x = self.nets(inputs) # (batch*T, 512, H, W)
        x = x.view(*input_shape[:2], *x.shape[1:]) # (batch, T, 512, H, W)
        if self.flatten:
            x = x.view(*x.shape[:-3], -1) # (batch, T, 512*H*W)
        elif self._output_dim != 512:
            x = x.permute(0, 1, 3, 4, 2) # (batch, T, H, W, 512)
        x = self.output_proj(x)
        return x

class ZeroPosEmb(nn.Module):
    """
    A pos embedding that returns zeros
    """
    def __init__(self, output_dim):
        super(ZeroPosEmb, self).__init__()
        self.output_dim = output_dim
    def forward(self, x):
        return torch.zeros_like(x)

class TrajVisionResizer(nn.Module):
    """
    This is a wrapper to torchvision.transforms.Resize which resizes the last two dimensions
    """
    def __init__(self, size):
        super(TrajVisionResizer, self).__init__()
        self.size = size
        self._resizer = torchvision.transforms.Resize(size)
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, *x.shape[-3:])
        x = self._resizer(x)
        return x.view(*x_shape[:-3], *x.shape[-3:])


class ObsEncoder:
    """
    This encoder is designed to encode the observation into a sequence of tokens (Batch, Tokens, Dim of token). Together with the positional embeddings (Batch, Tokens, Dim of token) and mask
    The inputs of the observation are put into groups, each group is encoded by a separate encoder. The output of the encoder is concatenated along the token dimension.
    The positional embedding of the final ouput is build by two parts: the positional embedding within each group and the positional embedding of the group.
    """
    def __init__(self, output_dim, obs_groups_cfg, group_emb_cfg):
        """
        args:
            output_dim: the dimension of the output token
            obs_groups_cfg: a dictionary of the observation groups [group_name: group_dict], each group_dict is a dictionary with the following keys
            - datakeys: a list of keys in the observation dictionary that are used to encode the group
            - encoder_cfg: a dictionary of the encoder configuration
            - posemb_cfg: a dictionary of the positional embedding configuration
            group_emb_cfg: config how we embed each observation group. A dict with the following keys:
            - type: the type of the group embedding
        """
        self.backbones_mapping = {}
        self.nets = nn.ModuleDict()
        self.obs_groups_cfg = obs_groups_cfg
        self.output_dim = output_dim
        self.group_emb_cfg = group_emb_cfg if group_emb_cfg is not None else {"type": "none"}
        self.group_emb_type = self.group_emb_cfg["type"]
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
        elif type=="lowdim_concat":
            return LowdimConcat(datakeys, cfg["input_dim_total"], self.output_dim)
        elif type=="resnet18":
            return ResNet18(self.output_dim, 
                                input_channel=cfg.get("input_channel", 3), 
                                pretrained=cfg.get("pretrained", False), 
                                flatten=False
                    )
        elif type=="crop_resnet18":
            return nn.Sequential(
                TrajVisionResizer(
                    size = cfg["resize_shape"] # h w
                ),
                CropRandomizer(
                    crop_height = cfg["crop_shape"][0],
                    crop_width = cfg["crop_shape"][1]
                ),
                ResNet18(self.output_dim, 
                            input_channel=cfg.get("input_channel", 3), 
                            pretrained=cfg.get("pretrained", False), 
                            flatten=cfg.get("flatten", False), 
                            input_shape=cfg.get("input_shape", cfg["crop_shape"])
                ))

        elif type=="none":
            return lambda x:x

    def build_posemb(self, group_name, type, **cfg):
        if type=="seq_fixlenth": # this is deprecated
            return PositionEncodingSineTable(cfg['seq_len'], self.output_dim, freeze=True)

        elif type=="seq":
            return PositionEmbeddingSineSeq(self.output_dim, 
                **{k:v for k, v in cfg.items() if k in ["temperature"]} # filter the keys
                )

        elif type=="grid":
            return PositionEmbeddingSineGrid(self.output_dim, 
                **{k:v for k, v in cfg.items() if k in ["temperature", "normalize", "scale", "offset", "eps"]} # filter the keys
                )

        elif type=="none":
            return ZeroPosEmb(self.output_dim)


    def build_networks(self):
        obs_groups_cfg = self.obs_groups_cfg

        for group_name, cfg in obs_groups_cfg.items():
            encoder_cfg = cfg.get("encoder_cfg", {})
            posemb_cfg = cfg.get("posemb_cfg", {})
            datakeys = cfg['datakeys']
            self.nets[f'{group_name}_proj'] = self.build_encoder(group_name, datakeys=datakeys, **encoder_cfg)
            self.nets[f'{group_name}_posemb'] = self.build_posemb(group_name, **posemb_cfg)
        
        if self.group_emb_type == "none":
            pass
        elif self.group_emb_type == "each_group_learned":
            self.nets["group_emb"] = nn.Embedding(len(obs_groups_cfg), self.output_dim)
        elif self.group_emb_type == "whole_seq_sine":
            self.nets["group_emb"] = PositionEmbeddingSineSeq(self.output_dim)

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
        for gidx, gn in enumerate(group_names):
            res = self.nets[f'{gn}_proj'](*[obs_batch[k] for k in self.obs_groups_cfg[gn]["datakeys"]])
            posemb = self.nets[f'{gn}_posemb'](res)
            if len(res.shape) > 3: 
                token_per_step =  functools.reduce(lambda x, y: x*y, res.shape[2 : -1]) 
            else:
                token_per_step = 1
            res = res.view(res.shape[0], res.shape[1]*token_per_step, self.output_dim) # (batch, tokens, self.output_dim)
            posemb = posemb.view(posemb.shape[0], posemb.shape[1]*token_per_step, self.output_dim) # (tokens, self.output_dim)

            if self.group_emb_type == "each_group_learned":
                group_emb = self.nets["group_emb"](torch.tensor([gidx]).to(res.device))
                group_emb = group_emb.unsqueeze(1).repeat(posemb.shape[0], posemb.shape[1], 1)
                posemb = posemb + group_emb

            encoder_res.append(res)
            posembs.append(posemb)
            
            first_key_in_group = self.obs_groups_cfg[gn]["datakeys"][0]
            if first_key_in_group in obs_masks.keys():
                res_mask = obs_masks[first_key_in_group]  
                res_mask = res_mask.unsqueeze(-1).repeat(1, 1, token_per_step) # (batch, tokens, token_per_step)
                res_mask = res_mask.view(res_mask.shape[0], -1) # (batch, tokens)
            else:
                res_mask = torch.ones_like(res[:, :, 0])
            res_masks.append(res_mask)
        encoder_res = torch.cat(encoder_res, dim=1) # (batch, tokens, channels, ...)
        posembs = torch.cat(posembs, dim=1) # (batch, tokens, channels, ...)
        if self.group_emb_type == "whole_seq_sine":
            group_emb = self.nets["group_emb"](posembs)
            posembs = posembs + group_emb
        res_mask = torch.cat(res_masks, dim=1)
        return encoder_res, posembs, res_mask.to(encoder_res.device)


class ObsEncoderMixin(AutoInit, cfgname_and_funcs=(("obs_encoder_cfg", "_init_obs_encoder"),)):
    """
    Mixin class for algorithms that require an observation encoder.
        neural networks are saved in self._models["obs_encoder"]
        offers the following functions:
            encode_obs(obs_batch): 
    """
    def _init_obs_encoder(self, output_dim, obs_groups_cfg, group_emb_cfg):
        """
        Initialize the observation encoder according to the config
        """
        self._obs_encoder = ObsEncoder(output_dim, obs_groups_cfg, group_emb_cfg)
        self._models["obs_encoder"] = self._obs_encoder.nets
        self.set_device() # ensure set device after constructing the networks
        self._obs_shape_checker={} # a dictionary recording the shape of the obs last time. Useful to avoid some config bugs
    
    def _obs_check_shapes(self, k, data):
        if k not in self._obs_shape_checker.keys():
            self._obs_shape_checker[k] = data.shape[1:] # ignore the batch dim
        if data.shape[1:] != self._obs_shape_checker[k]:
            print(f"Warning: observation {k}'s shape doesn't match last time it pass through encoder")
            print(f"last time {self._obs_shape_checker[k]}, got {data.shape[1:]}")
            # assert False
        self._obs_shape_checker[k] = data.shape[1:]

    def normalize_obs(self, traj_dict, group_names=None):
        """
        Normalize the observation batch (NOT INPLACE but to return a new one)
        For a key in traj_dict named `k`, the std and mean stored in self._normalizers 
        """

        group_names = group_names if group_names is not None else self._obs_encoder.obs_groups_cfg.keys()
        keys_for_encoders = set([k for gn in group_names for k in self._obs_encoder.obs_groups_cfg[gn]["datakeys"]])

        return_dict = {}
        for k in keys_for_encoders:
            if k in self._normalizers.keys():
                self._obs_check_shapes(k, traj_dict[k])
                return_dict[k] = self._normalizers[k].normalize(traj_dict[k]) 
            else:
                raise ValueError(f"Normalizer for {k} is not found, please set _normalizers, with proper modules")
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
            obs_posembs = torch.zeros(bs, 0, self._obs_encoder.output_dim, device=self.device)
            obs_masks = torch.zeros(bs, 0, device=self.device)
        return obs_embeds, obs_posembs, obs_masks
         
