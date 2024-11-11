# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
from typing import Optional, List
import numpy as np


class SinusoidalTimeStepEmb(nn.Module):
    """
    Copied from diffuion policy
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PositionEmbeddingSineSeq(nn.Module):
    """
    A position embedding that make sine on the 2's dimension. B,T,D
    """
    def __init__(self, num_pos_feats=64, temperature=10000):
        super().__init__()
        self.temperature = temperature
        self.num_pos_feats = num_pos_feats

    @staticmethod
    def positional_encoding(input_tensor, temperature, d_model):
        """
        Generates sinusoidal positional encodings and adds them to the input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (B, T, D)
            d_model (int, optional): The embedding dimension D. If None, it will be inferred from input_tensor.

        Returns:
            torch.Tensor: Tensor of the same shape as input_tensor with positional encoding added.
        """
        B, T = input_tensor.shape[:2]

        device = input_tensor.device

        # Create position indices (T, 1)
        position = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(1)

        # Compute the div_term (D/2,)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * 
                            -(math.log(temperature) / d_model))

        # Initialize positional encoding tensor (T, D)
        pe = torch.zeros(T, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        # Expand pe to (1, T, D) and add to input tensor
        pe = pe.unsqueeze(0).repeat(B,1,1)  # Shape becomes (1, T, D)
        return pe

    def forward(self, input_tensor):
        return self.positional_encoding(input_tensor, self.temperature, self.num_pos_feats)
        
class PositionEmbeddingSineGrid(nn.Module):
    """
    Adapted from detr. https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    Please see `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for more details.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, offset:float=0.0, eps: float = 1e-6):
        super().__init__()
        self.num_pos_feats = num_pos_feats//2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, tensor, mask: Optional[torch.Tensor] = None):
        """
        tensor: torch.Tensor, shape (batch_size, T, height, width, channels)
        mask: torch.Tensor, shape (batch_size, height, width). 1 for valid positions, 0 for invalid positions
        (note this is the opposite to detrex's mask)
        """
        x = tensor
        assert x.dim() == 5, f"Expect 4D input tensor, got {x.dim()}"
        assert x.shape[4] == self.num_pos_feats*2, f"Expect last dimension to be {self.num_pos_feats*2}, got {x.shape[3]}"
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[2], x.shape[3], dtype=torch.bool, device=x.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = self.eps
            y_embed = (y_embed+self.offset) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed+self.offset) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()
        pos = pos.unsqueeze(1).repeat(1, x.shape[1], *([1] * (x.dim() - 2)))
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        x = tensor
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)

class PositionEncodingSineTable(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int, freeze: bool = True):
        super(PositionEncodingSineTable, self).__init__()
        # Initialize the position encodings
        self.weight = nn.Parameter(
            get_sinusoid_encoding_table(num_positions, embedding_dim)
        )
        if freeze:
            self.weight.requires_grad = False
    
    def forward(self, x):
        # x is expected to be of shape (batch_size, sequence_length, embedding_dim)
        # and the weight is expected to match the sequence_length
        # This example assumes x is the sequence of positions you want to encode
        sequence_length = x.size(1)
        position_encodings = self.weight[:sequence_length, :]
        return position_encodings.unsqueeze(0).repeat(x.shape[0], 1, 1)
