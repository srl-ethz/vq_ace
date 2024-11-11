# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import MultiheadAttention
from collections import OrderedDict

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output



class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                incremental_state: Optional[Dict[str, tuple]] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,
                           incremental_state=incremental_state)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     incremental_state: Optional[Dict[str, tuple]] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              incremental_state=incremental_state)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if memory is not None:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask,
                                    incremental_state=incremental_state)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    incremental_state: Optional[Dict[str, tuple]] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              incremental_state=incremental_state)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   incremental_state=incremental_state)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                incremental_state: Optional[Dict[str, tuple]] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,
                                    incremental_state=incremental_state)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,
                                 incremental_state=incremental_state)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MIMO_Transformer(nn.Module):
    """
    Extension to Transformer Encoder to accept multiple observation dictionaries as input and 
    to output dictionaries of tensors. Inputs are specified as a dictionary of
    observation dictionaries, with each key corresponding to an observation group.

    This module utilizes @ObservationGroupEncoder to process the multiple input dictionaries and
    to generate tensor dictionaries. The default behavior

    This module contains only a TransformerEncoder, the output are infered with [CLS] placeholders.
    args:
        extra_output_shapes: list of tuples, each tuple contains:
            - list of time indices. The length of the list is the number of the tokens. time indices are used for constructing the causal attention mask.
            - dictionary of output heads, where keys are the output head names and values are the output head dimensions
    """
    
    def __init__(self, 
        extra_output_shapes,
        d_model=512, nhead=8, 
        num_encoder_layers=6,
        dim_feedforward=2048, 
        dropout=0.1, 
        activation="relu",
        normalize_before=False,
        encoder_kwargs=None
    ):
        super(MIMO_Transformer, self).__init__()
        # create an observation encoder per observation group
        ## Manually create observation encoders for each observation group
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.transformerEncoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.out_heads = nn.ModuleDict()
        self.query_length_presum = [0]
        self.query_tidxs = []
        self.output_shapes = torch.jit.Attribute(extra_output_shapes, # work as a type check
            List[Tuple[List[int], Dict[str, int]]] 
        ).value
        for t_idxs, heads in extra_output_shapes:
            self.query_length_presum.append(self.query_length_presum[-1] + len(t_idxs))
            self.query_tidxs.extend(t_idxs)
            for outhead, outheaddim in heads.items():
                assert outhead not in self.out_heads.keys()
                self.out_heads[outhead] = nn.Linear(d_model, outheaddim)

        self.cls_embed = nn.Embedding(self.query_length_presum[-1], d_model)            

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.transformerEncoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed, mask, src_tidxs = None):
        """
        Process each set of inputs in its own observation group.

        Args:
            src: bs, seq, d_model
            pos_embed (torch.Tensor): positional embeddings: bs, token, d_model
            mask

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes
        """
        # process each observation group
        bs, lenseq = src.shape[:2]

        cls_embed = torch.unsqueeze(self.cls_embed.weight, dim=0).repeat(bs, 1, 1) # (bs, cls, d_model)

        encoder_input = torch.cat([src, cls_embed], dim=1) # (bs, seq+cls, d_model)
        encoder_input = encoder_input.permute(1, 0, 2) # (seq+cls, bs, d_model)

        cls_mask = torch.ones(cls_embed.shape[:2], dtype=torch.bool).to(cls_embed.device) # True: valid, False: pad
        mask = torch.cat([mask, cls_mask], dim=1).to(dtype=torch.bool)  # (bs, seq+cls)
        
        pos_embed = torch.cat([pos_embed, torch.zeros_like(cls_embed)], dim=1) # (bs, seq+cls, d_model)
        pos_embed = pos_embed.permute(1, 0, 2)  # (seq, 1, d_model)
        lenall = pos_embed.shape[0]

        # construct the causal attention mask
        if src_tidxs is not None:
            #  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
            # `For a binary mask, a True value indicates that the corresponding position is not allowed to attend. `
            tidx = torch.cat([torch.tensor(src_tidxs), torch.tensor(self.query_tidxs)], dim=0).to(device=mask.device) # (seq_len+cls_len)
            # Expand t_idx to (seq_len, seq_len)
            tidx1 = tidx.unsqueeze(-1)  # shape: (seq_len, 1)
            tidx2 = tidx.unsqueeze(0)   # shape: (1, seq_len)
            assert len(tidx) == lenall
            causal_mask = tidx1+ 1e-6 < tidx2    # shape: seq_len, seq_len)
            causal_mask = causal_mask.to(dtype=torch.bool, device=mask.device)
            assert mask.dtype == causal_mask.dtype , "If both attn_mask and key_padding_mask are supplied, their types should match. --https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html"
        else:
            causal_mask = None
        hs = self.transformerEncoder(encoder_input, pos=pos_embed, src_key_padding_mask=~mask, mask=causal_mask)
        hs = hs.permute(1, 0, 2) # (bs, seq+cls, d_model)

        query_start_ind = lenseq
        extra_out = OrderedDict()
        for i, (length, heads) in enumerate(self.output_shapes):
            for outhead, outheaddim in heads.items():
                extra_out[outhead] = self.out_heads[outhead](
                    hs[:, query_start_ind+self.query_length_presum[i]:query_start_ind+self.query_length_presum[i+1], :]
                )
        return hs, extra_out

