# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# and DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import constant_, normal_, uniform_, xavier_uniform_

from models.ops.temporal_deform_attn import DeformAttnCondition
from opts import cfg
from util.misc import inverse_sigmoid


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False, two_stage=False, look_forward_twice=False,
                 mixed_selection=False, use_dab=True, high_dim_query_update=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.mixed_selection = mixed_selection
        self.use_dab = use_dab
        self.high_dim_query_update = high_dim_query_update

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers,
            d_model=d_model, return_intermediate=return_intermediate_dec,
            look_forward_twice=look_forward_twice, use_dab=use_dab,
            high_dim_query_update=high_dim_query_update,
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 3, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            if not self.use_dab:
                self.reference_points = nn.Linear(d_model, 1)

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            assert not self.use_dab, "use_dab must be True"

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformAttnCondition):
                m._reset_parameters()

        if not self.two_stage and not self.use_dab:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 256
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, temporal_lengths):
        N_, S_, C_ = memory.shape
        proposals = []

        _cur = 0
        base_scale = 4.0
        for lvl, T_ in enumerate(temporal_lengths):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + T_)].view(N_, T_)
            valid_T = torch.sum(~mask_flatten_[:, :], 1)

            timeline = (torch.linspace(0, T_ - 1, T_, dtype=torch.float32, device=memory.device).unsqueeze(0) + 0.5) / valid_T.unsqueeze(1)

            scale = torch.ones_like(timeline) * 0.05 * (2.0 ** lvl)
            proposal = torch.stack((timeline, scale), -1).view(N_, -1, 2)
            proposals.append(proposal)
            _cur += T_

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, T = mask.shape
        valid_T = torch.sum(~mask, 1)
        valid_ratio = valid_T.float() / T
        return valid_ratio    # shape=(bs)

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        '''
        Params:
            srcs: list of Tensor with shape (bs, c, t)
            masks: list of Tensor with shape (bs, t)
            pos_embeds: list of Tensor with shape (bs, c, t)
            query_embed: list of Tensor with shape (nq, 2c)
        Returns:
            hs: list, per layer output of decoder
            init_reference_out: reference points predicted from query embeddings
            inter_references_out: reference points predicted from each decoder layer
            memory: (bs, c, t), final output of the encoder
        '''
        assert self.two_stage or query_embed is not None
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        temporal_lens = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, t = src.shape
            temporal_lens.append(t)
            # (bs, c, t) => (bs, t, c)
            src = src.transpose(1, 2)
            pos_embed = pos_embed.transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        temporal_lens = torch.as_tensor(temporal_lens, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((temporal_lens.new_zeros((1, )), temporal_lens.cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)   # (bs, nlevels)

        # deformable encoder
        memory = self.encoder(src_flatten, temporal_lens, level_start_index, valid_ratios,
            lvl_pos_embed_flatten if cfg.use_pos_embed else None,
            mask_flatten)  # shape=(bs, t, c)

        bs, t, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, temporal_lens)

            num_topk = 100
            # num_topk = t
            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.segment_embed[self.decoder.num_layers](output_memory)
            centers = enc_outputs_coord_unact[..., [0]] + output_proposals[..., [0]]
            enc_outputs_coord_unact = torch.cat([
                centers,
                enc_outputs_coord_unact[..., [1]],
                enc_outputs_coord_unact[..., [2]]
            ], dim=-1)

            # topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], num_topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 3))

            # topk_coords_unact = enc_outputs_coord_unact
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points

            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            if not self.mixed_selection:
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            else:
                # query_embed here is the content embed for deformable DETR
                tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)
                query_embed, _ = torch.split(pos_trans_out, c, dim=2)

        elif self.use_dab:
            # reference_points = query_embed[..., self.d_model:].unsqueeze(0).sigmoid()
            reference_points = query_embed[..., self.d_model:].sigmoid()
            reference_points = reference_points.unsqueeze(0).expand(bs, -1, -1)
            tgt = query_embed[..., :self.d_model]
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            init_reference_out = reference_points
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            temporal_lens, level_start_index, valid_ratios, query_embed if not self.use_dab else None, mask_flatten)
        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, memory.transpose(1, 2), enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, memory.transpose(1, 2), None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = DeformAttnCondition(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2, _ = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, T_ in enumerate(spatial_shapes):
            ref = torch.linspace(0.5, T_ - 0.5, T_, dtype=torch.float32, device=device)  # (t,)
            ref = ref[None] / (valid_ratios[:, None, lvl] * T_)                          # (bs, t)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]          # (N, t, n_levels)
        return reference_points[..., None]                                               # (N, t, n_levels, 1)

    def forward(self, src, temporal_lens, level_start_index, valid_ratios, pos=None, padding_mask=None):
        '''
        src: shape=(bs, t, c)
        temporal_lens: shape=(n_levels). content: [t1, t2, t3, ...]
        level_start_index: shape=(n_levels,). [0, t1, t1+t2, ...]
        valid_ratios: shape=(bs, n_levels).
        '''
        output = src
        # (bs, t, levels, 1)
        reference_points = self.get_reference_points(temporal_lens, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, temporal_lens, level_start_index, padding_mask)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = DeformAttnCondition(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        if not cfg.disable_query_self_att:
            # self attention
            q = k = self.with_pos_embed(tgt, query_pos)

            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        else:
            pass
        # cross attention
        # print(tgt.size(), query_pos.size(), reference_points.size(), src.size())
        tgt2, _ = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, d_model=256,
        return_intermediate=False, look_forward_twice=False, use_dab=False,
        high_dim_query_update=False, no_sine_embed=False
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.look_forward_twice = look_forward_twice
        self.use_dab = use_dab
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.segment_embed = None
        self.class_embed = None
        self.high_dim_query_update = high_dim_query_update
        self.no_sine_embed = no_sine_embed

        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, num_layers=2)
            self.ref_point_head = MLP(d_model * 3, d_model, d_model, num_layers=2)
        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            self.high_dim_query_proj = MLP(d_model, d_model, d_model, num_layers=2)


    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        '''
        tgt: [bs, nq, C]
        reference_points: [bs, nq, 1 or 2]
        src: [bs, T, C]
        src_valid_ratios: [bs, levels]
        '''
        output = tgt
        if self.use_dab:
            assert query_pos is None
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            # (bs, nq, 1, 1 or 2) x (bs, 1, num_level, 1) => (bs, nq, num_level, 1 or 2)
            # if reference_points.shape[-1] == 2:
            #     reference_points_input = reference_points[:, :, None] \
            #                              * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None] # bs, nq, 4, 4
            # else:
            #     assert reference_points.shape[-1] == 1
            #     reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None, :, None]

            # if reference_points.shape[-1] == 2:
            #     reference_points_input = reference_points[:, :, None] \
            #                              * src_valid_ratios[:, None, :, None] # bs, nq, 4, 2
            # else:
            #     assert reference_points.shape[-1] == 1
            #     # print(reference_points[None, :, :].size(), src_valid_ratios[:, None, :, None].size())
            #     reference_points_input = reference_points[None, :, :] * src_valid_ratios[:, None, :, None]
            if self.use_dab:
                # print(reference_points_input.size())
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # bs, nq, 256*2
                # print(query_sine_embed.size(), self.ref_point_head)
                # raw_query_pos = self.ref_point_head(query_sine_embed) if lid != 0 else query_sine_embed # bs, nq, 256
                raw_query_pos = self.ref_point_head(query_sine_embed)
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            if self.high_dim_query_update and lid != 0:
                query_pos = query_pos + self.high_dim_query_proj(output)

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for segment refinement
            if self.segment_embed is not None:
                # update the reference point/segment of the next layer according to the output from the current layer
                tmp = self.segment_embed[lid](output)
                # if reference_points.shape[-1] == 2:
                #     new_reference_points = tmp + inverse_sigmoid(reference_points)
                #     new_reference_points = new_reference_points.sigmoid()
                # else:
                #     # at the 0-th decoder layer
                #     # d^(n+1) = delta_d^(n+1)
                #     # c^(n+1) = sigmoid( inverse_sigmoid(c^(n)) + delta_c^(n+1))
                #     assert reference_points.shape[-1] == 1
                # new_reference_points = tmp
                # new_reference_points = tmp[..., :1] + inverse_sigmoid(reference_points)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(
                    new_reference_points
                    if self.look_forward_twice
                    else reference_points
                )
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


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
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deformable_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        return_intermediate_dec=True,
        two_stage=args.two_stage,
        look_forward_twice=args.look_forward_twice,
        mixed_selection=args.mixed_selection,
        num_feature_levels=1,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        use_dab=True,
    )



def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(512, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / 512)
    x_embed = pos_tensor[:, :, 0] * scale
    # y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    # pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    # pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 1:
        pos = pos_x
    elif pos_tensor.size(-1) == 2:
        w_embed = pos_tensor[:, :, 1] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_x, pos_w), dim=2)

    elif pos_tensor.size(-1) == 3:
        l_embed = pos_tensor[:, :, 1] * scale
        pos_l = l_embed[:, :, None] / dim_t
        pos_l = torch.stack((pos_l[:, :, 0::2].sin(), pos_l[:, :, 1::2].cos()), dim=3).flatten(2)

        r_embed = pos_tensor[:, :, 1] * scale
        pos_r = r_embed[:, :, None] / dim_t
        pos_r = torch.stack((pos_r[:, :, 0::2].sin(), pos_r[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_x, pos_l, pos_r), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos