import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask, LocalMask
import os


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        # print('queries shape',queries.shape)
        # print('keys shape',keys.shape)
        # print('values shape',values.shape)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = torch.softmax(scale * scores, dim=-1)
        A_drop = self.dropout(A)
        V = torch.einsum("bhls,bshd->blhd", A_drop, values)
        #print('output shape',V.shape)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

        
class SparseAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if attn_mask is None:
            attn_mask = LocalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        #print('output shape',V.shape)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)        


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
class AttentionEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", pre_norm=False):
        super(AttentionEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.pre_norm = pre_norm
        if pre_norm:
            self.pre_norm_x = nn.LayerNorm(d_model)
            self.pre_norm_k = nn.LayerNorm(d_model)
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, keys, attn_mask=None):
        x = x.permute(0,2,1)
        keys = keys.permute(0,2,1)

        if self.pre_norm:
            x = self.pre_norm_x(x)
            keys = self.pre_norm_k(keys)

        new_x, attn = self.attention(
            x, keys, keys,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = self.norm2(x + y)
        y = y.permute(0,2,1)
        return y, attn
    

class TV_AttentionLayer(nn.Module):
    def __init__(self, attention, d_t, d_v, n_heads):
        super(TV_AttentionLayer, self).__init__()

        self.inner_attention = attention
        self.t_query_projection = nn.Linear(d_t, d_t)
        self.t_key_projection = nn.Linear(d_t, d_t)
        self.t_value_projection = nn.Linear(d_t, d_t)
        self.v_query_projection = nn.Linear(d_v, d_v)
        self.v_key_projection = nn.Linear(d_v, d_v)
        self.v_value_projection = nn.Linear(d_v, d_v)
        self.t_out_projection = nn.Linear(d_t, d_t)
        self.v_out_projection = nn.Linear(d_v, d_v)
        self.n_heads = n_heads
        self.d_t = d_t
        self.d_v = d_v

    def forward(self, timestamp_emb, value_emb, attn_mask):

        B, S, _ = timestamp_emb.shape
        _, L, _ = value_emb.shape
        H = self.n_heads

        timestamp_emb_queries = self.t_query_projection(timestamp_emb).view(B, S, H, self.d_t//H)
        timestamp_emb_keys = self.t_key_projection(timestamp_emb).view(B, S, H, self.d_t//H)
        timestamp_emb_values = self.t_value_projection(timestamp_emb).view(B, S, H, self.d_t//H)

        value_emb_queries = self.v_query_projection(value_emb).view(B, L, H, self.d_v//H)
        value_emb_keys = self.v_key_projection(value_emb).view(B, L, H, self.d_v//H)
        value_emb_values = self.v_value_projection(value_emb).view(B, L, H, self.d_v//H)

        queries = torch.cat([timestamp_emb_queries, value_emb_queries], dim=-1)
        keys = torch.cat([timestamp_emb_keys, value_emb_keys], dim=-1)
        values = torch.cat([timestamp_emb_values, value_emb_values], dim=-1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        out_t = out[:, :, :, :self.d_t//H]
        out_v = out[:, :, :, -self.d_v//H:]
        out_t = out_t.reshape(B, L, self.d_t)
        out_v = out_v.reshape(B, L, self.d_v)
        out_t = self.t_out_projection(out_t)
        out_v = self.v_out_projection(out_v)

        return out_t, out_v, attn
    
class TV_AttentionEncoderLayer(nn.Module):
    def __init__(self, attention, d_t, d_v, d_ff=None, dropout=0.1, activation="relu", pre_norm=False):
        super(TV_AttentionEncoderLayer, self).__init__()
        d_ffv = 4 * d_v
        d_fft = 4 * d_t

        self.attention = attention

        if pre_norm:
            self.pre_norm_t = nn.LayerNorm(d_t)
            self.pre_norm_v = nn.LayerNorm(d_v)
        self.pre_norm = pre_norm

        self.t_conv1 = nn.Conv1d(in_channels=d_t, out_channels=d_fft, kernel_size=1)
        self.t_conv2 = nn.Conv1d(in_channels=d_fft, out_channels=d_t, kernel_size=1)
        self.t_norm1 = nn.LayerNorm(d_t)
        self.t_norm2 = nn.LayerNorm(d_t)

        self.v_conv1 = nn.Conv1d(in_channels=d_v, out_channels=d_ffv, kernel_size=1)
        self.v_conv2 = nn.Conv1d(in_channels=d_ffv, out_channels=d_v, kernel_size=1)
        self.v_norm1 = nn.LayerNorm(d_v)
        self.v_norm2 = nn.LayerNorm(d_v)

        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, t, v, attn_mask=None):
        t = t.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        if self.pre_norm:
            t = self.pre_norm_t(t)
            v = self.pre_norm_v(v)

        new_t, new_v, attn = self.attention(
            t, v,
            attn_mask=attn_mask
        )
        t = t + self.dropout(new_t)
        v = v + self.dropout(new_v)

        yt = self.t_norm1(t)
        yt = self.dropout(self.activation(self.t_conv1(yt.transpose(-1, 1))))
        yt = self.dropout(self.t_conv2(yt).transpose(-1, 1))
        yt = self.t_norm2(t + yt)
        yt = yt.permute(0, 2, 1)

        yv = self.v_norm1(v)
        yv = self.dropout(self.activation(self.v_conv1(yv.transpose(-1, 1))))
        yv = self.dropout(self.v_conv2(yv).transpose(-1, 1))
        yv = self.v_norm2(v + yv)
        yv = yv.permute(0, 2, 1)

        return yt, yv, attn
