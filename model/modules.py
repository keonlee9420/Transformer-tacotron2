import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import hyperparams as hp


class ConvNorm(nn.Module):
    """Construct a conv1d + batchnorm1d module."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, batch_norm=True, activation='relu', dropout=None):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_channels)
        else:
            self.batch_norm = None
        self.activation = self._get_activation(activation)

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        if self.activation is not None:
            out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)

        return out

    @staticmethod
    def _get_activation(activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            return None


class PositionalEncoding(nn.Module):
    """Implement the PE function."""
    def __init__(self, dim_hidden, dropout=hp.dropout, max_len=hp.positional_encoding_max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32))

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim_hidden)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_hidden, 2) *
                             -(math.log(10000.0) / dim_hidden))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Prenet output (batch_size x time length x d_model)
        returns: Prenet output + (alpha * PositionalEncoding)
        """
        x = x + self.alpha * self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, dim_hidden=hp.model_dim, dim_ffn=hp.d_ff, dropout=hp.dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_hidden, dim_ffn)
        self.w_2 = nn.Linear(dim_ffn, dim_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Attention(nn.Module):
    def __init__(self, dim_hidden, d_k, d_v, dropout_p):
        super(Attention, self).__init__()
        self.query = nn.Linear(dim_hidden, d_k, bias=False)
        self.key = nn.Linear(dim_hidden, d_k, bias=False)
        self.value = nn.Linear(dim_hidden, d_v, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, x, mask=None, memory=None):
        query = self.query(x)
        if memory is not None:
            key = self.key(memory)
            value = self.value(memory)
        else:
            key = self.key(x)
            value = self.value(x)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, -1)
        scores = self.dropout(scores)

        return torch.matmul(scores, value), scores


class MultiHeadAttention(nn.Module):
    def __init__(self, h=hp.num_heads, dim_hidden=hp.model_dim, dropout=hp.dropout):
        super(MultiHeadAttention, self).__init__()
        assert dim_hidden % h == 0
        self.d_k = dim_hidden // h
        self.d_v = dim_hidden // h
        self.h = h
        self.attention = nn.ModuleList([Attention(dim_hidden, self.d_k, self.d_v, dropout)
                                        for _ in range(self.h)])
        self.out_linear = nn.Linear(self.d_v * self.h, dim_hidden)

    def forward(self, x, mask=None, memory=None):
        heads = []
        attention_maps = []
        for i, attn in enumerate(self.attention):
            head, attn_map = attn(x, mask, memory)
            heads.append(head)
            attention_maps.append(attn_map)
        head_cat = torch.cat(heads, dim=-1)
        out = self.out_linear(head_cat)

        return out, tuple(attention_maps)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, sublayer, dim_hidden=hp.model_dim, dropout=hp.dropout):
        super(SublayerConnection, self).__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dim_hidden, eps=hp.layernorm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        """Apply residual connection to any sublayer with the same size."""
        out = self.sublayer(x, **kwargs)
        if isinstance(out, tuple):
            out, attn_maps = out
        else:
            attn_maps = None

        out = self.norm(x + self.dropout(out))

        if attn_maps is not None:
            return out, attn_maps
        else:
            return out
