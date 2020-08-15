#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import copy
import math
import hyperparams as hp
from model import PositionalEncoding

import utils


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=hp.layernorm_eps):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ConvNorm(nn.Module):
    """Construct a convnorm module. (from tacotron2)"""

    def __init__(self, in_channels, out_channels, kernel_size=hp.convnorm_kernel_size, stride=hp.convnorm_stride,
                 padding=hp.convnorm_padding, dilation=hp.convnorm_dilation, bias=hp.convnorm_bias,
                 w_init_gain=hp.convnorm_w_init_gain):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.pos = PositionalEncoding(hp.model_dim, hp.model_dropout)
        self.encoder_prenet = EncoderPrenet(
            hp.model_dim, hp.model_dim, hp.model_dropout)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        # prenet
        x = self.encoder_prenet(x.transpose(-2, -1))
        x = x.transpose(-2, -1)

        # positional encoding
        x = self.pos(x)

        # encoder
        for layer in self.layers:
            x = layer(x, mask)
        memory = self.norm(x)

        return memory


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class EncoderPrenet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(EncoderPrenet, self).__init__()

        self.conv1 = nn.Sequential(ConvNorm(in_channels, out_channels),
                                   nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True), nn.Dropout(dropout, inplace=True))
        self.convs = nn.ModuleList([self.conv1])
        for i in range(hp.encoder_n_conv - 1):
            self.convs.append(nn.Sequential(ConvNorm(out_channels, out_channels),
                                            nn.BatchNorm1d(out_channels),
                                            nn.ReLU(inplace=True), nn.Dropout(dropout, inplace=True)))

    def forward(self, x):
        for m in self.convs:
            x = m(x)
        return x


def sample_encoding(sample_batch):
    import attention
    import model
    import numpy as np
    import hyperparams as hp

    vocab = utils.build_phone_vocab(['hello world!'])
    print("vocab:\n", vocab)
    embed = Embeddings(hp.model_dim, len(vocab))
    prenet = EncoderPrenet(hp.model_dim, hp.model_dim, hp.model_dropout)
    # print(prenet)

    c = copy.deepcopy
    attn = attention.MultiHeadedAttention(hp.num_heads, hp.model_dim)
    ff = model.PositionwiseFeedForward(
        hp.model_dim, hp.hidden_dim, hp.model_dropout)

    encoder = Encoder(EncoderLayer(hp.model_dim, c(
        attn), c(ff), hp.model_dropout), hp.num_layers)

    x = [vocab[p] for p in utils.phoneme('hello world!').split(' ') if p]

    x = torch.tensor(x, dtype=torch.long)
    seq_len = x.shape[0]

    print("phoneme_size: ", x.size())
    emb = embed(x).unsqueeze(0)
    emb_batch = torch.cat([emb for _ in range(sample_batch)], 0)

    src_mask = torch.ones(sample_batch, 1, seq_len)
    memory = encoder(emb_batch, src_mask)
    # print("embedding batch size: ", emb_batch.size())
    # pre = prenet(emb_batch.transpose(-2, -1))
    # print("prenet output size: ", pre.size())
    # encoder_input = pre.transpose(-2, -1)
    # memory = encoder(encoder_input, torch.ones(sample_batch, 1, seq_len))
    # print("memory size: ", memory.size())

    return memory, src_mask


if __name__ == '__main__':
    memory, src_mask = sample_encoding(10)
