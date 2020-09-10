#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import copy
import math
import hyperparams as hp

import utils

from .modules import *
from .attention import *


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.centering = Linear(hp.model_dim, hp.model_dim)
        self.pos = PositionalEncoding(hp.model_dim, hp.model_dropout)
        self.encoder_prenet = EncoderPrenet(
            hp.model_dim, hp.model_dim, hp.model_dropout)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        # prenet
        x = self.encoder_prenet(x.transpose(-2, -1))
        x = x.transpose(-2, -1)

        # center consistency
        x = self.centering(x)

        # positional encoding
        x = self.pos(x)

        # encoder
        attn_enc = list()
        for layer in self.layers:
            x = layer(x, mask)
            attn_enc.append(layer.self_attn.attn)
        memory = self.norm(x)

        return memory, attn_enc


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


class EncoderPrenet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(EncoderPrenet, self).__init__()
        self.conv1 = nn.Sequential(ConvNorm(in_channels, out_channels),
                                   nn.BatchNorm1d(out_channels),
                                   nn.ReLU(), nn.Dropout(dropout))
        self.convs = nn.ModuleList([self.conv1])
        for i in range(hp.encoder_n_conv - 1):
            self.convs.append(nn.Sequential(ConvNorm(out_channels, out_channels),
                                            nn.BatchNorm1d(out_channels),
                                            nn.ReLU(), nn.Dropout(dropout)))
        self.projection = Linear(out_channels, out_channels)

    def forward(self, x):
        for m in self.convs:
            x = m(x)
        out = self.projection(x.transpose(-2, -1))
        return out.transpose(-2, -1)


def sample_encoding(sample_batch):
    import model
    import hyperparams as hp

    vocab = utils.build_phone_vocab(['hello world!'])
    print("vocab:\n", vocab)
    embed = Embeddings(hp.model_dim, len(vocab))
    prenet = EncoderPrenet(hp.model_dim, hp.model_dim, hp.model_dropout)
    # print(prenet)

    c = copy.deepcopy
    attn = MultiHeadedAttention(hp.num_heads, hp.model_dim)
    ff = model.PositionwiseFeedForward(
        hp.model_dim, hp.hidden_dim, hp.model_dropout)

    encoder = Encoder(EncoderLayer(hp.model_dim, c(
        attn), c(ff), hp.model_dropout), hp.num_layers)

    x = [vocab[p] for p in utils._phonemize('hello world!').split(' ') if p]

    x = torch.tensor(x, dtype=torch.long)
    seq_len = x.shape[0]

    print("phoneme_size: ", x.size())
    emb = embed(x).unsqueeze(0)
    emb_batch = torch.cat([emb for _ in range(sample_batch)], 0)

    src_mask = torch.ones(sample_batch, 1, seq_len)
    memory = encoder(emb_batch, src_mask)

    return memory, src_mask


if __name__ == '__main__':
    memory, src_mask = sample_encoding(10)
