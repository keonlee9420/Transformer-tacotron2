#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import copy
import math
import hyperparams as hp

import utils

from .modules import *


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self,
                 N=hp.num_layers,
                 h=hp.num_heads,
                 dim_hidden=hp.model_dim,
                 dim_embedding=hp.model_dim,
                 dim_ffn=hp.d_ff,
                 dropout=hp.dropout,
                 num_embeddings=hp.num_embeddings):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(h=h, dim_hidden=dim_hidden, dim_ffn=dim_ffn) for _ in range(N)])
        self.pos = PositionalEncoding(dim_hidden, dropout)
        self.encoder_prenet = EncoderPrenet(dim_embedding, dim_hidden, dropout)
        self.embedding = nn.Embedding(num_embeddings, dim_embedding)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        # Embedding
        x = self.embedding(x)

        # prenet
        x = self.encoder_prenet(x)

        # positional encoding
        x = self.pos(x)

        # encoder
        attn_enc = tuple()
        for layer in self.layers:
            x, attn = layer(x, mask)
            attn_enc += attn

        return x, attn_enc


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""
    def __init__(self,
                 h=hp.num_heads,
                 dim_hidden=hp.model_dim,
                 dim_ffn=hp.d_ff,
                 dropout=hp.dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = SublayerConnection(MultiHeadAttention(h=h,
                                                               dim_hidden=dim_hidden,
                                                               dropout=dropout))
        self.feed_forward = SublayerConnection(PositionwiseFeedForward(dim_hidden=hp.model_dim,
                                                                       dim_ffn=dim_ffn,
                                                                       dropout=dropout))

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        attn_out, attn_maps = self.self_attn(x, mask=mask)
        out = self.feed_forward(attn_out)
        return out, attn_maps


class EncoderPrenet(nn.Module):
    def __init__(self, in_channels=hp.num_embeddings, out_channels=hp.model_dim, dropout=hp.dropout):
        super(EncoderPrenet, self).__init__()
        self.conv1 = ConvNorm(in_channels, out_channels,
                              batch_norm=True, activation='relu', dropout=dropout)
        self.conv2 = ConvNorm(out_channels, out_channels,
                              batch_norm=True, activation='relu', dropout=dropout)
        self.conv3 = ConvNorm(out_channels, out_channels,
                              batch_norm=True, activation='relu', dropout=dropout)

        self.projection = Linear(out_channels, out_channels)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out = self.projection(out3)
        return out
