#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from encoder import clones, LayerNorm, ConvNorm, SublayerConnection


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys,
                           subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


class DecoderPrenet(nn.Module):
    """
    Decoder pre-net for TTS
    For the alignment between phoneme and mel frame to be measured
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(DecoderPrenet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, decoder_input):
        return self.layer(decoder_input)


class MelLinear(nn.Module):
    """Linear projections to predict the mel spectrogram same as Tacotron2."""

    def __init__(self, num_hidden, mel_channels):
        super(MelLinear, self).__init__()
        self.mel_linear = nn.Linear(num_hidden, mel_channels)

    def forward(self, decoder_input):
        return self.mel_linear(decoder_input)


class StopLinear(nn.Module):
    """Linear projections to predict the stop token same as Tacotron2."""

    def __init__(self, num_hidden):
        super(StopLinear, self).__init__()
        self.stop_linear = nn.Linear(num_hidden, 1, w_init='sigmoid')

    def forward(self, decoder_input):
        return self.stop_linear(decoder_input)


class Postnet(nn.Module):
    """Linear projections to produce a residual to refine the reconstruction of mel spectrogram same as Tacotron2."""

    def __init__(self, mel_channels, hidden_dim, kernel_size, num_conv=5, dropout=0.1):
        super(Postnet, self).__init__()
        self.mel_channels = mel_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_conv = num_conv

        self.dropout = nn.Dropout(dropout)
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(self.mel_channels, self.hidden_dim,
                         kernel_size=self.kernel_size, stride=1,
                         padding=int((self.kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(self.hidden_dim))
        )

        for i in range(1, self.num_conv - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(self.hidden_dim,
                             self.hidden_dim,
                             kernel_size=self.kernel_size, stride=1,
                             padding=int(
                                 (self.kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(self.hidden_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(self.hidden_dim, self.mel_channels,
                         kernel_size=self.kernel_size, stride=1,
                         padding=int((self.kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(self.mel_channels))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = self.dropout(torch.tanh(self.convolutions[i](x)))
        x = self.dropout(self.convolutions[-1](x))

        return x
