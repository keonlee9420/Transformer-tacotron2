#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import copy
import math
import hyperparams as hp

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

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


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


if __name__ == '__main__':
    import attention
    import model
    import numpy as np

    vocab = utils.build_phone_vocab(['hello world!'])
    print(vocab)
    embed = Embeddings(512, len(vocab))
    prenet = EncoderPrenet(512, 512, 0.1)
    print(prenet)

    c = copy.deepcopy
    attn = attention.MultiHeadedAttention(8, 512)
    ff = model.PositionwiseFeedForward(512, 256, 0.1)

    encoder = Encoder(EncoderLayer(512, c(attn), c(ff), 0.1), 6)

    model = nn.ModuleList([embed, prenet, encoder])
    x = [p for p in utils.phoneme('hello world!').split(' ') if p]
    t = np.array([np.zeros(len(vocab)) for _ in x])
    for i, p in enumerate(x):
        t[i][vocab[p]] = 1

    t = torch.tensor(t, dtype=torch.long)
    print(t.size())
    emb = embed(t).unsqueeze(0)
    print(emb.size())
    pre = prenet(emb)
    print(pre.size())
    enc = encoder(pre)
    print(enc.size())
