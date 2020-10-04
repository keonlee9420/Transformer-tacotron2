#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import hyperparams as hp
from .modules import *


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    input shape: (batch, n_frames, mel_channels)
    output shape: (batch, mel_channels, n_frames)

    """

    def __init__(self,
                 N=hp.num_layers,
                 h=hp.num_heads,
                 dim_hidden=hp.model_dim,
                 dim_ffn=hp.d_ff,
                 dropout=hp.dropout,
                 mel_channels=hp.mel_channels,
                 dim_mel_hidden=hp.hidden_dim,
                 post_num_conv=hp.post_num_conv):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(h=h, dim_hidden=dim_hidden,
                                                  dim_ffn=dim_ffn, dropout=dropout) for _ in range(N)])
        self.pos = PositionalEncoding(dim_hidden, dropout)
        self.decoder_prenet = DecoderPrenet(
            mel_channels, dim_mel_hidden, dim_hidden, dropout)
        self.decoder_postnet = Postnet(dim_hidden, mel_channels, dim_mel_hidden, dropout, post_num_conv)

    def forward(self, x, memory, src_mask, tgt_mask):
        # prenet
        x = self.decoder_prenet(x)  # x: mel_batch

        # positional encoding
        x = self.pos(x)

        # decoder
        attn_dec, attn_endec = tuple(), tuple()
        for layer in self.layers:
            x, out_dec, out_cross = layer(x, memory, src_mask, tgt_mask)
            attn_dec += out_dec
            attn_endec += out_cross

        # postnet
        # (batch, mel_channels, n_frames)
        mels, stop_tokens = self.decoder_postnet(x)

        return mels, stop_tokens, attn_dec, attn_endec


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self,
                 h=hp.num_heads,
                 dim_hidden=hp.model_dim,
                 dim_ffn=hp.d_ff,
                 dropout=hp.dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = SublayerConnection(MultiHeadAttention(h=h, dim_hidden=dim_hidden, dropout=dropout),
                                            dim_hidden=dim_hidden, dropout=dropout)
        self.cross_attn = SublayerConnection(MultiHeadAttention(h=h, dim_hidden=dim_hidden, dropout=dropout),
                                             dim_hidden=dim_hidden, dropout=dropout)
        self.feed_forward = SublayerConnection(PositionwiseFeedForward(dim_hidden=dim_hidden,
                                                                       dim_ffn=dim_ffn, dropout=dropout),
                                               dim_hidden=dim_hidden, dropout=dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        out_self, dec_attn = self.self_attn(x, mask=tgt_mask)
        out_cross, cross_attn = self.cross_attn(out_self, mask=src_mask, memory=memory)
        out = self.feed_forward(out_cross)

        return out, dec_attn, cross_attn


class DecoderPrenet(nn.Module):
    """
    Decoder pre-net for TTS
    For the alignment between phoneme and mel frame to be measured
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(DecoderPrenet, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.projection = nn.Linear(output_dim, output_dim)

    def forward(self, decoder_input):
        out = self.layer(decoder_input)
        out = self.projection(out)
        return out


class Postnet(nn.Module):
    def __init__(self,
                 dim_hidden=hp.model_dim,
                 mel_channels=hp.mel_channels,
                 dim_mel_hidden=hp.hidden_dim,
                 dropout=hp.dropout,
                 n_conv=hp.post_num_conv):
        super(Postnet, self).__init__()
        self.mel_linear = nn.Linear(dim_hidden, mel_channels)
        self.stop_linear = nn.Linear(dim_hidden, 1)

        self.post_conv = nn.ModuleList()
        self.post_conv.append(
            ConvNorm(mel_channels, dim_mel_hidden, kernel_size=5,
                     batch_norm=True, activation='tanh', dropout=dropout)
        )
        for _ in range(n_conv - 2):
            self.post_conv.append(
                ConvNorm(dim_mel_hidden, dim_mel_hidden, kernel_size=5,
                         batch_norm=True, activation='tanh', dropout=dropout)
            )
        self.post_conv.append(
            ConvNorm(dim_mel_hidden, mel_channels, kernel_size=5,
                     batch_norm=True, activation=None, dropout=dropout)
        )

    def forward(self, x):
        mel_out = self.mel_linear(x)
        stop_out = self.stop_linear(x)

        conv_out = mel_out.clone()
        for conv_layer in self.post_conv:
            conv_out = conv_layer(conv_out)
        mel_out += conv_out     # Residual path

        return mel_out, stop_out



