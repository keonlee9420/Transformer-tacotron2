#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from model import PositionalEncoding
from encoder import clones, LayerNorm, ConvNorm, SublayerConnection, sample_encoding
import hyperparams as hp


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.pos = PositionalEncoding(hp.model_dim, hp.model_dropout)
        self.decoder_prenet = DecoderPrenet(
            hp.mel_channels, hp.hidden_dim, hp.model_dim, hp.pre_dropout)
        self.mel_linear = MelLinear(hp.model_dim, hp.mel_channels)
        self.stop_linear = StopLinear(hp.model_dim)
        self.decoder_postnet = Postnet(hp.mel_channels, hp.hidden_dim)

    def forward(self, x, memory, src_mask, tgt_mask):
        # prenet
        x = self.decoder_prenet(x)  # x: mel_batch

        # positional encoding
        x = self.pos(x)

        # decoder
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.norm(x)

        # mel linear
        mel_linear = self.mel_linear(x)

        # stop linear
        stop_tokens = self.stop_linear(x)

        # postnet
        mels = self.decoder_postnet(mel_linear.transpose(-2, -1))

        return mels, stop_tokens


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


class Linear(nn.Module):
    """Linear Module."""

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class MelLinear(nn.Module):
    """Linear projections to predict the mel spectrogram same as Tacotron2."""

    def __init__(self, num_hidden, mel_channels):
        super(MelLinear, self).__init__()
        self.mel_linear = Linear(num_hidden, mel_channels)

    def forward(self, decoder_input):
        return self.mel_linear(decoder_input)


class StopLinear(nn.Module):
    """Linear projections to predict the stop token same as Tacotron2."""

    def __init__(self, num_hidden):
        super(StopLinear, self).__init__()
        self.stop_linear = Linear(num_hidden, 1, w_init='sigmoid')

    def forward(self, decoder_input):
        return self.stop_linear(decoder_input)


class Postnet(nn.Module):
    """Linear projections to produce a residual to refine the reconstruction of mel spectrogram same as Tacotron2."""

    def __init__(self, mel_channels, hidden_dim, kernel_size=hp.post_kernel_size, num_conv=hp.post_num_conv, dropout=hp.post_dropout):
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


if __name__ == "__main__":
    from utils import *

    PAD_TOKEN = 0
    # START_TOKEN = 2
    # END_TOKEN = 3

    sample_batch = 10
    print("TOTAL BATCH: {}".format(sample_batch))

    print("\n-------------- mel-preprocessing --------------")
    audio_dirs = ['/home/keon/speech-datasets/LJSpeech-1.1/wavs/LJ001-{}.wav'
                  .format((4-len(str(i+1)))*'0' + str(i+1)) for i in range(sample_batch)]

    mel_batch = torch.tensor(
        pad_mel([get_mel(audio_dir) for audio_dir in audio_dirs], pad_token=PAD_TOKEN))
    mal_maxlen = mel_batch.shape[1]
    # (batch, n_frames, mel_channels)
    print("mel_batch.shape:\n", mel_batch.shape)

    # # save spectrogram
    # for i in range(mel_batch.shape[0]):
    #     print("{}th mel_spec saved".format(i+1), mel_batch[i].T.shape)
    #     save_mel(i+1, mel_batch[i].T)
    # print("Save ALL!\n")

    from attention import *
    from model import *
    from encoder import *
    from decoder import *
    import hyperparams as hp

    c = copy.deepcopy
    attn = MultiHeadedAttention(8, hp.model_dim)
    ff = PositionwiseFeedForward(hp.model_dim, hp.hidden_dim, 0.1)
    position = PositionalEncoding(hp.model_dim, 0.1)
    decoder = Decoder(DecoderLayer(
        hp.model_dim, c(attn), c(attn), c(ff), 0.1), 6)

    print("\n-------------- encoder --------------")
    # sample encoding
    memory, seq_maxlen = sample_encoding(sample_batch)
    # encoder output, (batch, n_sequences, model_dim)
    print("memory.shape:\n", memory.shape)

    print("\n-------------- decoder --------------")
    mels, stop_tokens = decoder(mel_batch, memory,
                                torch.ones((sample_batch, 1, seq_maxlen)), torch.ones((sample_batch, mal_maxlen, mal_maxlen)))
    print("decoderoutput mels, stop_tokens\n:", mels.shape, stop_tokens.shape)

    # print("\n-------------- pre-decoder --------------")
    # decoder_input = decoder.decoder_prenet(mel_batch)
    # print("decoder_input.shape:\n", decoder_input.shape) # (batch, n_frames, model_dim)

    # print("\n-------------- decoder --------------")
    # # memory = torch.ones((sample_batch, seq_maxlen, hp.model_dim)) # only for decoder without encoder
    # print("memory.shape:\n", memory.shape) # encoder output, (batch, n_sequences, model_dim)

    # mal_maxlen = mel_batch.shape[1]
    # # from schedule import *
    # # batch = Batch(torch.ones((sample_batch, seq_maxlen)), torch.ones((sample_batch, mal_maxlen)))
    # # print("Batch.src, trg: ", batch.src.shape, batch.trg.shape)
    # # print("Batch.src_mask, trg_mask: ", batch.src_mask.shape, batch.trg_mask.shape)
    # decoder_input = decoder(decoder_input, memory, \
    #     torch.ones((sample_batch, 1, seq_maxlen)), torch.ones((sample_batch, mal_maxlen, mal_maxlen)))
    # print("decoder OUTPUT.shape:\n:", decoder_input.shape)

    # print("\n-------------- post-decoder --------------")
    # mel_linear_output = decoder.mel_linear(decoder_input)
    # print("MEL LINEAR~", mel_linear_output.shape)
    # stop_linear = decoder.stop_linear(decoder_input)
    # print("STOP LINEAR~", stop_linear.shape)

    # decoder_postnet = decoder.decoder_postnet(mel_linear_output.transpose(-2, -1))
    # print("decoder_postnet.shape:", decoder_postnet.shape)
