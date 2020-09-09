import copy

import torch.nn as nn

import hyperparams as hp
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .attention import MultiHeadedAttention
from .modules import Embeddings, PositionwiseFeedForward


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        memory, attn_enc = self.encode(src, src_mask)
        mels, stop_tokens, attn_dec, attn_endec = self.decode(memory, src_mask, tgt, tgt_mask)
        return mels, stop_tokens, attn_enc, attn_dec, attn_endec

    def encode(self, src, src_mask):
        # print("ENCODER INPUT shape:", self.src_embed(src).shape)
        # print("ENCODER OUTPUT shape:", self.encoder(self.src_embed(src), src_mask).shape)
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # print("DECODER INPUT shape:", self.tgt_embed(tgt).shape)
        # print("DECODER OUTPUT shape:", self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask).shape)
        return self.decoder(tgt, memory, src_mask, tgt_mask)


def make_model(src_vocab=hp.num_embeddings, N=hp.num_layers,
               d_model=hp.model_dim, d_ff=hp.d_ff, h=hp.num_heads, dropout=hp.model_dropout):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        Embeddings(d_model, src_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model