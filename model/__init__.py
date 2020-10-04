import copy

import torch.nn as nn

import hyperparams as hp
from .encoder import Encoder
from .decoder import Decoder


class TransformerTTS(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self,
                 N=hp.num_layers,
                 h=hp.num_heads,
                 dim_embedding=hp.model_dim,
                 dim_hidden=hp.model_dim,
                 dim_ffn=hp.d_ff,
                 dropout=hp.dropout,
                 num_embeddings=hp.num_embeddings,
                 mel_channels=hp.mel_channels,
                 dim_mel_hidden=hp.hidden_dim,
                 post_num_conv=hp.post_num_conv):
        super(TransformerTTS, self).__init__()
        self.encoder = Encoder(N=N, h=h,
                               dim_hidden=dim_hidden,
                               dim_ffn=dim_ffn,
                               dim_embedding=dim_embedding,
                               dropout=dropout,
                               num_embeddings=num_embeddings)
        self.decoder = Decoder(N=N, h=h,
                               dim_hidden=dim_hidden,
                               dim_ffn=dim_ffn,
                               dropout=dropout,
                               mel_channels=mel_channels,
                               dim_mel_hidden=dim_mel_hidden,
                               post_num_conv=post_num_conv)

        self._initialize_weight(self.encoder)
        self._initialize_weight(self.decoder)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        memory, attn_enc = self.encode(src, src_mask)
        mels, stop_tokens, attn_dec, attn_endec = self.decode(memory, src_mask, tgt, tgt_mask)
        return mels, stop_tokens, attn_enc, attn_dec, attn_endec

    def encode(self, src, src_mask):
        # print("ENCODER INPUT shape:", self.src_embed(src).shape)
        # print("ENCODER OUTPUT shape:", self.encoder(self.src_embed(src), src_mask).shape)
        return self.encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # print("DECODER INPUT shape:", self.tgt_embed(tgt).shape)
        # print("DECODER OUTPUT shape:", self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask).shape)
        return self.decoder(tgt, memory, src_mask, tgt_mask)

    @staticmethod
    def _initialize_weight(module):
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
