#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import hyperparams as hp

from torchtext import data, datasets
from utils import EOS


class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, trg_set=None, pad=hp.pad_token):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg_set is not None:
            trg = trg_set['trg']
            self.trg = trg[:, :-1, :]
            self.trg_y = trg[:, 1:, :]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.nframes = (self.trg_y.sum(dim=-1) != pad).data.sum()
            self.trg_stops = trg_set['trg_stops']
            # print("trg_stops.shape:", self.trg_stops.shape)
            self.stop_tokens = self.trg_stops[:, 1:, :]
            # print("stop_tokens.shape:", self.stop_tokens.shape)

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt.sum(dim=-1) != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(
            tgt.size(-2)).type_as(tgt_mask.data)
        return tgt_mask


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def data_gen(V, batch, nbatches, device):
    """Generate random data for a src-tgt copy task."""
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = data.to(device)
        tgt = data.to(device)
        yield Batch(src, tgt, 0)


def data_prepare_tt2(batch_size, nbatches, random=False, sequential=False):
    """Prepare data for a src-tgt copy task of tt2 given src and tgt batch."""
    from utils import get_sample_batch
    src, tgt, tgt_stops, vocab = [], [], [], None

    def _append(b):
        src.append(b['src'])
        tgt.append(b['tgt'])
        tgt_stops.append(b['tgt_stops'])

    # different(sequential) data in each batch
    if sequential:
        for i in range(nbatches):
            batch = get_sample_batch(batch_size, start=i * batch_size,
                                     vocab=vocab, random=random)
            _append(batch)
            vocab = batch['vocab']

    # same data in each batch
    else:
        batch = get_sample_batch(batch_size, vocab=vocab, random=random)
        _append(batch)
        vocab = batch['vocab']

        for _ in range(nbatches-1):
            if random:
                batch = get_sample_batch(
                    batch_size, vocab=vocab, random=random)
                vocab = batch['vocab']
            _append(batch)

    return {'src': src, 'tgt': tgt, 'tgt_stops': tgt_stops, 'vocab': vocab}


def data_gen_tt2(data, device):
    """Generate data for a src-tgt copy task of tt2 given src and tgt batch."""
    for phoneme_batch, mel_batch, mel_stops in zip(data['src'], data['tgt'], data['tgt_stops']):
        yield Batch(phoneme_batch.to(device), {'trg': mel_batch.to(device), 'trg_stops': mel_stops.to(device)})


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm


class SimpleTT2LossCompute:
    """A simple loss compute and train function for tt2."""

    def __init__(self, criterion, stop_criterion, opt=None):
        self.criterion = criterion
        self.stop = stop_criterion
        self.opt = opt

    def __call__(self, x, y, stop_x, stop_y, norm):
        # calculate stop loss including impose positive weight as Sec 3.7.
        # print("stop_x shape and dtype", stop_x.shape, stop_x.dtype, stop_x[:,-3:,:])
        # print("stop_y shape and dtype", stop_y.shape, stop_y.dtype, stop_y[:,-3:,:])
        stop_loss = self.stop(stop_x, stop_y)
        stop_loss[:, -1, :] *= hp.positive_stop_weight
        stop_loss = torch.mean(stop_loss)

        loss = self.criterion(x, y) + stop_loss
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
            # self.opt.zero_grad()
        return loss.data.item() * norm


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    """Fix order in torchtext to match ours"""
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


# class MultiGPULossCompute:
#     """A multi-gpu loss compute and train function."""

#     def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
#         # Send out to different gpus.
#         self.generator = generator
#         self.criterion = nn.parallel.replicate(criterion,
#                                                devices=devices)
#         self.opt = opt
#         self.devices = devices
#         self.chunk_size = chunk_size

#     def __call__(self, out, targets, normalize):
#         total = 0.0
#         generator = nn.parallel.replicate(self.generator,
#                                           devices=self.devices)
#         out_scatter = nn.parallel.scatter(out,
#                                           target_gpus=self.devices)
#         out_grad = [[] for _ in out_scatter]
#         targets = nn.parallel.scatter(targets,
#                                       target_gpus=self.devices)

#         # Divide generating into chunks.
#         chunk_size = self.chunk_size
#         for i in range(0, out_scatter[0].size(1), chunk_size):
#             # Predict distributions
#             out_column = [[Variable(o[:, i:i + chunk_size].data,
#                                     requires_grad=self.opt is not None)]
#                           for o in out_scatter]
#             gen = nn.parallel.parallel_apply(generator, out_column)

#             # Compute loss.
#             y = [(g.contiguous().view(-1, g.size(-1)),
#                   t[:, i:i + chunk_size].contiguous().view(-1))
#                  for g, t in zip(gen, targets)]
#             loss = nn.parallel.parallel_apply(self.criterion, y)

#             # Sum and normalize loss
#             l = nn.parallel.gather(loss,
#                                    target_device=self.devices[0])
#             l = l.sum()[0] / normalize
#             total += l.data[0]

#             # Backprop loss to output of transformer
#             if self.opt is not None:
#                 l.backward()
#                 for j, l in enumerate(loss):
#                     out_grad[j].append(out_column[j][0].grad.data.clone())

#         # Backprop all loss through transformer.
#         if self.opt is not None:
#             out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
#             o1 = out
#             o2 = nn.parallel.gather(out_grad,
#                                     target_device=self.devices[0])
#             o1.backward(gradient=o2)
#             self.opt.step()
#             self.opt.optimizer.zero_grad()
#         return total * normalize
