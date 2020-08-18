#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py: Run Script for transformer-tacotron2
Keon Lee <keonlee9420@gmail.com>
Kyumin Park <pkm9403@gmail.com>

Usage:
    run.py [options]

Options:
    -h --help                               show this screen
    --cuda                                  use GPU
    --simple-train                          train with a simple copy-task
    --spacy-train                           train with spacy en-de data
    --simple-tt2                            train with a simple copy-task for Transformer-TTS
"""

import sys
import time
from docopt import docopt
import torch
import torch.nn as nn
from torchtext import data, datasets

from attention import *
from model import *
from encoder import *
from decoder import *
from schedule import *
from utils import *
import hyperparams as hp

import spacy


def make_model(src_vocab, N=hp.num_layers,
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


def run_epoch(data_iter, model, loss_compute, begin_time):
    """Standard Training and Logging Function"""
    start = time.time()
    total_frames = 0
    total_loss = 0
    frames = 0
    for i, batch in enumerate(data_iter):
        # print("INNER run_epoch: ", batch.src.shape, batch.trg.shape)
        # print("INNER run_epoch: ", batch.src_mask.shape, batch.trg_mask.shape)
        out, stop_tokens = model.forward(batch.src, batch.trg,
                                         batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y.transpose(-2, -1),
                            stop_tokens, batch.stop_tokens, batch.nframes)
        total_loss += loss.data
        total_frames += batch.nframes
        frames += batch.nframes
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Batch: %d Loss: %f Franmes per Sec: %f Total Sec: %.2f" %
                  (i, loss / batch.nframes, frames / elapsed, time.time() - begin_time))
            start = time.time()
            frames = 0
    return total_loss / total_frames


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens + padding."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


if __name__ == "__main__":
    args = docopt(__doc__)

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    # Greedy Decoding
    # Train the simple copy-task.
    if args['--simple-train']:
        """
        We can begin by trying out a simple copy-task.
        Given a random set of input symbols from a small vocabulary, 
        the goal is to generate back those same symbols.
        """
        V = 11
        criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
        criterion.to(device)
        model = make_model(V, V, N=2)
        model = model.to(device)
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        # train
        begin_time = time.time()
        for epoch in range(10):
            model.train()
            run_epoch(data_gen(V, hp.batch_size, 20, device=device), model,
                      SimpleLossCompute(model.generator, criterion, model_opt), begin_time)
            model.eval()
            print(run_epoch(data_gen(V, hp.batch_size, 5, device=device), model,
                            SimpleLossCompute(model.generator, criterion, None), begin_time))

        # test
        model.eval()
        model.to("cpu")
        src = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
        src_mask = torch.ones(1, 1, 10)
        print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

    elif args['--spacy-train']:
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')

        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        BOS_WORD = '<s>'
        EOS_WORD = '</s>'
        BLANK_WORD = "<blank>"
        SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
        TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                         eos_token=EOS_WORD, pad_token=BLANK_WORD)

        MAX_LEN = 100
        train, val, test = datasets.IWSLT.splits(
            exts=('.de', '.en'), fields=(SRC, TGT),
            filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
            len(vars(x)['trg']) <= MAX_LEN)
        MIN_FREQ = 2
        SRC.build_vocab(train.src, min_freq=MIN_FREQ)
        TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

        pad_idx = TGT.vocab.stoi["<blank>"]
        model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
        model.cuda()
        criterion = LabelSmoothing(
            size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
        criterion.cuda()
        BATCH_SIZE = 12000
        train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=device,
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=True)
        valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=device,
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=False)
        model_par = nn.DataParallel(model, device_ids=[0])

        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        begin_time = time.time()
        for epoch in range(100):
            print(f'Epoch {epoch}')
            model_par.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter),
                      model_par,
                      SimpleLossCompute(model.generator, criterion,
                                        opt=model_opt), begin_time)
            model_par.eval()
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                             model_par,
                             SimpleLossCompute(model.generator, criterion, opt=None), begin_time)
            print(f'Loss: {loss}')

        torch.save((model, SRC, TGT), './weights/spacy-en-de-100.pt')
        model = model.to('cpu')

        # Example
        model.eval()
        sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split()
        src = torch.tensor([[SRC.vocab.stoi[w]
                             for w in sent]], dtype=torch.long)
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        print("Translation:", end="\t")
        trans = "<s> "
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>":
                break
            trans += sym + " "
        print(trans)

    elif args['--simple-tt2']:

        # forward testing
        phoneme_batch, mel_batch, mel_stops, vocab = get_sample_batch(3)
        model = make_model(len(vocab), N=hp.num_layers)
        batch = Batch(phoneme_batch, {
                      'trg': mel_batch, 'trg_stops': mel_stops})
        print("batch.stop_tokens.shape:", batch.stop_tokens.shape)
        mels, stop_tokens = model.forward(
            batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        print("decoder_output mels, stop_tokens:\n{}, {}\n\n".format(
            mels.shape, stop_tokens.shape))

        """
        We can begin by trying out a simple copy-task.
        Given a ordered set of input phoneme and mel, 
        the goal is to generate back those same mel from text.
        """
        batch_size = 1
        nbatches = 1
        data = data_prepare_tt2(batch_size, nbatches, random=False)
        print(data['vocab'])

        criterion = nn.MSELoss()
        # nn.BCELoss(reduction="none") nn.BCEWithLogitsLoss(reduction="none")
        stop_criterion = nn.BCEWithLogitsLoss(reduction="none")
        criterion.to(device)
        stop_criterion.to(device)
        model = model.to(device)
        model_opt = NoamOpt(hp.model_dim, 1, 400,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.999), eps=1e-9))

        # train
        begin_time = time.time()
        for epoch in range(50):
            model.train()
            run_epoch(data_gen_tt2(data, device=device), model,
                      SimpleTT2LossCompute(criterion, stop_criterion, model_opt), begin_time)
            model.eval()
            print(run_epoch(data_gen_tt2(data, device=device), model,
                            SimpleTT2LossCompute(criterion, stop_criterion, None), begin_time))

        # test
        # model.eval()
        # src = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
        # src_mask = torch.ones(1, 1, 10)
        # print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

    else:
        print("\n\nWhat else?\n\n")
