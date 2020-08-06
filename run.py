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
"""

import sys
import time
from docopt import docopt

from attention import *
from model import *
from encoder import *
from decoder import *
from schedule import *

import spacy

def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def run_epoch(data_iter, model, loss_compute, begin_time):
    """Standard Training and Logging Function"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss.data
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f Total Sec: %.2f" %
                  (i, loss / batch.ntokens, tokens / elapsed, time.time() - begin_time))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


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
            run_epoch(data_gen(V, 30, 20, device=device), model,
                      SimpleLossCompute(model.generator, criterion, model_opt), begin_time)
            model.eval()
            print(run_epoch(data_gen(V, 30, 5, device=device), model,
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
        criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
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
        for epoch in range(10):
            model_par.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter),
                      model_par,
                      SimpleLossCompute(model.generator, criterion,
                                        opt=model_opt), begin_time)
            model_par.eval()
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                             model_par,
                             SimpleLossCompute(model.generator, criterion, opt=None), begin_time)
            print(loss)

    else:
        print("\n\nWhat else?\n\n")
