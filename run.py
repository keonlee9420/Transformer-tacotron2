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
    --simple-tt2                            train with a simple copy-task for Transformer-TTS
    --epoch=<int>                           simple-tt2 epoch
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


def run_epoch(data_iter, model, loss_compute, begin_time):
    """Standard Training and Logging Function"""
    start = time.time()
    total_frames = 0
    total_loss = 0
    frames = 0
    for i, batch in enumerate(data_iter):
        # print("INNER run_epoch: ", batch.src.shape, batch.trg.shape)
        # print("INNER run_epoch: ", batch.src_mask.shape, batch.trg_mask.shape)
        out, stop_tokens, attn_enc, attn_dec, attn_endec = model.forward(batch.src, batch.trg,
                                         batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y.transpose(-2, -1),
                            stop_tokens, batch.stop_tokens, batch.nframes, model)
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

    if args['--simple-tt2']:
        """
        We can begin by trying out a simple copy-task.
        Given a ordered set of input phoneme and mel, 
        the goal is to generate back those same mel from text.
        """
        import os
        from synthesize import synthesize

        epoch = 10
        if args['--epoch']:
            epoch = int(args['--epoch'])
        batch_size = 3
        nbatches = 5
        print("batch_size, nbatches:", batch_size, nbatches)

        data = None
        sequential = True
        data_dir = os.path.join(
            hp.weight_dir, 'prepared-data-{}-{}_{}.pt'.format(batch_size, nbatches, 'seq' if sequential else 'same'))
        if not os.path.isfile(data_dir):
            print("Prepare Dataset")
            data = data_prepare_tt2(
                batch_size, nbatches, sequential=sequential)
            torch.save({'data': data}, data_dir)
        else:
            data = torch.load(data_dir)['data']
            print("Loaded Prepared Dataset!")
        # exit("END OF TEST")

        criterion = nn.MSELoss()
        # nn.BCELoss(reduction="none") nn.BCEWithLogitsLoss(reduction="none")
        stop_criterion = nn.BCEWithLogitsLoss(reduction="none")
        criterion.to(device)
        stop_criterion.to(device)
        model = make_model(len(data['vocab']), N=hp.num_layers)
        model = model.to(device)

        keep = False
        lr = 'NoamOpt'
        model_opt = NoamOpt(hp.model_dim, 2, 4000,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.999), eps=1e-9))

        # lr = 3e-3 #2e-5 'NoamOpt'  # hp.lr
        # # , betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5
        # my_optim = torch.optim.Adam(model.parameters(), lr)
        # my_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(my_optim,
        #                                                       max_lr=lr,
        #                                                       steps_per_epoch=int(
        #                                                           nbatches),
        #                                                       epochs=epoch,
        #                                                       anneal_strategy='cos',
        #                                                       pct_start=0.5,
        #                                                       base_momentum=0.75,
        #                                                       max_momentum=0.99,
        #                                                       div_factor=1,
        #                                                       final_div_factor=1e4)
        # my_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(my_optim,
        #                                                              factor=0.5,
        #                                                              patience=10,
        #                                                              cooldown=10,
        #                                                              threshold=1e-6,
        #                                                              threshold_mode='abs')
        # model_opt = CustomAdam(my_optim)

        # train
        run = True
        begin_time = time.time()
        model_save_path = hp.weight_dir
        output_save_path = hp.output_dir
        model_pt_filename = 'simple-tt2-{}-{}-{}_{}-loss{}.pt'.format(
            epoch, batch_size, nbatches, str(lr), hp.loss_w_stop)
        model_saved_path = os.path.join(model_save_path, model_pt_filename)
        print("model_saved_path:", model_saved_path)

        # setup directories
        if not os.path.isdir(model_save_path):
            os.mkdir(model_save_path)
        if not os.path.isdir(output_save_path):
            os.mkdir(output_save_path)

        if os.path.isfile(model_saved_path):
            params = torch.load(model_saved_path)
            best_lowest_loss = params['best_lowest_loss']
            model_opt.optimizer.load_state_dict(
                params['optimizer_state_dict'])
            model.load_state_dict(params['state_dict'])
            model = model.to(device)
            print("Loaded! best_lowest_loss:\n", best_lowest_loss)
            if not keep:
                run = False

        best_lowest_loss = float('inf')
        print(
            "\n\n--------------- train start with EPOCH: {}---------------".format(epoch))
        if run:
            total_epoch = epoch
            for epoch in range(epoch):
                model.train()
                run_epoch(data_gen_tt2(data, device=device), model,
                          SimpleTT2LossCompute(criterion, stop_criterion, model_opt), begin_time)
                model.eval()
                loss = run_epoch(data_gen_tt2(data, device=device), model,
                                 SimpleTT2LossCompute(criterion, stop_criterion, None), begin_time)
                print("{}th epoch:".format(epoch+1), loss)

                if epoch > total_epoch*0.5 and best_lowest_loss > loss:
                    best_lowest_loss = loss
                    torch.save({'state_dict': model.state_dict(), 'optimizer_state_dict': model_opt.optimizer.state_dict(
                    ), 'best_lowest_loss': best_lowest_loss}, model_saved_path)
                    print("Saved the best-model!")

        # load the best model and synthesize with it
        params = torch.load(model_saved_path)
        model.load_state_dict(params['state_dict'])
        model_opt.optimizer.load_state_dict(params['optimizer_state_dict'])
        model = model.to(device)
        model.eval()
        print("Load the best-model and test:", run_epoch(data_gen_tt2(data, device=device), model,
                                                         SimpleTT2LossCompute(criterion, stop_criterion, None), begin_time))
        print("best_lowest_loss:\n", params['best_lowest_loss'])
        print("train done!\n")

        # Synthesize using the very first data
        print("\n--------------- synthesize! ---------------\n")
        # device='cpu' # for debugging

        # synthesize
        synthesize(model_saved_path, next(data_gen_tt2(data, device=device)), len(data['vocab']), device)

        # test sampling
        with torch.no_grad():
            # model = make_model(hp.sample_vocab_size, N=hp.num_layers)
            # model.to(device)
            for i, batch in enumerate(data_gen_tt2(data, device=device)):
                out, stop_tokens = model.forward(batch.src, batch.trg,
                                                 batch.src_mask, batch.trg_mask)
                # print(out.shape)
                # directly save every teacher-forced output
                for b in range(out.shape[0]):
                    print(out[b, :, :-1].unsqueeze(0).shape)
                    wav = mel_to_wav(
                        out[b, :, :-1].unsqueeze(0), filename="output_{}_{}".format(i+1, b+1))
                    save_wav(wav, 'wav_output_{}_{}'.format(i+1, b+1))

                print(
                    "\n--------------- reconstruct mel to wave under same converter ---------------")
                wav_original = mel_to_wav(batch.trg.transpose(
                    -2, -1)[0, :, 1:].unsqueeze(0), filename="reconstruct")
                save_wav(wav_original, 'wav_reconstruct')

                # print(
                #     "\n--------------- source conversion only using librosa ---------------")
                # wav_source = mel_to_wav(
                #     get_mel('./outputs/samples/LJ001-0001.wav').T.unsqueeze(0), filename="source")
                # save_wav(wav_source, 'wav_source')

    else:
        print("\n\nWhat else?\n\n")
