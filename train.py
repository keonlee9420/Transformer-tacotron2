#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py: Training Script for transformer-tacotron2
Keon Lee <keonlee9420@gmail.com>
Kyumin Park <pkm9403@gmail.com>

Usage:
    train.py [options]

Options:
    -h --help                               show this screen
    --cuda                                  use GPU
    --dataset=<name>                        specify data to train on  ['ljspeech' | '']
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from docopt import docopt
from tqdm import tqdm

import hyperparams as hp
from preprocess import DataLoader, collate_fn_transformer, get_data_dir, get_dataset
from schedule import NoamOpt, Batch, SimpleTT2LossCompute
from model import TransformerTTS
from plot_utils import plot_alignment_to_numpy


def main():
    args = docopt(__doc__)

    if not os.path.isdir(hp.checkpoint_path):
        os.mkdir(hp.checkpoint_path)

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    dataset = get_dataset(args['--dataset'])
    global_step = 0

    model = TransformerTTS().to(device)
    model = nn.DataParallel(model)

    model.train()
    model_opt = NoamOpt(hp.model_dim, 2, 4000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.999), eps=1e-9))

    criterion = nn.MSELoss()
    stop_criterion = nn.BCEWithLogitsLoss(reduction="none")
    criterion.to(device)
    stop_criterion.to(device)

    loss_compute = SimpleTT2LossCompute(criterion, stop_criterion, model_opt)

    writer = SummaryWriter(hp.log_dir)
    for epoch in range(hp.epochs):

        total_loss = 0
        total_frames = 0

        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=False, collate_fn=collate_fn_transformer,
                                drop_last=True, num_workers=16)
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d" % epoch)
            global_step += 1

            t3 = time.time()
            text, mel, stop_token, text_length = data

            text = text.to(device)
            mel = mel.to(device)
            stop_token = stop_token.to(device)
            text_length = text_length.to(device)

            batch = Batch(text, {'trg': mel, 'trg_stops': stop_token})

            out, stop_tokens, attn_enc, attn_dec, attn_endec = model.forward(batch.src, batch.trg, batch.src_mask,
                                                                             batch.trg_mask)
            loss = loss_compute(out, batch.trg_y.transpose(-2, -1),
                                stop_tokens, batch.stop_tokens, batch.nframes, model)
            total_loss += loss.data
            total_frames += batch.nframes
            pbar.set_postfix(loss='{0:.6f}'.format(total_loss / total_frames * 1000))

            writer.add_scalars('training_loss', {
                'loss': loss,
            }, global_step)

            writer.add_scalars('alphas', {
                'encoder_alpha': model.module.encoder.pos.alpha.data,
                'decoder_alpha': model.module.decoder.pos.alpha.data,
            }, global_step)

            if global_step % hp.image_step == 1:
                for ii, prob in enumerate(attn_endec):
                    num_h = prob.size(0)
                    for j in range(hp.num_heads):
                        # x = vutils.make_grid((prob[j * hp.batch_size] * 255))
                        x = plot_alignment_to_numpy((prob[j * hp.batch_size] * 255).cpu().detach().numpy())
                        writer.add_image('Attention_%d_0' % global_step, x, ii * hp.num_heads + j, dataformats='HWC')

                for ii, prob in enumerate(attn_enc):
                    num_h = prob.size(0)
                    for j in range(hp.num_heads):
                        # x = vutils.make_grid(prob[j * hp.batch_size] * 255)
                        x = plot_alignment_to_numpy((prob[j * hp.batch_size] * 255).cpu().detach().numpy())
                        writer.add_image('Attention_enc_%d_0' % global_step, x, ii * hp.num_heads + j, dataformats='HWC')

                for ii, prob in enumerate(attn_dec):
                    num_h = prob.size(0)
                    for j in range(hp.num_heads):
                        # x = vutils.make_grid(prob[j * hp.batch_size] * 255)
                        x = plot_alignment_to_numpy((prob[j * hp.batch_size] * 255).cpu().detach().numpy())
                        writer.add_image('Attention_dec_%d_0' % global_step, x, ii * hp.num_heads + j, dataformats='HWC')
                print('Update Attention! at global step: {}'.format(global_step))

            if global_step % hp.save_step == 0:
                torch.save({'model': model.state_dict(),
                            'optimizer': model_opt.optimizer.state_dict()},
                           os.path.join(hp.checkpoint_path, 'checkpoint_tt2_%d.pth.tar' % global_step))
                print('Saved! Model|checkpoint_tt2_{}.pth.tar, LOSS|{}'.format(global_step, loss))


if __name__ == '__main__':
    main()
