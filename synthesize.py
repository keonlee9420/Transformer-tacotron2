#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import hyperparams as hp
from utils import *
from run import make_model, run_epoch
from schedule import data_gen_tt2

import os
temp_dir = os.path.join(hp.output_dir, 'temp_syns.pt')


def synthesize(model, batch_one, vocab_size=hp.sample_vocab_size, device='cpu'):

    # params = torch.load(model_saved_path)
    # model = make_model()
    # # comment out to compare raw-model to trained-model
    # model.load_state_dict(params['model'])
    # model = model.to(device)
    model.eval()

    actual_len = batch_one.trg.shape[1]
    given_hints = 0
    # mel_inout = torch.empty((batch_one.trg.shape[0], 1, 80)).fill_(BOS).to(device)
    mel_inout = batch_one.trg[:, :given_hints+1, :]

    with torch.no_grad():

        # if os.path.isfile(temp_dir):
        #     temp = torch.load(temp_dir)
        #     out = temp['out']
        #     stop_tokens = temp['stop_tokens']
        #     mel_inout = temp['mel_inout']
        # else:
        for i in range(actual_len-given_hints):
            # if i % 1 == 0:  # if condition is i%1 == 'teacher forcing'
            #     hints = batch_one.trg[:, :given_hints+i+1, :]
            # else:
            hints = mel_inout[:, :given_hints+i+1, :]
            # hints = mel_inout
            trg_mask = batch_one.make_std_mask(hints, hp.pad_token)
            out, stop_tokens, attn_enc, attn_dec, attn_endec = model.forward(batch_one.src, batch_one.trg, batch_one.src_mask,
                                                                             batch_one.trg_mask)
            mel_inout = torch.cat(
                (mel_inout, out.transpose(-2, -1)[:, -1:, :]), dim=1)
            print(hints.shape, mel_inout.shape, trg_mask.shape, out.shape)
        # print(torch.mean(score), score[:, -1, :])
        # torch.save({'out': out, 'stop_tokens': stop_tokens,
        #             'mel_inout': mel_inout}, temp_dir)

        score = nn.functional.softmax(stop_tokens, dim=1)
        val, idx = torch.max(stop_tokens.to('cpu'), dim=1)
        print(stop_tokens, idx, val,
              score[:, idx], torch.max(score, dim=1)[0])
        print("output shape, trg actual shape:\n",
              out.shape, batch_one.trg_y.shape)

        # save results
        for b in range(out.shape[0]):
            print(out[b, :, :-1].unsqueeze(0).shape)
            wav = mel_to_wav(out[b, :, :-1].unsqueeze(0), filename="syn_{}".format(b+1))
            save_wav(wav, 'wav_syn_{}'.format(b+1))

if __name__ == '__main__':
    model_saved_path = './checkpoint/checkpoint_tt2_4304000.pth.tar'
    device = 'cuda'
    model = make_model()
    model = nn.DataParallel(model.to(device))
    # load the best model and synthesize with it
    params = torch.load(model_saved_path)
    model.load_state_dict(params['model'])

    model.eval()

    data_dir = './weights/prepared-data-2-5_seq.pt'
    data = torch.load(data_dir)['data']

    # Synthesize using the very first data
    print("\n--------------- synthesize! ---------------\n")
    # synthesize
    synthesize(model, next(data_gen_tt2(data, device=device)), len(data['vocab']), device)

    # test sampling
    with torch.no_grad():
        # model = make_model(hp.sample_vocab_size, N=hp.num_layers)
        # model.to(device)
        for i, batch in enumerate(data_gen_tt2(data, device=device)):
            out, stop_tokens, attn_enc, attn_dec, attn_endec = model.forward(batch.src, batch.trg, batch.src_mask,
                                                                             batch.trg_mask)
            # print(out.shape)
            # directly save every teacher-forced output
            for b in range(out.shape[0]):
                print(out[b, :, :-1].unsqueeze(0).shape)
                wav = mel_to_wav(
                    out[b, :, :-1].unsqueeze(0), filename="output_{}_{}".format(i + 1, b + 1))
                save_wav(wav, 'wav_output_{}_{}'.format(i + 1, b + 1))

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
