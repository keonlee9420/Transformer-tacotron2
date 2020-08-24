import torch
import torch.nn as nn
import numpy as np
import hyperparams as hp
from utils import *
from run import make_model, run_epoch

import os
temp_dir = os.path.join(hp.output_dir, 'temp_syns.pt')


def synthesize(model_saved_path, batch_one, vocab_size=hp.sample_vocab_size, device='cpu'):

    params = torch.load(model_saved_path)
    model = make_model(vocab_size, N=hp.num_layers)
    # comment out to compare raw-model to trained-model
    model.load_state_dict(params['state_dict'])
    model = model.to(device)
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
            out, stop_tokens = model.forward(batch_one.src, hints,
                                                batch_one.src_mask, trg_mask)
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
