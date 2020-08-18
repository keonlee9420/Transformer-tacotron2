import torch
import torch.nn as nn
import numpy as np
import hyperparams as hp
from utils import *
from run import make_model, run_epoch


def synthesize(model_saved_path, batch_one, device='cpu'):

    params = torch.load(model_saved_path)
    model = make_model(hp.sample_vocab_size, N=hp.num_layers)
    model.load_state_dict(params['state_dict']) # comment out to compare raw-model to trained-model
    model = model.to(device)
    model.eval()

    # mel_inout = torch.empty((1, 1, 80)).fill_(BOS).to(device)
    actual_len = batch_one.trg.shape[1]
    given_hints = 385
    mel_inout = batch_one.trg[:, :given_hints, :]

    with torch.no_grad():
        for _ in range(actual_len-given_hints):
            trg_mask = batch_one.make_std_mask(mel_inout, hp.pad_token)
            out, stop_tokens = model.forward(batch_one.src, mel_inout,
                                             batch_one.src_mask, trg_mask)
            mel_inout = torch.cat(
                (mel_inout, out.transpose(-2, -1)[:, -1:, :]), dim=1)
            print(mel_inout.shape, trg_mask.shape, out.shape)
        score = nn.functional.softmax(stop_tokens, dim=1)
        # print(torch.mean(score), score[:, -1, :])

        val, idx = torch.max(stop_tokens.to('cpu'), dim=1)
        print(stop_tokens, idx, val,
              score[:, idx.data.item()], torch.max(score, dim=1)[0])
        print("output shape, trg actual shape:\n",
              mel_inout.shape, batch_one.trg.shape)
        wav = mel_to_wav(out, save_mel=True)
