#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hyperparams as hp
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils import *
import os
import librosa
import numpy as np
from text import text_to_sequence
import math
import collections


class LJDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir, out_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.
            out_dir  (string): Directory with all the prepared data.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir
        self.out_dir = out_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sr)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        prepared_data = np.load(os.path.join(self.out_dir, self.landmarks_frame.iloc[idx, 0] + '.npy'), allow_pickle=True)

        mel = prepared_data.item().get('mel')
        text = prepared_data.item().get('text')
        text_length = len(text)

        sample = {'mel': mel, 'text': text, 'text_length': text_length}

        return sample


def collate_fn_transformer(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):

        text = [d['text'] for d in batch]
        mel = [d['mel'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        
        text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        mel = [i for i, _ in sorted(zip(mel, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)

        # padding
        text = pad_seq(text).astype(np.int32)
        mel, stop_token = pad_mel(mel)

        return torch.LongTensor(text), torch.FloatTensor(mel),  torch.FloatTensor(stop_token), torch.LongTensor(text_length)
    
    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                .format(type(batch[0]))))


def get_data_dir(d_name):
    if not d_name:
        d_dir = 'LJSpeech-1.1'
    # elif d_name=='libritts':
    #     d_dir = 'LibriTTS'
    else:
        raise NotImplementedError(hp.load_error_msg)

    return os.path.join(hp.prepared_data_dir, d_dir)


def get_dataset(prepared_data_dir):
    data_dir = get_data_dir(prepared_data_dir)
    print('use dataset: %s' % data_dir.split("/")[-1])
    return LJDatasets(os.path.join(hp.data_dir, 'metadata.csv'), os.path.join(hp.data_dir, 'wavs'), data_dir)