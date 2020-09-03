#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import get_mel
from text import text_to_sequence
import hyperparams as hp
import librosa

class PrepareDataset(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir, out_dir=hp.prepared_data_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.
            out_dir  (string): Directory with all the prepared data.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir
        self.out_dir = out_dir
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sr)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0]) + '.wav'
        mel= np.asarray(get_mel(wav_name))

        text = self.landmarks_frame.iloc[idx, 1]
        text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
        
        sample = {'mel':mel, 'text': text}
        np.save(os.path.join(self.out_dir, self.landmarks_frame.iloc[idx, 0]), sample)

        return sample
    
if __name__ == '__main__':
    dataset = PrepareDataset(os.path.join(hp.data_dir,'metadata.csv'), os.path.join(hp.data_dir,'wavs'))
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=8)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
