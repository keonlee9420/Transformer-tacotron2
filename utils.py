#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import hyperparams as hp
from phonemizer import phonemize, separator


def build_phone_vocab(texts):
    vocab = {'<unk>': 0,
             '<blank>': 1,
             '<s>': 2,
             '</s>': 3, }
    idx = 4
    for text in texts:
        phoneset = [p for p in phoneme(text).split(' ') if p]

        for phone in phoneset:
            if phone.strip() not in vocab:
                vocab[phone.strip()] = idx
                idx += 1
    return vocab


def phoneme(text):
    return phonemize(text,
                     separator=separator.Separator(
                         word=' <wb> ', syllable=' <sylb> ', phone=' '),
                     preserve_punctuation=True)


def get_phoneme(text, vocab):
    to_phoneme = [vocab[p] for p in phoneme(text).split(' ') if p]
    return torch.tensor(to_phoneme)


def pad_seq(seqs, pad_token=hp.pad_token):
    padded_seqs = []
    max_len = max((seq.shape[0] for seq in seqs))
    for seq in seqs:
        seq_len = seq.shape[0]
        padded_seqs.append(np.pad(seq, (0, max_len - seq_len),
                                  mode='constant', constant_values=pad_token))
    return np.stack(padded_seqs)


def phoneme_batch(texts):
    vocab = build_phone_vocab(texts)
    phoneme_batch = torch.tensor(
        pad_seq([get_phoneme(text, vocab) for text in texts]))
    return phoneme_batch, vocab


def get_mel(audio_dir):
    y, sr = librosa.load(audio_dir, sr=hp.sr)

    # Trimmings
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    mag, _ = librosa.magphase(linear)
    mel = librosa.feature.melspectrogram(S=mag**2, sr=sr, n_mels=80)

    # to decibel
    mel = librosa.power_to_db(mel, ref=np.max)

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # as input
    mel_input = torch.from_numpy(mel.T)

    return mel_input


def pad_mel(mels, pad_token=hp.pad_token):
    padded_mels = []
    max_len = max((mel.shape[0] for mel in mels))
    for mel in mels:
        mel_len = mel.shape[0]
        padded_mels.append(np.pad(
            mel, [[0, max_len - mel_len], [0, 0]], mode='constant', constant_values=pad_token))
    return np.stack(padded_mels)


def mel_batch(audio_dirs):
    return torch.tensor(
            pad_mel([get_mel(audio_dir) for audio_dir in audio_dirs]))

def save_mel(idx, mel):
    plt.figure()
    librosa.display.specshow(np.array(mel), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig('fig{}.png'.format(idx), dpi=300)
