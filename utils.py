#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import hyperparams as hp
from phonemizer import phonemize, separator

UNK = 0
BLANK = 1
BOS = 2
EOS = 3


def build_phone_vocab(texts, vocab=None):
    if vocab is None:
        vocab = {'<unk>': UNK,
                 '<blank>': BLANK,
                 '<s>': BOS,
                 '</s>': EOS, }
    idx = len(vocab)
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


def phoneme_batch(texts, vocab=None):
    vocab = build_phone_vocab(texts, vocab)
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
    mel_input = torch.from_numpy(mel.T)  # (n_frames, mel_chennels)

    return mel_input


def _add_ends(mel):
    return np.pad(mel, [[1, 1], [0, 0]], mode='constant', constant_values=[BOS, EOS])


def pad_mel(mels, pad_token=hp.pad_token):
    padded_mels = []
    stop_tokens = []
    _mels = [_add_ends(mel) for mel in mels]
    max_len = max((mel.shape[0] for mel in _mels))
    for i, mel in enumerate(_mels):
        mel_len = mel.shape[0]
        padded_mels.append(np.pad(
            mel, [[0, max_len - mel_len], [0, 0]], mode='constant', constant_values=pad_token))
        stop_token = np.zeros((max_len, 1))  # base on padded size
        stop_token[mel_len-1] = 1.  # end of actual size
        stop_tokens.append(stop_token)
    return np.stack(padded_mels), np.stack(stop_tokens)


def mel_batch(audio_dirs):
    # (batch, n_frames, mel_channels)
    padded_mels, stop_tokens = pad_mel(
        [get_mel(audio_dir) for audio_dir in audio_dirs])
    return torch.tensor(padded_mels), torch.tensor(stop_tokens, dtype=torch.float32)


def save_mel(mel_batch):
    print("Save mels...")
    for i in range(mel_batch.shape[0]):
        plt.figure()
        librosa.display.specshow(
            np.array(mel_batch[i].T), y_axis='mel', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.savefig('fig{}.png'.format(i+1), dpi=300)
        # print("{}th mel_spec saved".format(i+1), mel_batch[i].T.shape)
    print("Save ALL!")


def get_sample_batch(batch_size, vocab=None, random=False):
    idx = np.arange(batch_size)
    if random:
        idx = np.random.randint(0, 50, batch_size)
    idx = np.sort(idx)
    print("idx:", idx.shape, idx)

    # text-to-phoneme
    csv_data = pd.read_csv(hp.csv_dir, sep='|', header=None)

    texts = list(csv_data[2][idx])
    # print("texts:", texts)

    src, vocab = phoneme_batch(texts, vocab)
    # (batch, max_seq_len)
    # print("phoneme_batch.shape:", src.shape)
    # print("vocab:\n", vocab)

    # audio-to-mel
    audio_dirs = [hp.audio_dir
                  .format((4-len(str(i+1)))*'0' + str(i+1)) for i in idx]

    tgt, tgt_stops = mel_batch(audio_dirs)
    # print("mel_batch.shape:", tgt.shape)
    return src, tgt, tgt_stops, vocab


def mel_to_wav(mel):
    import librosa
    # denormalize
    M = (np.clip(mel, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    S = librosa.feature.inverse.mel_to_stft(M)
    y = librosa.griffinlim(S)

    librosa.output.write_wav("./simple_tt2.wav", y, hp.sr, norm=False)
