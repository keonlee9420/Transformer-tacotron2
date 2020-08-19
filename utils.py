#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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


def _normalize(mel):
    return np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)


def _denormalize(mel):
    return (np.clip(mel, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db


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
    mel = _normalize(mel)

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


def _save_mel_fig(mel, save_dir):  # (mel_chennels, n_frames)
    plt.figure()
    librosa.display.specshow(
        np.array(mel), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(save_dir, dpi=300)
    plt.close()
    print("mel_fig saved!")


def save_mel(mel_batch, normalized=False):
    print("Save mels...")
    for i in range(mel_batch.shape[0]):
        # (mel_chennels, n_frames)
        mel = mel_batch[i].T
        if normalized:
            # denormalize
            mel = _denormalize(mel)
        _save_mel_fig(mel, 'fig{}.png'.format(i+1))
        print("{}th mel_spec saved".format(i+1), mel.shape)
    print("Save ALL!")


def get_sample_batch(batch_size, start=0, vocab=None, random=False):
    idx = np.arange(start, start + batch_size)
    if random:
        idx = np.random.randint(0, 100, batch_size)
    idx = np.sort(idx)
    print("get_sample_batch idx.shape, idx:", idx.shape, idx)

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
    return {'src': src, 'tgt': tgt, 'tgt_stops': tgt_stops, 'vocab': vocab}


def save_wav(wav, filename):
    from scipy.io.wavfile import write
    path = os.path.join(hp.output_dir, '{}.wav'.format(filename))
    # write(path, hp.sr, wav)
    wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    write(path, hp.sr, wav.astype(np.int16))
    print("wav saved!")


def _save_amp_time_fig(y, save_dir):
    plt.plot(y)
    plt.title('Signal')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.savefig(save_dir, dpi=300)
    plt.close()
    print("amp_time_fig saved!")


def mel_to_wav(decoder_output, filename=None):
    mel = decoder_output.squeeze(0).to('cpu')  # (mel_channels, n_frames)

    # denormalize
    M = _denormalize(mel)

    # to power
    power = librosa.db_to_power(M, ref=1.)

    # S = librosa.feature.inverse.mel_to_stft(
    #     power.numpy(), sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
    # y = librosa.griffinlim(S)
    import time
    bt = time.time()
    print("stpe_1 {:.2f}".format(time.time() - bt))
    y = librosa.feature.inverse.mel_to_audio(
        power.numpy(), sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
    print("stpe_2 {:.2f}".format(time.time() - bt))

    # de-preemphasis
    from scipy import signal
    wav = signal.lfilter([1], [1, -hp.preemphasis], y)
    print("stpe_3 {:.2f}".format(time.time() - bt))

    # Trimmings
    wav, _ = librosa.effects.trim(wav)
    print("stpe_4 {:.2f}".format(time.time() - bt))

    # save results
    if filename:
        _save_amp_time_fig(wav, save_dir=os.path.join(
            hp.output_dir, '{}.png'.format(str(filename) + "_amp_time")))
        print("stpe_5 {:.2f}".format(time.time() - bt))
        _save_mel_fig(M, save_dir=os.path.join(
            hp.output_dir, '{}.png'.format(str(filename) + "_mel")))
        print("stpe_6 {:.2f}".format(time.time() - bt))

    return wav
