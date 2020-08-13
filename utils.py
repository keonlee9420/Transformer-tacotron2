#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
                     separator=separator.Separator(word=' <wb> ', syllable=' <sylb> ', phone=' '),
                     preserve_punctuation=True)
