import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import pickle as pkl
import torchaudio
import scipy
import librosa
import numpy as np
from text_utils import symbols, clean_text
import re


### for debugging
import pdb
import json
import os
from torch.utils import data
import sounddevice as sd
import soundfile as sf

"""
pytorchaudio has several similar functions, we might switch to them if we want in the future. 
stft, istft, angle .... 
"""

def wave2mel(waveform, sample_rate, preemphasis, num_freq, frame_size_ms, frame_hop_ms, min_level_db, num_mel):
    n_fft = (num_freq - 1)*2
    win_length = int(frame_size_ms / 1000 * sample_rate)
    hop_length = int(frame_hop_ms / 1000 * sample_rate)
    

    # preemphasis
    x = scipy.signal.lfilter([1, -preemphasis], [1], waveform)

    # spectrograms 
    x = librosa.stft(y=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # mel spectrograms
    mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels=num_mel)
    x = np.dot(mel_basis, np.abs(x))

    # log mel spectrograms 
    x = librosa.core.amplitude_to_db(x)
    
    # normalize log mel spectrograms 
    x = np.clip((x - min_level_db) / -min_level_db, 0, 1)

    return x

def mel2wave(spec, sample_rate, preemphasis, num_freq, frame_size_ms, frame_hop_ms, min_level_db, num_mel, power, gl_iter):
    n_fft = (num_freq - 1)*2
    win_length = int(frame_size_ms / 1000 * sample_rate)
    hop_length = int(frame_hop_ms / 1000 * sample_rate)


    def _griffin_lim(S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = librosa.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
        for i in range(gl_iter):
            angles = np.exp(1j * np.angle(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)))
            y = librosa.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
        return y

    # de normalize = log mel spectrograms
    x = (np.clip(spec, 0, 1) * -min_level_db) + min_level_db
    
    # de log = mel spectrograms
    x = librosa.core.db_to_amplitude(x)

    # de mel = spectrograms
    mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels=num_mel)
    inv_mel_basis = np.linalg.pinv(mel_basis)
    x = np.maximum(1e-5, np.dot(inv_mel_basis, x))

    # de spectrograms
    x = _griffin_lim(x ** power)

    # de preemphasis
    x = scipy.signal.lfilter([1], [1, preemphasis], x)
    return x


def text2seq(text):
    sequence = []
    symbols_to_id = {s: i for i, s in enumerate(symbols)}
    curly = re.compile(r'(.*?)\{(.+?)\}(.*)')
    while len(text):
        m = curly.match(text)
        if not m:
            sequence += [symbols_to_id[s] for s in clean_text(text) if s is not '_' and s is not '~']
            break
        sequence += [symbols_to_id[s] for s in clean_text(m.group(1)) if s is not '_' and s is not '~']
        group2 = ['@' + s for s in m.group(2).split()]
        sequence += [symbols_to_id[s] for s in group2 if s is not '_' and s is not '~']
        text = m.group(3)    
    return sequence

def seq2text(seq):
    id_to_symbols = {i: s for i, s in enumerate(symbols)}
    result = ''
    for id in sequence:
        if id in id_to_symbols:
            s = id_to_symbols[id]
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s

    return result.replace('}{', ' ')


def precompute_spectrograms(path, params):
    wav_path = os.path.join(path,'wavs')
    mel_path = os.path.join(path,'mels')
    if not os.path.isdir(mel_path):
        os.makedirs(mel_path)
    files = Path(wav_path).glob('*.wav')
    for filename in files:
        waveform, sample_rate = torchaudio.load(filename)
        melspec = wave2mel(waveform[0].tolist(), params['sample_rate'], params['preemphasis'], params['num_freq'], params['frame_size_ms'], params['frame_hop_ms'], params['min_level_db'], params['num_mel'])
        mel_filename = os.path.join(mel_path,os.path.basename(filename).split('.')[0]) + '.pkl'
        with open(mel_filename, 'B') as f:
            pkl.dump(melspec, f)
        

if __name__ == "__main__":

    # path = "../data/LJSpeech-1.1"

    with open('./params.json') as json_file:
            params = json.load(json_file)['data']

    # precompute_spectrograms(path, params)


    filename_1 = '../data/LJSpeech-1.1/wavs/LJ002-0010.wav'
    waveform, sample_rate = torchaudio.load(filename_1)
    melspec = wave2mel(waveform[0].tolist(), params['sample_rate'], params['preemphasis'], params['num_freq'], params['frame_size_ms'], params['frame_hop_ms'], params['min_level_db'], params['num_mel'])

    print(melspec)
    
    filename_2 = '../data/LJSpeech-1.1/mels/LJ002-0010.pkl'
    file = open(filename_2, 'rb')
    data = pkl.load(file)
    print(data)
