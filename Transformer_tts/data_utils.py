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
from g2p_en import G2p
import codecs
from os import walk


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

def seq2id(seq, symbol_to_id):
        sequence=[symbol_to_id['^']]
        sequence.extend([symbol_to_id[c] for c in text])
        sequence.append(symbol_to_id['~'])
        return sequence

def text2seq(text, type='char'):
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    clean_char = clean_text(text.rstrip())

    if type == 'char':
        seq = seq2id(clean_char, symbol_to_id)
    else:
        clean_phone = []
        for s in g2p(clean_char.lower()):
            if '@'+s in symbol_to_id:
                clean_phone.append('@'+s)
            else:
                clean_phone.append(s)
        seq = seq2id(clean_phone, symbol_to_id)

    return seq
    
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
        with open(mel_filename, 'wb') as f:
            pkl.dump(melspec, f)


def precompute_char_phone(path):

    metadata_file = os.path.join(path, 'metadata.csv')
    char_folder = os.path.join(path, 'chars')
    phone_folder = os.path.join(path, 'phones')
    if not os.path.isdir(char_folder):
        os.makedirs(char_folder)
    if not os.path.isdir(phone_folder):
        os.makedirs(phone_folder)
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    g2p = G2p()

    data = {}
    with codecs.open(metadata_file, 'r', 'utf-8') as metadata:
        for line in metadata.readlines():
            id, _, text = line.split("|")
            id = re.sub(r'"', '', id)
            clean_char = clean_text(text.rstrip())
            char_seq = seq2id(clean_char, symbol_to_id)
            clean_phone = []

            for s in g2p(clean_char.lower()):
                if '@'+s in symbol_to_id:
                    clean_phone.append('@'+s)
                else:
                    clean_phone.append(s)
            phone_seq = seq2id(clean_phone, symbol_to_id)
    
            char={'char':clean_char,
                  'char_seq':char_seq}
            char_file = os.path.join(char_folder, id+'.pkl')
            with open(char_file, 'wb') as f:
                pkl.dump(char, f)
                
            phone={'phone':clean_phone,
                  'phone_seq':phone_seq}
            phone_file = os.path.join(phone_folder, id+'.pkl')
            with open(phone_file, 'wb') as f:
                pkl.dump(phone, f)
                

if __name__ == "__main__":

    path = "../data/LJSpeech-1.1"

    # with open('./params.json') as json_file:
    #         params = json.load(json_file)['data']

    # # precompute_spectrograms(path, params)


    # filename_1 = '../data/LJSpeech-1.1/wavs/LJ002-0010.wav'
    # waveform, sample_rate = torchaudio.load(filename_1)
    # melspec = wave2mel(waveform[0].tolist(), params['sample_rate'], params['preemphasis'], params['num_freq'], params['frame_size_ms'], params['frame_hop_ms'], params['min_level_db'], params['num_mel'])

    # print(melspec)
    
    # filename_2 = '../data/LJSpeech-1.1/mels/LJ002-0010.pkl'
    # file = open(filename_2, 'rb')
    # data = pkl.load(file)
    # print(data)

    # precompute_char_phone(path)

    # for (_, _, filename) in walk('../data/LJSpeech-1.1/wavs'):
    #     print(len(filename))
        

    
