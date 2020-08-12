import os
import csv
import json
import pdb
import pickle as pkl
import re

import torch
import torchaudio
from torchaudio.datasets.utils import download_url, extract_archive, unicode_csv_reader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np 

from data_utils import precompute_spectrograms, precompute_char_phone

"""
LJSPEECH and load_ljspeech_item is exactly copy of the original code
But if we want to improve the speed, might save the melspec as picture before and use load_ljspeech_item to load it -- might do it later
We can modify the dataset and dataloader for other dataset 
"""

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
FOLDER_IN_ARCHIVE = "wavs"




def load_ljspeech_mel_item(line, wav_path, mel_path, char_path, phone_path, ext_audio, ext_mel):
    assert len(line) == 3
    fileid, transcript, normalized_transcript = line
    fileid = re.sub(r'"', '', fileid)
    fileid_audio = fileid + ext_audio
    fileid_audio = os.path.join(wav_path, fileid_audio)
    # Load audio
    waveform, sample_rate = torchaudio.load(fileid_audio)

    fileid_pkl = fileid + ext_mel
    
    fileid_mel = os.path.join(mel_path, fileid_pkl)
    with open(fileid_mel, 'rb') as pkl_in:
        melspec = pkl.load(pkl_in)
        
    file_char = os.path.join(char_path, fileid_pkl)
    with open(file_char, 'rb') as pkl_in:
        data = pkl.load(pkl_in)
        char = data['char']
        char_seq = data['char_seq']

    file_phone = os.path.join(phone_path, fileid_pkl)
    with open(file_phone, 'rb') as pkl_in:
        data = pkl.load(pkl_in)
        phone = data['phone']
        phone_seq = data['phone_seq']

        
    return (
        waveform,
        melspec,
        sample_rate,
        transcript,
        normalized_transcript,
        char,
        char_seq,
        phone,
        phone_seq)

def data_mel_processing(data, params):
    type = params['seq_type']
    
    mels = []
    mel_len = []
    seqs = []
    seq_len = []
    gates = []
    waveforms = []
    
    for (waveform, melspec, _, _, text, char, char_seq, phone, phone_seq) in data:
        mel_len.append(melspec.shape[1])
        mel = torch.tensor(melspec).squeeze(0).transpose(0,1)
        mels.append(mel)
        # print(char)
        # print(phone)
        gate = torch.zeros(melspec.shape[1])
        gate[-1] = 1
        gates.append(gate)
        if type == 'char':
            label = char_seq
        else:
            label = phone_seq
        # print(label)
        seq_len.append(len(label))
        seqs.append(torch.LongTensor(label))
    mels = pad_sequence(mels, batch_first=True)
    seqs = pad_sequence(seqs, batch_first=True)
    gates = pad_sequence(gates, batch_first=True)

    return mels, seqs, gates, torch.tensor(mel_len), torch.tensor(seq_len)

class LJSPEECH_MEL(Dataset):
    """
    Create a Dataset for LJSpeech-1.1. Each item is a tuple of the form:
    waveform, log mel spectrogram, sample_rate, transcript, normalized_transcript
    """

    _ext_audio = ".wav"
    _ext_archive = '.tar.bz2'
    _ext_mel = ".pkl"
    

    def __init__(
            self, root, params, url=URL, download=False):

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(self._ext_archive)[0]
        base_folder = os.path.join(root, basename)
        
        self._wav_path = os.path.join(base_folder, 'wavs')
        self._mel_path = os.path.join(base_folder, 'mels')
        self._char_path = os.path.join(base_folder, 'chars')
        self._phone_path = os.path.join(base_folder, 'phones')
        self._metadata_path = os.path.join(base_folder, 'metadata.csv')

        if download:
            if not os.path.isdir(self._wav_path):
                if not os.path.isfile(archive):
                    download_url(url, root)
                extract_archive(archive)

        if not os.path.isdir(self._mel_path):
            precompute_spectrograms(base_folder, params)

        if not os.path.isdir(self._char_path) or not os.path.isdir(self._phone_path):
            precompute_char_phone(base_folder)
            
        with open(self._metadata_path, "r") as metadata:
            walker = unicode_csv_reader(metadata, delimiter="|", quoting=csv.QUOTE_NONE)
            self._walker = list(walker)

    def __getitem__(self, n):
        line = self._walker[n]
        return load_ljspeech_mel_item(line, self._wav_path, self._mel_path, self._char_path, self._phone_path, self._ext_audio, self._ext_mel)

    def __len__(self):
        return len(self._walker)
