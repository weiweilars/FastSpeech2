import os
import csv
import json
import pdb
import pickle as pkl

import torch
import torchaudio
from torchaudio.datasets.utils import download_url, extract_archive, unicode_csv_reader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np 

from data_utils import wave2mel, text2seq, precompute_spectrograms

"""
LJSPEECH and load_ljspeech_item is exactly copy of the original code
But if we want to improve the speed, might save the melspec as picture before and use load_ljspeech_item to load it -- might do it later
We can modify the dataset and dataloader for other dataset 
"""

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
FOLDER_IN_ARCHIVE = "wavs"

# with open('./params.json') as json_file:
#     params = json.load(json_file)['data']


def load_ljspeech_item(line, path, ext_audio):
    assert len(line) == 3
    fileid, transcript, normalized_transcript = line
    fileid_audio = fileid + ext_audio
    fileid_audio = os.path.join(path, fileid_audio)
    # Load audio
    waveform, sample_rate = torchaudio.load(fileid_audio)

    return (
        waveform,
        sample_rate,
        transcript,
        normalized_transcript,
    )

def data_processing(data, params):
    mels = []
    seqs = []
    mel_len = []
    seq_len = []
    gates = []

    for (waveform, _, _,text) in data:
        melspec =  wave2mel(waveform[0].tolist(), params['sample_rate'], params['preemphasis'], params['num_freq'], params['frame_size_ms'], params['frame_hop_ms'], params['min_level_db'], params['num_mel'])
        mel_len.append(melspec.shape[1])
        mel = torch.tensor(melspec).squeeze(0).transpose(0,1)
        mels.append(mel)
        gate = torch.ones(melspec.shape[1])
        gate[-1] = 0
        gates.append(gate)
        label = text2seq(text)
        seq_len.append(len(label))
        seqs.append(torch.LongTensor(label))
    mels = pad_sequence(mels, batch_first=True)
    seqs = pad_sequence(seqs, batch_first=True)
    gates = pad_sequence(gates, batch_first=True)
    
    return mels, seqs, gates, torch.tensor(mel_len), torch.tensor(seq_len)

class LJSPEECH_WAV(Dataset):
    """
    Create a Dataset for LJSpeech-1.1. Each item is a tuple of the form:
    waveform, log mel spectrogram, sample_rate, transcript, normalized_transcript
    """

    _ext_audio = ".wav"
    _ext_archive = '.tar.bz2'

    def __init__(
            self, root, url=URL, folder_in_archive=FOLDER_IN_ARCHIVE, download=False
    ):

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(self._ext_archive)[0]
        folder_in_archive = os.path.join(basename, folder_in_archive)

        self._path = os.path.join(root, folder_in_archive)
        self._metadata_path = os.path.join(root, basename, 'metadata.csv')

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    download_url(url, root)
                extract_archive(archive)

        with open(self._metadata_path, "r") as metadata:
            walker = unicode_csv_reader(metadata, delimiter="|", quoting=csv.QUOTE_NONE)
            self._walker = list(walker)

    def __getitem__(self, n):
        line = self._walker[n]
        return load_ljspeech_item(line, self._path, self._ext_audio)

    def __len__(self):
        return len(self._walker)


def load_ljspeech_mel_item(line, wav_path, mel_path, ext_audio, ext_mel):
    assert len(line) == 3
    fileid, transcript, normalized_transcript = line
    fileid = re.sub(r'"', '', fileid)
    fileid_audio = fileid + ext_audio
    fileid_audio = os.path.join(wav_path, fileid_audio)
    # Load audio
    waveform, sample_rate = torchaudio.load(fileid_audio)

    fileid_mel = fileid + ext_mel
    fileid_mel = os.path.join(mel_path, fileid_mel)

    with open(fileid_mel, 'rb') as pkl_in:
        melspec = pkl.load(pkl_in)

    return (
        waveform,
        melspec,
        sample_rate,
        transcript,
        normalized_transcript,
    )

def data_mel_processing(data, params):
    mels = []
    seqs = []
    mel_len = []
    seq_len = []
    gates = []

    for (waveform, melspec, _, _,text) in data:
        mel_len.append(melspec.shape[1])
        mel = torch.tensor(melspec).squeeze(0).transpose(0,1)
        mels.append(mel)
        gate = torch.ones(melspec.shape[1])
        gate[-1] = 0
        gates.append(gate)
        label = text2seq(text)
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
            self, root, params, url=URL, wav_folder='wavs', mel_folder="mels", char_folder="chars", phone_folder="phones", download=False
    ):

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(self._ext_archive)[0]
        base_folder = os.path.join(root, basename)
        
        self._wav_path = os.path.join(base_folder, wav_folder)
        self._mel_path = os.path.join(base_folder, mel_folder)
        self._char_path = os.path.join(base_folder, char_folder)
        self._phone_path = os.path.join(base_folder, phone_folder)
        self._metadata_path = os.path.join(base_folder, 'metadata.csv')

        if download:
            if not os.path.isdir(self._wav_path):
                if not os.path.isfile(archive):
                    download_url(url, root)
                extract_archive(archive)

        if not os.path.isdir(self._mel_path):
            os.makdirs(self._mel_path)
            precompute_spectrograms(self._mel_path, params)

        if not os.path.isdir(self._char_path) or not os.path.isdir(self._phone_path):
            os.makdirs(self._char_path)
            os.makdirs(self._phone_path)
            precompute_spectrograms(self._mel_p, params)
            
        with open(self._metadata_path, "r") as metadata:
            walker = unicode_csv_reader(metadata, delimiter="|", quoting=csv.QUOTE_NONE)
            self._walker = list(walker)

    def __getitem__(self, n):
        line = self._walker[n]
        return load_ljspeech_mel_item(line, self._wav_path, self._mel_path, self._ext_audio, self._ext_mel)

    def __len__(self):
        return len(self._walker)
