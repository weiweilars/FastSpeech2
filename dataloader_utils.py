import os
import csv
import json
import pdb

import torch
import torchaudio
from torchaudio.datasets.utils import download_url, extract_archive, unicode_csv_reader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np 

from data_utils import wave2mel, text2seq

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
    mel_pos = []
    seq_pos = []

    for ( waveform, _, _,text) in data:
        melspec =  wave2mel(waveform[0].tolist(), params['sample_rate'], params['preemphasis'], params['num_freq'], params['frame_size_ms'], params['frame_hop_ms'], params['min_level_db'], params['num_mel'])
        mel = torch.tensor(melspec).squeeze(0).transpose(0,1)
        mel =torch.cat([torch.zeros([params['num_output_per_step'], params['num_mel']]),mel], dim=0)
        mels.append(mel)
        mel_pos.append(torch.LongTensor(np.arange(1, mel.shape[0]+1)))
        label = text2seq(text)
        seqs.append(torch.LongTensor(label))
        seq_pos.append(torch.LongTensor(np.arange(1, len(label)+1)))
    mels = pad_sequence(mels, batch_first=True)
    seqs = pad_sequence(seqs, batch_first=True)
    mel_pos = pad_sequence(mel_pos, batch_first=True)
    seq_pos = pad_sequence(seq_pos, batch_first=True)
    
    return mels, seqs, mel_pos, seq_pos

class LJSPEECH_MEL(Dataset):
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
