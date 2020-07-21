import os
import torch
import torchaudio
import json
import pdb
from torch.utils import data
import torch.nn as nn

from dataloader_utils import LJSPEECH_MEL, data_processing
from models import Model, TSSLoss
from utilities import train, test, save_model, inference

import argparse

parser = argparse.ArgumentParser(description='The arguments input to the training.')
parser.add_argument('-m', action="store", dest='model_path', default="./models/saved_model")

def main(model_path):

    torch.manual_seed(0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    if not os.path.isfile(os.path.join(model_path, 'params.json')):
        with open('./params.json') as json_file:
            params = json.load(json_file)
    else:
        with open(os.path.join(model_path, 'params.json')) as json_file:
            params = json.load(json_file)

    data_params = params['data']
    train_params = params['train']

    dataset = LJSPEECH_MEL('./data',
                           url='https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
                           folder_in_archive='wavs', download=True)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [12000, 1100])

    rand_sampler = torch.utils.data.RandomSampler(test_dataset, num_samples=1, replacement=True)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=train_params['batch_size'],
                                  shuffle=False,
                                  sampler=rand_sampler,
                                  collate_fn=lambda x: data_processing(
                                      x, params['data']),
                                  **kwargs)
    
    model = Model(params, device).to(device)
    criterion = TSSLoss(params).to(device)
    if os.path.isfile(os.path.join(model_path, 'model.pt')):
        model_file = os.path.join(model_path, 'model.pt')
        model.load_state_dict(torch.load(model_file))
    else:
        raise ValueError('The model file does not exist.')
    

    inference(model, device, test_loader, criterion, data_params)


if __name__ == "__main__":
    model = parser.parse_args()
    main(model_path = model.model_path)
