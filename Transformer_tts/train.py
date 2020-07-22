import os
import torch
import torchaudio
import json
import pdb
import time

from torch.utils import data
import torch.nn as nn

from dataloader_utils import LJSPEECH_MEL, data_mel_processing
from models import Model, TSSLoss, DecoderPrenet, EncoderPrenet
from utilities import train, test, save_model

import argparse

parser = argparse.ArgumentParser(description='The arguments input to the training.')
parser.add_argument('-m', action="store", dest='model_path', default="./models/saved_model")


def main(model_path):

    torch.manual_seed(0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if not os.path.isdir("../data"):
        os.makedirs("../data")

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

    dataset = LJSPEECH_MEL('../data',
                           params['data'],
                           url='https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
                           wav_folder='wavs',
                           mel_folder='mels',
                           download=True)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [12000, 1100])

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=train_params['batch_size'],
                                   shuffle=True,
                                   collate_fn=lambda x: data_mel_processing(
                                       x, params['data']),
                                   **kwargs)

    rand_sampler = torch.utils.data.RandomSampler(test_dataset, num_samples=100, replacement=True)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  sampler=rand_sampler,
                                  collate_fn=lambda x: data_processing(
                                      x, params['data']),
                                  **kwargs)

    mel, seq, gate, mel_len, seq_len = next(iter(train_loader))
    # model = Model(params).to(device)

    # model(mel.to(device), seq.to(device), mel_len.to(device), seq_len.to(device))

    # print(model)
    # print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr = train_params['lr'],
    #     			 betas = (0.9, 0.98),
    #                              eps = 1e-09)

    # criterion = TSSLoss(params).to(device)

    # epochs = train_params['epochs']

    # model_file = os.path.join(model_path, 'model.pt')
    # if os.path.isfile(os.path.join(model_path, 'model.pt')):       
    #     model.load_state_dict(torch.load(model_file))
    #     best_valid_loss= test(model, device, test_loader, criterion)
    # else:
    #     best_valid_loss = float("inf")
    # print(best_valid_loss)

    
    # for epoch in range(1, epochs + 1):
    #     train(model, device, train_loader, criterion, optimizer, epoch, train_params)
    #     if epoch == 1 and not os.path.isfile(model_file):
    #         save_model(model, params, model_path)
    #     test_loss = test(model, device, test_loader, criterion)
    #     if test_loss < best_valid_loss:
    #         print("The validation loss is improved by {}, new model is saving".format(best_valid_loss-test_loss))
    #         best_valid_loss = test_loss
    #         save_model(model, params, model_path)


if __name__ == "__main__":
    model = parser.parse_args()
    main(model_path = model.model_path)
