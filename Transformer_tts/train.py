import os
import torch
import torchaudio
import json
import pdb
import time

from torch.utils import data
import torch.nn as nn
import matplotlib.pyplot as plt

from dataloader_utils import LJSPEECH_MEL, data_mel_processing
from models import Model, TTSLoss
from utilities import train, validate
from save_util import save_model, get_writer

from text_utils import symbols


import argparse
parser = argparse.ArgumentParser(description='The arguments input to the training.')
parser.add_argument('-p', action="store", dest='model_path', default="./checkpoint/saved_model")
parser.add_argument('-l', action="store", dest='log_path', default="./log")
parser.add_argument('-n', action="store", dest='model_name', default="model.pt")

def main(model_path, model_name, log_path):

    torch.manual_seed(0)

    

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_gpu = torch.cuda.device_count() 
    print("There are {} gups.".format(num_gpu))
    
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
                           download=True)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [12000, 1100])

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=train_params['batch_size'],
                                   shuffle=True,
                                   collate_fn=lambda x: data_mel_processing(
                                       x, params['data']),
                                   **kwargs)

    rand_sampler = torch.utils.data.RandomSampler(test_dataset, num_samples=100, replacement=True)
    val_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  sampler=rand_sampler,
                                  collate_fn=lambda x: data_mel_processing(
                                      x, params['data']),
                                  **kwargs)

    model = Model(params, device).to(device)

    # if num_gpu > 1:
    #     model = nn.DataParallel(model)
        
    criteriate = TTSLoss(device).to(device)
    # print(model)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print("parameter {} is {}".format(name, param.shape))
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))
    epochs = train_params['epochs']
    writer = get_writer(model_path, log_path)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = train_params['lr'],
        			 betas = (0.9, 0.98),
                                 eps = 1e-09)
    
    iteration = 0
    model_file = os.path.join(model_path, 'model.pt')
    if os.path.isfile(model_file):
        file = torch.load(model_file)
        iteration = file['iteration']
        optimizer.load_state_dict(file['optimizer'])
        model.load_state_dict(file['model_dict'])
        best_valid_loss= validate(model, criteriate, device, val_loader, iteration, writer, train_params)
    else:
        best_valid_loss = float("inf")
        
    for epoch in range(1, epochs + 1):
        iteration = train(model, criteriate, device, train_loader, optimizer, iteration, train_params, writer)
        test_loss = validate(model, criteriate, device, val_loader, iteration, writer, train_params)
        if test_loss < best_valid_loss:
            print("The validation loss is improved by {}, new model is saving".format(best_valid_loss-test_loss))
            best_valid_loss = test_loss
            save_model(model, optimizer, iteration, params, model_path, model_name)

        if epoch%10 == 0:
            temp_model_path = os.path.join(model_path, str(iteration))
            os.makedirs(temp_model_path)
            temp_model_name = 'model'+str(iteration)+'.pt'
            save_model(model, optimizer, iteration, params, temp_model_path, temp_model_name)
            


if __name__ == "__main__":
    model_info = parser.parse_args()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    main(model_path = model_info.model_path, model_name=model_info.model_name, log_path=model_info.log_path)

    
