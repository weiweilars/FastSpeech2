import os
import pdb
import json
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import sounddevice as sd
import soundfile as sf

from data_utils import text2seq, mel2wave

def adjust_learning_rate(optimizer, step_num, init_lr, warmup_step=4000):
    lr = init_lr* min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model, device, train_loader, criterion, optimizer, epoch, params):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        mel, seq, mel_pos, seq_pos = _data 
        input = mel.to(device), seq.to(device), mel_pos.to(device), seq_pos.to(device)
        output = model(input[0], input[1], input[2], input[3])
        
        loss,mel_pre_loss,mel_post_loss,gate_loss = criterion(output, input)
        loss.backward()

        adjust_learning_rate(optimizer, epoch, params['lr'], warmup_step=params['warmup_step'])
        nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip_thresh'])
        optimizer.step()
        optimizer.zero_grad() 
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tMel_pre Loss: {:.6f}\tMel_post Loss: {:.6f}\tGate Loss: {:.6f}'.format(
                epoch, batch_idx * len(mel), data_len,
                100. * batch_idx / len(train_loader), loss.item(), mel_pre_loss.item(), mel_post_loss.item(), gate_loss.item()))
            for param_group in optimizer.param_groups:
                print(param_group['lr'])

def test(model, device, test_loader, criterion):
    print('\nevaluating…')
    model.eval()
    test_loss = 0
    test_mel_pre_loss = 0
    test_mel_post_loss = 0
    test_gate_loss = 0
    with torch.no_grad():
        for I, _data in enumerate(test_loader):
            mel, seq, mel_pos, seq_pos = _data 
            input = mel.to(device), seq.to(device), mel_pos.to(device), seq_pos.to(device)

            # mel_input = torch.zeros([1,1,80]).to(device)
            # for i in range(mel.shape[1]):
            #     pos_mel = torch.arange(1, mel_input.size(1)+1).unsqueeze(0).to(device)
            #     mel_pred, postnet_pred, stop_token, _ = model(mel_input, seq.to(device), pos_mel, seq_pos.to(device))
            #     mel_input = torch.cat([mel_input, postnet_pred[:,-1:]], dim=1)

            output = model(input[0], input[1], input[2], input[3])
            # output = mel_input[:,1:,:], mel_input[:,1:,:], stop_token

            loss, mel_pre_loss, mel_post_loss, gate_loss = criterion(output, input)
            
            test_loss += loss.item() / len(test_loader)
            test_mel_pre_loss += mel_pre_loss.item() / len(test_loader)
            test_mel_post_loss += mel_post_loss.item() / len(test_loader)
            test_gate_loss += gate_loss.item() / len(test_loader)
            

    print('Test set: Average total loss: {:.4f}'.format(test_loss))
    print('Test set: Average mel_pre loss: {:.4f}'.format(test_mel_pre_loss))
    print('Test set: Average mel_post loss: {:.4f}'.format(test_mel_post_loss))
    print('Test set: Average gate loss: {:.4f}'.format(test_gate_loss))
    return test_loss

def inference(model, device, test_loader, criterion, params):
    print('\ninference…')
    model.eval()
    with torch.no_grad():
        for I, _data in enumerate(test_loader):
            mel, seq, mel_pos, seq_pos = _data 
            input = mel.to(device), seq.to(device), mel_pos.to(device), seq_pos.to(device)

            mel_input = torch.zeros([1,1,80]).to(device)

            for i in range(mel.shape[1]):
                pos_mel = torch.arange(1, mel_input.size(1)+1).unsqueeze(0).to(device)
                mel_pred, postnet_pred, stop_token,mask = model(mel_input, seq.to(device), pos_mel, seq_pos.to(device))
                mel_input = torch.cat([mel_input, postnet_pred[:,-1:]], dim=1)

            
            output = mel_input[:,1:,:], mel_input[:,1:,:], stop_token, mask
            print(mel_input)
            loss, mel_pre_loss, mel_post_loss, gate_loss = criterion(output, input)

            print(loss)
            print(mel_pre_loss)
            print(mel_post_loss)
            print(gate_loss)


            output = model(input[0], input[1], input[2], input[3])

            print(output[0])
            loss, mel_pre_loss, mel_post_loss, gate_loss = criterion(output, input)

            pdb.set_trace()
            print(loss)
            print(mel_pre_loss)
            print(mel_post_loss)
            print(gate_loss)

            
            # input_test = mel.squeeze(0)
            # input_test = input_test.transpose(0,1).cpu().numpy()

            # wav = mel2wave(input_test, params['sample_rate'], params['preemphasis'], params['num_freq'], params['frame_size_ms'], params['frame_hop_ms'], params['min_level_db'], params['num_mel'], params['power'], params['gl_iter'])

            # sd.play(wav, 22050)
            # status = sd.wait()


            # output_test = output[0].squeeze(0)
            # output_test = output_test.transpose(0,1).cpu().numpy()

            # wav = mel2wave(output_test, params['sample_rate'], params['preemphasis'], params['num_freq'], params['frame_size_ms'], params['frame_hop_ms'], params['min_level_db'], params['num_mel'], params['power'], params['gl_iter'])

            # sd.play(wav, 22050)
            # status = sd.wait()
            
            # print("here is output")
            # output = mel_input.squeeze(0)
            # output = output.transpose(0,1).cpu().numpy()

            # wav = mel2wave(output, params['sample_rate'], params['preemphasis'], params['num_freq'], params['frame_size_ms'], params['frame_hop_ms'], params['min_level_db'], params['num_mel'], params['power'], params['gl_iter'])

            # sd.play(wav, 22050)
            # status = sd.wait()

def inference_single(model, device, params):
    model.eval()
    text = "This is for testing."
    text = np.asarray(text2seq(text))
    text = torch.LongTensor(text).unsqueeze(0).to(device)
    mel_input = torch.zeros([1,1,80]).to(device)
    pos_text = torch.arange(1, text.size(1)+1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for i in range(800):
            pos_mel = torch.arange(1, mel_input.size(1)+1).unsqueeze(0).to(device)
            mel_pred, postnet_pred, stop_token = model(mel_input, text, pos_mel, pos_text)
            mel_input = torch.cat([mel_input, postnet_pred[:,-1:]], dim=1)
    mel_input = mel_input[:,1:,:].squeeze(0)
    mel_input = mel_input.transpose(0,1).cpu().numpy()    
    wav = mel2wave(mel_input, params['sample_rate'], params['preemphasis'], params['num_freq'], params['frame_size_ms'], params['frame_hop_ms'], params['min_level_db'], params['num_mel'], params['power'], params['gl_iter'])

    sd.play(wav, 22050)
    status = sd.wait()
    
    

def save_model(model, params_dict, model_path):
    model_file = os.path.join(model_path, 'model.pt')
    torch.save(model.state_dict(), model_file)
    params_file = os.path.join(model_path, 'params.json')
    with open(params_file, 'w') as f:
        json.dump(params_dict, f)
