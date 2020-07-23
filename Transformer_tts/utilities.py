import os
import pdb
import json
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import sounddevice as sd
import soundfile as sf

# from data_utils import text2seq, mel2wave

def adjust_learning_rate(optimizer, step_num, init_lr, warmup_step=4000):
    lr = init_lr* min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model, device, train_loader, optimizer, iteration, params, writer):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        mel, seq, gate, mel_pos, seq_pos = _data 
        mel_linear_loss,mel_post_loss,gate_loss, guide_loss = model(mel.to(device),
                                                                    seq.to(device),
                                                                    mel_pos.to(device),
                                                                    seq_pos.to(device),
                                                                    gate.to(device))
        total_loss = (mel_linear_loss+mel_post_loss+gate_loss+guide_loss)/params['accumulation']
        total_loss.backward()
        #loss = loss+total_loss.item()
        iteration +=1 
        if iteration%params['accumulation'] == 0:
            
            adjust_learning_rate(optimizer, iteration, params['lr'], warmup_step=params['warmup_step'])
            nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip_thresh'])
            optimizer.step()
            optimizer.zero_grad()
            writer.add_losses(mel_linear_loss.item(),
                              mel_post_loss.item(),
                              gate_loss.item(),
                              guide_loss.item(),
                              iteration//params['accumulation'], 'Train')
            
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Iteration: {} [{}/{} ({:.0f}%)]\tMel_linear Loss: {:.6f}\t Mel_post Loss: {:.6f}\t Gate Loss: {:.6f}\tGuide Loss: {:.6f}'.format(
                iteration, batch_idx * len(mel), data_len,
                100. * batch_idx / len(train_loader), total_loss.item(), mel_linear_loss.item(), mel_post_loss.item(), gate_loss.item()))
            for param_group in optimizer.param_groups:
                print(param_group['lr'])

    return iteration

def validate(model, device, val_loader, iteration, writer, params):
    print('\nevaluating…')
    model.eval()
    with torch.no_grad():
        mel_linear_avg, mel_post_avg, gate_avg, guide_avg = 0, 0, 0, 0
        for i, batch in enumerate(val_loader):
            mel, seq, gate, mel_len, seq_len = batch
            mel_linear, mel_post, gate_out, seq_align, mel_align, mel_seq_align, _ = model.output(mel.to(device),
                                                                                                  seq.to(device),
                                                                                                  mel_len.to(device),
                                                                                                  seq_len.to(device))

            mel_linear_loss,mel_post_loss,gate_loss, guide_loss = model(mel.to(device),
                                                                    seq.to(device),
                                                                    mel_len.to(device),
                                                                    seq_len.to(device),
                                                                    gate.to(device))
            
            mel_linear_avg += mel_linear_loss.item()
            mel_post_avg += mel_post_loss.item()
            gate_avg += gate_loss.item()
            guide_avg += guide_loss.item()

        print('Test set: Average mel linear loss: {:.4f}'.format(mel_linear_avg/(i+1)))
        print('Test set: Average mel post loss: {:.4f}'.format(mel_post_avg/(i+1)))
        print('Test set: Average gate loss: {:.4f}'.format(gate_avg/(i+1)))
        print('Test set: Average guide loss: {:.4f}'.format(guide_avg/(i+1)))

        total_loss = (mel_linear_avg + mel_post_avg + gate_avg + guide_avg)/(i+1)

    
        
    writer.add_losses(mel_linear_loss.item(),
                              mel_post_loss.item(),
                              gate_loss.item(),
                              guide_loss.item(),
                              iteration//params['accumulation'], 'Validation')
    
    writer.add_specs(mel.detach().cpu(),
                     mel_linear.detach().cpu(),
                     mel_post.detach().cpu(),
                     mel_len.detach().cpu(),
                     iteration//params['accumulation'], 'Validation')
    
    writer.add_alignments(mel_align.detach().cpu(),
                          seq_align.detach().cpu(),
                          mel_seq_align.detach().cpu(),
                          mel_len.detach().cpu(),
                          seq_len.detach().cpu(),
                          iteration//params['accumulation'], 'Validation')
    
    writer.add_gates(gate_out.detach().cpu(),
                    iteration//params['accumulation'], 'Validation')

    return total_loss


# def inference(model, device, test_loader, criterion, params):
#     print('\ninference…')
#     model.eval()
#     with torch.no_grad():
#         for I, _data in enumerate(test_loader):
#             mel, seq, mel_pos, seq_pos = _data 
#             input = mel.to(device), seq.to(device), mel_pos.to(device), seq_pos.to(device)

#             mel_input = torch.zeros([1,1,80]).to(device)

#             for i in range(mel.shape[1]):
#                 pos_mel = torch.arange(1, mel_input.size(1)+1).unsqueeze(0).to(device)
#                 mel_pred, postnet_pred, stop_token,mask = model(mel_input, seq.to(device), pos_mel, seq_pos.to(device))
#                 mel_input = torch.cat([mel_input, postnet_pred[:,-1:]], dim=1)

            
#             output = mel_input[:,1:,:], mel_input[:,1:,:], stop_token, mask
#             print(mel_input)
#             loss, mel_pre_loss, mel_post_loss, gate_loss = criterion(output, input)

#             print(loss)
#             print(mel_pre_loss)
#             print(mel_post_loss)
#             print(gate_loss)


#             output = model(input[0], input[1], input[2], input[3])

#             print(output[0])
#             loss, mel_pre_loss, mel_post_loss, gate_loss = criterion(output, input)

#             pdb.set_trace()
#             print(loss)
#             print(mel_pre_loss)
#             print(mel_post_loss)
#             print(gate_loss)

            
#             # input_test = mel.squeeze(0)
#             # input_test = input_test.transpose(0,1).cpu().numpy()

#             # wav = mel2wave(input_test, params['sample_rate'], params['preemphasis'], params['num_freq'], params['frame_size_ms'], params['frame_hop_ms'], params['min_level_db'], params['num_mel'], params['power'], params['gl_iter'])

#             # sd.play(wav, 22050)
#             # status = sd.wait()


#             # output_test = output[0].squeeze(0)
#             # output_test = output_test.transpose(0,1).cpu().numpy()

#             # wav = mel2wave(output_test, params['sample_rate'], params['preemphasis'], params['num_freq'], params['frame_size_ms'], params['frame_hop_ms'], params['min_level_db'], params['num_mel'], params['power'], params['gl_iter'])

#             # sd.play(wav, 22050)
#             # status = sd.wait()
            
#             # print("here is output")
#             # output = mel_input.squeeze(0)
#             # output = output.transpose(0,1).cpu().numpy()

#             # wav = mel2wave(output, params['sample_rate'], params['preemphasis'], params['num_freq'], params['frame_size_ms'], params['frame_hop_ms'], params['min_level_db'], params['num_mel'], params['power'], params['gl_iter'])

#             # sd.play(wav, 22050)
#             # status = sd.wait()

# def inference_single(model, device, params):
#     model.eval()
#     text = "This is for testing."
#     text = np.asarray(text2seq(text))
#     text = torch.LongTensor(text).unsqueeze(0).to(device)
#     mel_input = torch.zeros([1,1,80]).to(device)
#     pos_text = torch.arange(1, text.size(1)+1).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         for i in range(800):
#             pos_mel = torch.arange(1, mel_input.size(1)+1).unsqueeze(0).to(device)
#             mel_pred, postnet_pred, stop_token = model(mel_input, text, pos_mel, pos_text)
#             mel_input = torch.cat([mel_input, postnet_pred[:,-1:]], dim=1)
#     mel_input = mel_input[:,1:,:].squeeze(0)
#     mel_input = mel_input.transpose(0,1).cpu().numpy()    
#     wav = mel2wave(mel_input, params['sample_rate'], params['preemphasis'], params['num_freq'], params['frame_size_ms'], params['frame_hop_ms'], params['min_level_db'], params['num_mel'], params['power'], params['gl_iter'])

#     sd.play(wav, 22050)
#     status = sd.wait()
    
    

