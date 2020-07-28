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
    optimizer.zero_grad()
    for batch_idx, _data in enumerate(train_loader):
        iteration += 1 
        adjust_learning_rate(optimizer, iteration, params['lr'], warmup_step=params['warmup_step'])
        
        mel, seq, gate, mel_pos, seq_pos = _data 
        mel_linear_loss,mel_post_loss,gate_loss, guide_loss, _ = model(mel.to(device),
                                                                       seq.to(device),
                                                                       mel_pos.to(device),
                                                                       seq_pos.to(device),
                                                                       gate.to(device))
        
        total_loss = (mel_linear_loss+mel_post_loss+gate_loss+guide_loss)#/params['accumulation']
        total_loss.backward()
        
        if iteration % params['accumulation'] == 0:
            nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip_thresh'])
            optimizer.step()
            losses = {"mel_linear_loss":mel_linear_loss.item(),
                      "mel_post_loss":mel_post_loss.item(),
                      "gate_loss":gate_loss.item(),
                      "guide_loss":guide_loss.item()}
            writer.add_losses(losses, iteration//params['accumulation'], 'Train')
            optimizer.zero_grad()
            
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Iteration: {} [{}/{} ({:.0f}%)]\tMel_linear Loss: {:.6f}\tMel_post Loss: {:.6f}\tGate Loss: {:.6f}\tGuide Loss: {:.6f}'.format(
                iteration, batch_idx * len(mel), data_len,
                100. * batch_idx / len(train_loader), mel_linear_loss.item(), mel_post_loss.item(), gate_loss.item(), guide_loss.item()))
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
            mel_linear, gate_out, seq_align, mel_align, mel_seq_align, _ = model.output(mel.to(device),
                                                                                        seq.to(device),
                                                                                        mel_len.to(device),
                                                                                        seq_len.to(device))

            mel_linear_loss, mel_post_loss, gate_loss, guide_loss, mel_post = model(mel.to(device),
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

        print('Test set: Average Total loss: {:.4f}'.format(total_loss))

    mel_inf, mel_post_inf = model.inference(seq.to(device), seq_len.to(device))

    mels = (mel.detach().cpu(), mel_inf.detach().cpu(), mel_post_inf.detach().cpu())
    writer.add_specs(mels,
                     mel_len.detach().cpu(),
                     iteration//params['accumulation'], 'Validation_without_mel')

    losses = {"mel_linear_loss":mel_linear_loss.item(),
              "mel_post_loss":mel_post_loss.item(),
              "gate_loss":gate_loss.item(),
              "guide_loss":guide_loss.item()}
    writer.add_losses(losses,iteration//params['accumulation'], 'Validation')

    mels = (mel.detach().cpu(),mel_linear.detach().cpu(), mel_post.detach().cpu())
    writer.add_specs(mels,
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


    
    

