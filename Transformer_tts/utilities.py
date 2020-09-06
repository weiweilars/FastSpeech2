import os
import pdb
import json
import numpy as np
import matplotlib.pyplot as plt
import gc 

import torch
import torch.nn.functional as F
import torch.nn as nn
import sounddevice as sd
import soundfile as sf

# from data_utils import text2seq, mel2wave

def adjust_learning_rate(optimizer, step_num, init_lr, warmup_step=10000):
    lr = init_lr* min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model, criteriate, device, train_loader, optimizer, iteration, params, writer, weight):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        iteration += 1 
        mel, seq, gate, mel_len, seq_len = _data
        mel, seq, gate, mel_len, seq_len = mel.to(device), seq.to(device), gate.to(device), mel_len.to(device), seq_len.to(device)

        mel_linear, mel_out, gate_out, seq_align, mel_align, mel_seq_align, mel_key_mask = model(mel, seq, mel_len, seq_len, gate)

        mel_linear_loss, mel_post_loss, gate_loss, guide_loss = criteriate((mel_linear, mel_out, gate_out),
                                                                           (mel, gate, mel_key_mask, mel_len, seq_len),
                                                                           (mel_align, seq_align, mel_seq_align))
        
        total_loss = (mel_linear_loss+mel_post_loss+weight*gate_loss+guide_loss)/params['accumulation']
        total_loss.backward()
        
        if iteration % params['accumulation'] == 0:
            adjust_learning_rate(optimizer, iteration/params['accumulation'], params['lr'], warmup_step=params['warmup_step'])
            nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip_thresh'])
            optimizer.step()
            optimizer.zero_grad()
            losses = {"mel_linear_loss":mel_linear_loss.item(),
                      "mel_post_loss":mel_post_loss.item(),
                      "gate_loss":gate_loss.item(),
                      "guide_loss":guide_loss.item()}
            writer.add_losses(losses, iteration//params['accumulation'], 'Train')
            
            
        if batch_idx % (50*params['accumulation']) == 0 or batch_idx == data_len:
            print('Train Iteration: {} [{}/{} ({:.0f}%)]\tMel_linear Loss: {:.6f}\tMel_post Loss: {:.6f}\tGate Loss: {:.6f}\tGuide Loss: {:.6f}'.format(
                iteration, batch_idx * len(mel), data_len,
                100. * batch_idx / len(train_loader), mel_linear_loss.item(), mel_post_loss.item(), gate_loss.item(), guide_loss.item()))
            for param_group in optimizer.param_groups:
                print(param_group['lr'])

        
        del mel_linear, mel_out, gate_out, seq_align, mel_align, mel_seq_align, mel_key_mask
        gc.collect()
        torch.cuda.empty_cache()

    return iteration

def validate(model, criteriate, device, val_loader, iteration, writer, params):
    print('\nevaluating…')
    model.eval()
    with torch.no_grad():
        mel_linear_avg, mel_post_avg, gate_avg, guide_avg = 0, 0, 0, 0
        for i, batch in enumerate(val_loader):
            mel, seq, gate, mel_len, seq_len = batch
            mel, seq, gate, mel_len, seq_len = mel.to(device), seq.to(device), gate.to(device), mel_len.to(device), seq_len.to(device)
            mel_linear, mel_post, gate_out, seq_align, mel_align, mel_seq_align, mel_key_mask = model(mel, seq, mel_len, seq_len, gate)

            mel_linear_loss, mel_post_loss, gate_loss, guide_loss = criteriate((mel_linear, mel_post, gate_out),
                                                                               (mel, gate, mel_key_mask, mel_len, seq_len),
                                                                               (mel_align, seq_align, mel_seq_align))
            mel_linear_avg += mel_linear_loss.item()
            mel_post_avg += mel_post_loss.item()
            gate_avg += gate_loss.item()
            guide_avg += guide_loss.item()


        print('Test set: Average mel linear loss: {:.6f}'.format(mel_linear_avg/(i+1)))
        print('Test set: Average mel post loss: {:.6f}'.format(mel_post_avg/(i+1)))
        print('Test set: Average gate loss: {:.6f}'.format(gate_avg/(i+1)))
        print('Test set: Average guide loss: {:.6f}'.format(guide_avg/(i+1)))

        total_loss = (mel_linear_avg + mel_post_avg + gate_avg + guide_avg)/(i+1)

        print('Test set: Average Total loss: {:.4f}'.format(total_loss))

        mel_inf, mel_post_inf, gate_out_inf = model.inference(seq[-1:], seq_len[-1:], test_len=mel_len[-1:])

        #gate_out_inf = torch.tensor([gate_out_inf]).to(device)
        
        mel_linear_loss_inf, mel_post_loss_inf, gate_loss_inf, _ = criteriate((mel_inf, mel_post_inf, gate_out_inf),
                                                                  (mel[-1:], gate[-1:], mel_key_mask[-1:], mel_len[-1:], seq_len[-1:]))
        print('Inference \tMel_linear Loss: {:.6f} \tMel_post Loss: {:.6f} \tMel_gate Loss: {:.6f}'.format(mel_linear_loss_inf, mel_post_loss_inf, gate_loss_inf))
        
    
    mels = (mel.detach().cpu(), mel_inf.detach().cpu(), mel_post_inf.detach().cpu())
    writer.add_specs(mels,
                     mel_len.detach().cpu(),
                     iteration//params['accumulation'], 'Validation_without_mel')

    losses = {"mel_linear_loss":mel_linear_loss.item(),
              "mel_post_loss":mel_post_loss.item(),
              "gate_loss":gate_loss.item(),
              "guide_loss":guide_loss.item()}
    writer.add_losses(losses,iteration//params['accumulation'], 'Validation')

    losses = {"mel_linear_loss":mel_linear_loss_inf.item(),
              "mel_post_loss":mel_post_loss_inf.item(),
              "gate_loss":gate_loss_inf.item()}
    writer.add_losses(losses,iteration//params['accumulation'], 'Inference')

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
    writer.add_gates(gate_out_inf.detach().cpu(),
                    iteration//params['accumulation'], 'Inference')

    torch.cuda.empty_cache()
    return total_loss

def find_lr(model, optimizer, train_loader, init_lr=1e-8, final_lr=10.0):
    numer_in_epoch = len(train_loader) -1
    update_step = (final_lr / init_lr) ** (1 / numer_in_epoch)

    lr = init_lr
    optmizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    losses = []
    log_lrs = []
    for batch_num, data in train_loader:
        mel, seq, gate, mel_pos, seq_pos = data
        optimizer.zero_grad()
        mel_linear_loss,mel_post_loss,gate_loss, guide_loss, _ = model(mel.to(device),
                                                                       seq.to(device),
                                                                       mel_pos.to(device),
                                                                       seq_pos.to(device),
                                                                       gate.to(device))
        loss = (mel_linear_loss+mel_post_loss+gate_loss+guide_loss)
        if batch_num>1 and loss > 4*best_loss:
            return log_lrs[10:-5], losses[10:-5]

        if loss < best_loss or batch_num == 1:
            best_loss = loss

        losses.append(loss)
        log_lrs.append(math.log10(lr))

        loss.backward()
        optimizer.step()

        lr *= update_step
        optimizer.param_groups[0]["lr"]=lr

        plt.plot(log_lrs[10:-5], losses[10,-5])
        
    return log_lrs[10:-5], losses[10,-5]
