import os
import torch
import pdb
import json
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def plot_melspec(mels, mel_lengths):
    fig, axes = plt.subplots(len(mels), 1, figsize=(20, 10*len(mels)))
    T = mel_lengths[-1]
    for i, mel in enumerate(mels):
        axes[i].imshow(mel[-1].transpose(0,1)[:,:T],
                       origin='lower',
                       aspect='auto')
    return fig

def plot_alignments(alignments, mel_len=[0], seq_len=[0], att_type='mel_seq'):
    
    if len(alignments.shape) == 4:
        alignments.unsqueeze_(2)
    
    n_layers = alignments.size(1)
    n_heads = alignments.size(2)
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(5*n_heads,5*n_layers))
    L, T = seq_len[-1], mel_len[-1]
    
    for layer in range(n_layers):
        for head in range(n_heads):
            if att_type=='seq':
                align = alignments[-1, layer, head].contiguous()
                if n_heads == 1:
                    axes[layer].imshow(align[:L, :L], aspect='auto')
                    axes[layer].xaxis.tick_top()
                else:
                    axes[layer, head].imshow(align[:L, :L], aspect='auto')
                    axes[layer, head].xaxis.tick_top()

            elif att_type=='mel':
                align = alignments[-1, layer, head].contiguous()
                if n_heads==1:
                    axes[layer].imshow(align[:T, :T], aspect='auto')
                    axes[layer].xaxis.tick_top()
                else:
                    axes[layer, head].imshow(align[:T, :T], aspect='auto')
                    axes[layer, head].xaxis.tick_top()

            elif att_type=='mel_seq':
                align = alignments[-1, layer, head].transpose(0,1).contiguous()
                if n_heads==1:
                    axes[layer].imshow(align[:L, :T], aspect='auto')
                else:
                    axes[layer, head].imshow(align[:L, :T], aspect='auto')
 
    return fig

def plot_gate(gate_out):
    fig = plt.figure(figsize=(10,5))
    plt.plot(torch.sigmoid(gate_out[-1]))
    return fig

def get_writer(output_directory, log_directory):
    logging_path=os.path.join(output_directory, log_directory)
    
    if not os.path.exists(logging_path):
        os.mkdir(logging_path)
    writer = TTSWriter(logging_path)
            
    return writer

class TTSWriter(SummaryWriter):
    def __init__(self, log_dir):
        super(TTSWriter, self).__init__(log_dir)
        
    def add_losses(self, mels, global_step, phase):
        for name, value in mels.items():
            self.add_scalar('{}_{}'.format(phase, name), value, global_step)
        
    def add_specs(self, mels, mel_len, global_step, phase):
        mel_fig = plot_melspec(mels, mel_len)
        self.add_figure('{}_melspec'.format(phase), mel_fig, global_step)
        
    def add_alignments(self, mel_align, seq_align, mel_seq_align, mel_len, seq_len, global_step, phase):
        enc_align_fig = plot_alignments(seq_align, seq_len=seq_len, att_type='seq')
        self.add_figure('{}_seq_alignments'.format(phase), enc_align_fig, global_step)

        dec_align_fig = plot_alignments(mel_align, mel_len=mel_len, att_type='mel')
        self.add_figure('{}_mel_alignments'.format(phase), dec_align_fig, global_step)

        enc_dec_align_fig = plot_alignments(mel_seq_align, mel_len=mel_len, seq_len=seq_len, att_type='mel_seq')
        self.add_figure('{}_mel_seq_alignments'.format(phase), enc_dec_align_fig, global_step)
        
    def add_gates(self, gate_out, global_step, phase):
        gate_fig = plot_gate(gate_out)
        self.add_figure('{}_gate_out'.format(phase), gate_fig, global_step)


def save_model(model, optimizer, iteration, params_dict, model_path='./checkpoint/saved_model', model_name='model.pt'):
    print("Saving the model, optimizer, and params at {} to {}".format(iteration,model_name))
    model_file_path = os.path.join(model_path, model_name)
    torch.save({'iteration':iteration,
                'model_dict':model.state_dict(),
                'optimizer':optimizer.state_dict()}, model_file_path)

    params_file = os.path.join(model_path, 'params.json')
    with open(params_file, 'w') as f:
        json.dump(params_dict, f)
