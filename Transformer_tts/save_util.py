import os
import torch
import pdb
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def plot_melspec(target, melspec, melspec_post, mel_lengths):
    fig, axes = plt.subplots(3, 1, figsize=(20,30))
    T = mel_lengths[-1]
    axes[0].imshow(target[-1].transpose(0,1)[:,:T],
                   origin='lower',
                   aspect='auto')

    axes[1].imshow(melspec[-1].transpose(0,1)[:,:T],
                   origin='lower',
                   aspect='auto')

    axes[2].imshow(melspec_post[-1].transpose(0,1)[:,:T],
                   origin='lower',
                   aspect='auto')

    return fig



def get_writer(output_directory, log_directory):
    logging_path='{output_directory}/{log_directory}'
    
    if os.path.exists(logging_path):
        raise Exception('The experiment already exists')
    else:
        os.mkdir(logging_path)
        writer = TTSWriter(logging_path)
            
    return writer

class TTSWriter(SummaryWriter):
    def __init__(self, log_dir):
        super(TTSWriter, self).__init__(log_dir)
        
    def add_losses(self, mel_linear_loss, mel_post_loss, gate_loss, guide_loss, global_step, phase):
        self.add_scalar('{phase}_mel_linear_loss', mel_linear_loss, global_step)
        self.add_scalar('{phase}_mel_pos_loss', mel_post_loss, global_step)
        self.add_scalar('{phase}_gate_loss', gate_loss, global_step)
        self.add_scalar('{phase}_guide_loss', guide_loss, global_step)
        
    def add_specs(self, mel_padded, mel_out, mel_out_post, mel_lengths, global_step, phase):
        mel_fig = plot_melspec(mel_padded, mel_out, mel_out_post, mel_lengths)
        self.add_figure('{phase}_melspec', mel_fig, global_step)
        
    # def add_alignments(self, enc_alignments, dec_alignments, enc_dec_alignments,
    #                    text_padded, mel_lengths, text_lengths, global_step, phase):
    #     enc_align_fig = plot_alignments(enc_alignments, text_padded, mel_lengths, text_lengths, 'enc')
    #     self.add_figure('{phase}_enc_alignments', enc_align_fig, global_step)

    #     dec_align_fig = plot_alignments(dec_alignments, text_padded, mel_lengths, text_lengths, 'dec')
    #     self.add_figure('{phase}_dec_alignments', dec_align_fig, global_step)

    #     enc_dec_align_fig = plot_alignments(enc_dec_alignments, text_padded, mel_lengths, text_lengths, 'enc_dec')
    #     self.add_figure('{phase}_enc_dec_alignments', enc_dec_align_fig, global_step)
        
    # def add_gates(self, gate_out, global_step, phase):
    #     gate_fig = plot_gate(gate_out)
    #     self.add_figure('{phase}_gate_out', gate_fig, global_step)
