import torch
from torch import nn
import torch.nn.functional as F


from text_utils import symbols

import pdb

from layers import Conv1d, Linear, EncoderEmbeddingLayer, PosEmbeddingLayer, EncoderLayer, DecoderLayer

class EncoderPrenet(nn.Module):
    def __init__(self, params):
        super(EncoderPrenet, self).__init__()
        
        voc_len = params['voc_len']
        emb_dim = params['emb_dim']
        hid_dim = params['hid_dim']
        num_conv = params['num_conv']
        kernel_size = params['kernel_size']
        device = params['device']
        dropout = params['dropout']
        pos_dropout = params['pos_dropout']
        num_pos = params['num_pos']
        
        self.embed = EncoderEmbeddingLayer(voc_len, emb_dim, padding_idx=0)

        self.first_convolution = nn.Sequential(
            Conv1d(emb_dim,
                        hid_dim,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int((kernel_size-1)/2),
                        dilation=1,
                        w_init_gain='linear'),
            nn.BatchNorm1d(hid_dim),
        )

        convolution = nn.Sequential(
            Conv1d(emb_dim,
                        emb_dim,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int((kernel_size-1)/2),
                        dilation=1,
                        w_init_gain='linear'),
            nn.BatchNorm1d(hid_dim),
        )

        self.convolutions = nn.ModuleList(
            [convolution for _ in range(num_conv - 1)])

        self.projection = Linear(hid_dim, hid_dim, w_init_gain="linear")

        self.dropout = dropout

        self.pos_emb = PosEmbeddingLayer(num_pos, hid_dim, device, padding_idx=0)


    def forward(self, x, x_pos):
        x = self.embed(x)
        x = x.transpose(1, 2)
        x = F.dropout(F.relu(self.first_convolution(x)), self.dropout, self.training)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), self.dropout, self.training)
        x = x.transpose(1, 2)
        x = self.projection(x)
        x = self.pos_emb(x)
        return x


class DecoderPrenet(nn.Module):
    def __init__(self, params):
        super(DecoderPrenet,self).__init__()

        input_dim = params['mel_dim']
        hid_dim = params['hid_dim']
        out_dim = params['out_dim']
        dropout = params['dropout']
        pos_dropout = params['pos_dropout']
        device = params['device']
        num_pos = params['num_pos']

        self.layers = nn.Sequential(
            Linear(input_dim, hid_dim, w_init_gain="relu"),
            Linear(hid_dim, hid_dim, w_init_gain="relu"))

        self.projection = Linear(hid_dim, out_dim, w_init_gain="linear")
        self.pos_emb = PosEmbeddingLayer(num_pos, out_dim, device, padding_idx=0)

        self.dropout = dropout 
        
    def forward(self, x, x_pos):
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), self.dropout, self.training)
        x = self.projection(x)
        x = self.pos_emb(x)
        return x



class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        
        hid_dim = params['hid_dim']
        n_layers = params['num_layers']
        n_heads = params['num_heads']
        pf_dim = params['pf_dim']
        device = params['device']
        dropout = params['dropout']

        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                     for _ in range(n_layers)])

    def forward(self, src, src_mask):

        for layer in self.layers:
            src = layer(src, src_mask)

        return src



class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        hid_dim = params['hid_dim']
        n_layers = params['num_layers']
        n_heads = params['num_heads']
        pf_dim = params['pf_dim']
        device = params['device']
        dropout = params['dropout']
        num_mel = params['num_mel']

        self.device = device 

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.mel_linear = Linear(hid_dim, num_mel)
        self.stop_linear = Linear(hid_dim, 1, w_init_gain='sigmoid')

    def forward(self, trg, enc_src, trg_mask, src_mask):

        for layer in self.layers:
            trg, _ = layer(trg, enc_src, trg_mask, src_mask)

        stop_tokens = self.stop_linear(trg)
        mel_out = self.mel_linear(trg)
        
        return mel_out, stop_tokens

class Postnet(nn.Module):
    def __init__(self, params):
        super(Postnet, self).__init__()

        num_mel = params['num_mel']
        hid_dim = params['hid_dim']
        kernel_size = params['kernel_size']
        num_conv = params['num_conv']
        dropout = params['dropout']
        
        self.first_cov = nn.Sequential(
                Conv1d(num_mel, hid_dim,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=int((kernel_size - 1) / 2),
                            dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hid_dim))

        convolution = nn.Sequential(
            Conv1d(hid_dim,
                        hid_dim,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int((kernel_size-1)/2),
                        dilation=1,
                        w_init_gain='tanh'),
            nn.BatchNorm1d(hid_dim),
        )

        self.convolutions = nn.ModuleList(
            [convolution for _ in range(num_conv - 2)])
        

        self.last_cov = nn.Sequential(
            Conv1d(hid_dim, num_mel,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=int((kernel_size - 1) / 2),
                            dilation=1, w_init_gain='linear'),
            nn.BatchNorm1d(num_mel))

        self.normalization=nn.BatchNorm1d(num_mel)
        self.dropout = dropout

    def forward(self, x):
        x = x.transpose(1,2)
        _x = F.dropout(torch.tanh(self.first_cov(x)), self.dropout, self.training)
        for i in range(len(self.convolutions)):
            _x = F.dropout(torch.tanh(self.convolutions[i](
                _x)), self.dropout, self.training)
        _x = F.dropout(self.last_cov(_x), self.dropout, self.training)
        x = x + _x
        x = x.transpose(1,2)
        return x


class Model(nn.Module):
    def __init__(self,params, device):
        super(Model, self).__init__()
        
        enprenet_params = params['encoderPrenet'].copy()
        enprenet_params['voc_len'] = len(symbols)
        enprenet_params['device'] = device

        encoder_params = params['encoder'].copy()
        encoder_params['device'] = device

        deprenet_params = params['decoderPrenet'].copy()
        deprenet_params['device'] = device

        decoder_params = params['decoder'].copy()
        decoder_params['device'] = device

        postnet_params = params['postnet'].copy()

        self.encoder_prenet = EncoderPrenet(enprenet_params)
        self.decoder_prenet = DecoderPrenet(deprenet_params)
        self.encoder = Encoder(encoder_params)
        self.decoder = Decoder(decoder_params)
        self.postnet = Postnet(postnet_params)

        self.device = device

    def make_seq_mask(self, seq_pos):
        return seq_pos.ne(0).type(torch.float).unsqueeze(1).unsqueeze(2)

    def make_mel_mask(self, mel_pos):

        batch_size = mel_pos.size(0)
        mel_len = mel_pos.size(1)

        mel_pad_mask = mel_pos.ne(0).unsqueeze(1)

        mel_sub_mask = torch.tril(torch.ones((mel_len, mel_len), device = self.device)).bool()

        mel_mask = mel_pad_mask.repeat(1, mel_len, 1).bool() & mel_sub_mask

        return mel_mask.unsqueeze(1), mel_pad_mask
    
        
    def forward(self, mel, seq, mel_pos, seq_pos):

        seq_mask = self.make_seq_mask(seq_pos)

        mel_mask, mel_pad_mask = self.make_mel_mask(mel_pos)

        seq = self.encoder_prenet(seq, seq_pos)

        seq = self.encoder(seq, seq_mask)

        mel = self.decoder_prenet(mel, mel_pos)

        mel_linear, stop_tokens = self.decoder(mel, seq, mel_mask, seq_mask)

        mel_out = self.postnet(mel_linear)

        return mel_linear, mel_out, stop_tokens, mel_pad_mask


class TSSLoss(nn.Module):
    def __init__(self, params):
        super(TSSLoss, self).__init__()

        self.num_output_per_step = params['data']['num_output_per_step']
        self.p = params['train']['loss_punish']
        self.num_mel = params['data']['num_mel']

    def forward(self, output, input):
        mask = output[3]
        seq_mask = mask.squeeze(1)
        mel_mask = mask.repeat(1,self.num_mel,1).transpose(1,2)
        mel_target = input[0][:,self.num_output_per_step:,:]
        mel_target = mel_target.masked_select(mel_mask[:,self.num_output_per_step:,:])

        mel_output = output[0][:,:-self.num_output_per_step,:]
        mel_output = mel_output.masked_select(mel_mask[:,self.num_output_per_step:,:])
        
        mel_post_output = output[1][:,:-self.num_output_per_step,:]
        mel_post_output = mel_post_output.masked_select(mel_mask[:,self.num_output_per_step:,:])

        gate_target = input[2]
        batch_size = gate_target.shape[0]
        gate_target = (gate_target.ne(0) & gate_target.lt(gate_target.max(-1, keepdim=True)[0])).float()
        gate_target = gate_target[:,self.num_output_per_step:]
        gate_target = gate_target.masked_select(seq_mask[:,self.num_output_per_step:])

        gate_out = output[2].view(1,batch_size,-1).squeeze(0)[:,:-self.num_output_per_step]
        gate_out = gate_out.masked_select(seq_mask[:, self.num_output_per_step:])

        mel_pre_loss = nn.L1Loss()(mel_output, mel_target)
        mel_post_loss = nn.L1Loss()(mel_post_output, mel_target)

        mel_loss = self.p*(mel_pre_loss + mel_post_loss)

        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        total_loss = mel_loss + gate_loss

        return total_loss, mel_pre_loss, mel_post_loss, gate_loss 
        
        
        

    
