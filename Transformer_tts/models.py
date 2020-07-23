import torch
from torch import nn
import torch.nn.functional as F


from text_utils import symbols
# from data_utils import text2seq

import pdb

from layers import Conv1d, Linear, Conv1dBatchNorm, PosEmbeddingLayer, EncoderLayer, DecoderLayer, TransformerEncoderLayer, TransformerDecoderLayer

class EncoderPrenet(nn.Module):
    def __init__(self, params):
        super(EncoderPrenet, self).__init__()
        
        voc_len = params['voc_len']
        emb_dim = params['emb_dim']
        hid_dim = params['hid_dim']
        num_conv = params['num_conv']
        kernel_size = params['kernel_size']
        dropout = params['dropout']
        pos_dropout = params['pos_dropout']
        num_pos = params['num_pos']
        
        self.embed = nn.Embedding(voc_len, emb_dim, padding_idx=0)

        self.first_conv = Conv1dBatchNorm(emb_dim,
                                         hid_dim,
                                         kernel_size=kernel_size,
                                         stride=1,
                                         padding=int((kernel_size-1)/2),
                                         dilation=1,
                                         dropout=dropout)
        conv = Conv1dBatchNorm(hid_dim,
                               hid_dim,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=int((kernel_size-1)/2),
                               dilation=1,
                               dropout=dropout)

        self.convolutions = nn.ModuleList(
            [conv for _ in range(num_conv - 1)])

        self.projection = Linear(hid_dim, hid_dim, w_init_gain="linear")

        self.pos_dropout = nn.Dropout(pos_dropout)
        
        self.pos_emb = PosEmbeddingLayer(num_pos, hid_dim, padding_idx=0)


    def forward(self, x):
        x = self.embed(x)
        x = x.transpose(1, 2)
        x = self.first_conv(x)
        for conv in self.convolutions:
            x = conv(x)
        x = x.transpose(1, 2)
        x = self.projection(x)
        x = self.pos_emb(x)
        x = self.pos_dropout(x)
        return x


class DecoderPrenet(nn.Module):
    def __init__(self, params):
        super(DecoderPrenet,self).__init__()

        input_dim = params['mel_dim']
        hid_dim = params['hid_dim']
        out_dim = params['out_dim']
        dropout = params['dropout']
        pos_dropout = params['pos_dropout']
        num_pos = params['num_pos']

        self.layers = nn.Sequential(
            Linear(input_dim, hid_dim, w_init_gain="relu"),
            Linear(hid_dim, hid_dim, w_init_gain="relu"))

        self.projection = Linear(hid_dim, out_dim, w_init_gain="linear")
        self.pos_emb = PosEmbeddingLayer(num_pos, out_dim, padding_idx=0)
        self.dropout = dropout 
        self.pos_dropout = nn.Dropout(pos_dropout)
        
    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), self.dropout, self.training)
        x = self.projection(x)
        x = self.pos_emb(x)
        x = self.pos_dropout(x)
        return x



class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        
        hid_dim = params['hid_dim']
        n_layers = params['num_layers']
        n_heads = params['num_heads']
        pf_dim = params['pf_dim']
        dropout = params['dropout']

        self.layers = nn.ModuleList([TransformerEncoderLayer(hid_dim,
                                                             n_heads,
                                                             pf_dim,
                                                             dropout)
                                     for _ in range(n_layers)])

    def forward(self, src, src_key_padding_mask):
        src_aligns = []
        ## change to the correct input shape for MultiheadAttention [seq_len, batch, emb_dim]
        src = src.transpose(0,1)
        for layer in self.layers:
            src, src_align = layer(src, src_key_padding_mask=src_key_padding_mask)
            src_aligns.append(src_align.unsqueeze(1))
        src_aligns = torch.cat(src_aligns, 1)
        src = src.transpose(0,1)
        return src, src_aligns 


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        hid_dim = params['hid_dim']
        n_layers = params['num_layers']
        n_heads = params['num_heads']
        pf_dim = params['pf_dim']
        dropout = params['dropout']
        num_mel = params['num_mel']

        self.layers = nn.ModuleList([TransformerDecoderLayer(hid_dim,
                                                             n_heads,
                                                             pf_dim,
                                                             dropout)
                                     for _ in range(n_layers)])

        self.mel_linear = Linear(hid_dim, num_mel)
        self.stop_linear = Linear(hid_dim, 1, w_init_gain='sigmoid')

    def forward(self, tgt, src, tgt_attn_mask, tgt_key_padding_mask, src_key_padding_mask):
   
        tgt_aligns, tgt_src_aligns = [], []

        ## change to the correct input shape for MultiheadAttention [seq_len, batch, emb_dim]
        tgt = tgt.transpose(0,1)
        src = src.transpose(0,1)
        
        for layer in self.layers:
            tgt, tgt_align, tgt_src_align = layer(tgt, src, tgt_attn_mask=tgt_attn_mask, tgt_key_padding_mask=tgt_key_padding_mask, src_key_padding_mask=src_key_padding_mask)
            tgt_aligns.append(tgt_align.unsqueeze(1))
            tgt_src_aligns.append(tgt_src_align.unsqueeze(1))

        
        tgt_aligns = torch.cat(tgt_aligns, 1)
        tgt_src_aligns = torch.cat(tgt_src_aligns, 1)

        tgt = tgt.transpose(0,1)

        stop_tokens = self.stop_linear(tgt).squeeze(-1)
        mel_out = self.mel_linear(tgt)
        
        return mel_out, stop_tokens, tgt_aligns, tgt_src_aligns

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

        encoder_params = params['encoder'].copy()

        deprenet_params = params['decoderPrenet'].copy()
        
        decoder_params = params['decoder'].copy()

        postnet_params = params['postnet'].copy()

        self.encoder_prenet = EncoderPrenet(enprenet_params)
        self.decoder_prenet = DecoderPrenet(deprenet_params)
        self.encoder = Encoder(encoder_params)
        self.decoder = Decoder(decoder_params)
        self.postnet = Postnet(postnet_params)
        self.loss_fn = TSSLoss(params)

        self.params = params
        self.device = device 

    def make_key_mask(self, pos):
        max_len = torch.max(pos).item()
        ids = (torch.arange(0, max_len)).to(self.device)
        mask = (pos.unsqueeze(1) <= ids).to(torch.bool)
        return mask

    def make_attn_mask(self, mel):
        T = mel.size(1)
        diag_mask = torch.triu(mel.new_ones(T,T)).transpose(0, 1)
        diag_mask[diag_mask == 0] = -float('inf')
        diag_mask[diag_mask == 1] = 0
        return diag_mask

    def output(self, mel, seq, mel_len, seq_len):

        mel_input=F.pad(mel.transpose(1,2),(1,-1)).transpose(1,2) ### switch the input to not leak the information

        seq_key_mask = self.make_key_mask(seq_len)

        mel_key_mask = self.make_key_mask(mel_len)

        mel_attn_mask = self.make_attn_mask(mel)

        seq = self.encoder_prenet(seq)

        seq, seq_align = self.encoder(seq, src_key_padding_mask=seq_key_mask)

        mel_input = self.decoder_prenet(mel_input)

        mel_linear, stop_tokens, mel_align, mel_seq_align = self.decoder(mel_input,
                                                                         seq,
                                                                         tgt_attn_mask=mel_attn_mask,
                                                                         tgt_key_padding_mask=mel_key_mask,
                                                                         src_key_padding_mask=seq_key_mask)

        mel_out = self.postnet(mel_linear)

        return mel_linear, mel_out, stop_tokens, seq_align, mel_align, mel_seq_align, mel_key_mask
        
    def forward(self, mel, seq, mel_len, seq_len, gate):

        mel_linear, mel_out, stop_tokens, seq_align, mel_align, mel_seq_align, mel_key_mask = self.output(mel, seq, mel_len, seq_len)

        mel_linear_loss, mel_post_loss, gate_loss, guide_loss = self.loss_fn((mel_linear, mel_out, stop_tokens),
                                                                             (mel, gate, mel_key_mask, mel_len, seq_len),
                                                                             (mel_align, seq_align, mel_seq_align))


        return mel_linear_loss, mel_post_loss, gate_loss, guide_loss

    def inference(self, text='', input=None, max_len=1024):
        pass

        # if input is not None:
        #     mel_org, seq, mel_len_org, seq_len, gate = input 
        # else:
        #     label = text2seq(text)
        
        #     seq = [torch.LongTensor(label)].to(device)

        #     seq_len = [len(label)].to(device)
        
        # mel = [text.new_zeros(1, max_len, self.params['data']['num_mel']).type(torch.LongTensor).to(device)]

        # mel_len = [max_len]

        # stop = []
        # mel_loss, gate_loss, guide_loss = 0, 0, 0
  
        # for i in range(max_len):
        #     mel_linear, mel_out, stop_tokens, _, _, mel_seq_align, mel_key_mask = self.output(mel, seq, mel_len, seq_len)
        #     stop.append(torch.sigmoid(stop_tokens[:,i]).item())

        #     if i < max_len -1:
        #         mel[:,i+1,:]=mel_out[:,i,:]

        #     if stop[-1]>0.8:
        #         break
        
        # if input is not None:
        #     mel_loss, _, _, gate_loss, guide_loss = self.loss_fn((mel, mel, stop),
        #                                                          (mel_org, gate, mel_key_mask, mel_org_len, seq_len),
        #                                                          mel_seq_align)

        # mel = mel[:,:len(stop),:]

        # return mel, stop, mel_loss, gate_loss, guide_loss

class TSSLoss(nn.Module):
    def __init__(self, params):
        super(TSSLoss, self).__init__()
        
    def forward(self, output, input, alignments):
        mel_linear, mel_post, gate_out = output
        mel_target, gate_target, mel_mask, mel_len, seq_len = input
        mel_mask = ~mel_mask

        mel_target = mel_target.transpose(1,2).masked_select(mel_mask.unsqueeze(1))
        mel_linear = mel_linear.transpose(1,2).masked_select(mel_mask.unsqueeze(1))
        mel_post = mel_post.transpose(1,2).masked_select(mel_mask.unsqueeze(1))

        gate_target = gate_target.masked_select(mel_mask)
        gate_out = gate_out.masked_select(mel_mask)

        mel_linear_loss = nn.L1Loss()(mel_linear, mel_target)
        mel_post_loss = nn.L1Loss()(mel_post, mel_target)

        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        guide_loss_0 = self.guide_loss(alignments[0], mel_len, mel_len)

        guide_loss_1 = self.guide_loss(alignments[1], seq_len, seq_len)

        guide_loss_2 = self.guide_loss(alignments[2], seq_len, mel_len)

        guide_loss = guide_loss_0 + guide_loss_1 + guide_loss_2
        
        return mel_linear_loss, mel_post_loss, gate_loss, guide_loss
    
    def guide_loss(self, alignments, seq_len, mel_len):

        batch, num_layers, T, L = alignments.shape

        cuda_check = alignments.is_cuda
        if cuda_check:
            device = 'cuda'
        else:
            device = 'cpu'

        W = alignments.new_zeros(batch, T, L)
        mask = alignments.new_zeros(batch, T, L)

        for i, (t, l) in enumerate(zip(mel_len, seq_len)):
            mel_seq = (torch.arange(t).to(torch.float32).unsqueeze(-1)/t).to(device)
            text_seq = (torch.arange(l).to(torch.float32).unsqueeze(0)/t).to(device)
            x = torch.pow(mel_seq-text_seq, 2)
            W[i, :t, :l] += (1-torch.exp(-3.125*x)).to(device)
            mask[i, :t, :l] = 1

        applied_align = alignments[:,-2:]

        losses = applied_align*(W.unsqueeze(1))

        losses = losses.masked_select(mask.unsqueeze(1).to(torch.bool))
        
        return torch.mean(losses)
        
        
        

    
