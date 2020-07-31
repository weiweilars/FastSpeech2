import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

from math import sqrt, cos, sin
import numpy as np
from numpy import inf


def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()

class Linear(nn.Linear):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(Linear, self).__init__(in_dim, out_dim, bias)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(w_init_gain))

class Conv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', w_init_gain="linear"):
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(w_init_gain))

        
class Conv1dBatchNorm(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, dilation=1, bias=True, activation="linear", dropout=0.1):
        super(Conv1dBatchNorm,self).__init__()

        self.conv = Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, w_init_gain=activation)
        self.norm = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)

        self.act = activation
        if activation is not 'linear':
            self.activation = _get_activation_fn(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.act is not 'linear':
            x = self.activation(x)
        out = self.dropout(x)
        return out 

class PosEmbeddingLayer(nn.Module):
    def __init__(self, num_pos, hid_dim):
        super(PosEmbeddingLayer, self).__init__()
        self.register_buffer('pe', self._get_pos_matrix(num_pos, hid_dim))
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x_len = x.shape[1]
        pos = self.pe[:x_len,:]*self.alpha
        return pos

    def _get_pos_matrix(self, num_pos, hid_dim):
        pe = torch.zeros(num_pos, hid_dim)
        position = torch.arange(0, num_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, hid_dim, 2).float() / hid_dim)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        return pe


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, device, dropout=0.1, bias=True):
        '''The class to calculate the mutltihead attentions 

        Args: 
          hid_dim : the output of the embedding dimension 
          h_head: the number of heads to choose 
          dropout: the rate to dropout 
          device: cup or gpu 

        '''
        super().__init__()

        # make sure the hid_dim can be evenly divided in to n_heads
        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = Linear(hid_dim, hid_dim, bias=bias)
        self.fc_k = Linear(hid_dim, hid_dim, bias=bias)
        self.fc_v = Linear(hid_dim, hid_dim, bias=bias)

        self.fc_o = Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        # self.scale = torch.FloatTensor([self.head_dim]).to(device)


    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        '''The forward calculation of the neural network

        Args:
          query: copy of the output of the word embedding + pos embedding
          key: copy of the output of the word embedding + pos embedding
          value: copy of the output of the word embedding + pos embedding
          mask: padding is masked by 0, others are 1 
        '''

        batch_size = query.shape[0]
        Q_len = query.shape[1]
        K_len = key.shape[1]

        # query = [batch_size, query_len, hid_dim]
        # key = [batch_size, key_len, hid_dim]
        # value = [batch_size, value_len, hid_dim]
        Q = self.fc_q(query).view(batch_size, -1, self.n_heads, self.head_dim)
        K = self.fc_k(key).view(batch_size, -1, self.n_heads, self.head_dim)
        V = self.fc_v(value).view(batch_size, -1, self.n_heads, self.head_dim)

        # Q = [batch_size, query_len, n_heads, head_dim]
        # K = [batch_size, key_len, n_heads, head_dim]
        # V = [batch_size, value_len, n_heads, head_dim]

        Q = Q.transpose(1,2).contiguous().view(batch_size*self.n_heads, -1, self.head_dim)
        K = K.transpose(1,2).contiguous().view(batch_size*self.n_heads, -1, self.head_dim)
        V = V.transpose(1,2).contiguous().view(batch_size*self.n_heads, -1, self.head_dim)

        # Q = [batch size*n_heads, query len, head dim]
        # K = [batch size*n_heads, key len, head dim]
        # V = [batch size*n_heads, value len, head dim]

        Q = Q/(self.head_dim ** 1/4)
        K = K/(self.head_dim ** 1/4)
        
        energy = torch.bmm(Q, K.transpose(1,2))

        if attn_mask is not None:
            # if self.training:
            #     energy.masked_fill_(attn_mask, float('-inf'))
            # else:
            if attn_mask.dtype == torch.bool:
                energy.masked_fill_(attn_mask, -1e9)
            else:
                energy += attn_mask

        # energy = [batch_size*n_heads, query_len, key_len]
        if key_padding_mask is not None:
            # masked_fill(mask, value) -> Tensor
            energy = energy.view(batch_size, self.n_heads, Q_len, K_len)
            energy.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            energy = energy.view(batch_size*self.n_heads, Q_len, K_len)
                
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        # attentions = [batch_size*n_heads, query_len, key_len]

        x = torch.bmm(attention, V)
        # x = [batch_size*n_heads, query_len, head_dim]

        attention = attention.view(batch_size, self.n_heads, Q_len, K_len)
        # attention = [batch_size, n_heads, query_len, key_len]

        x = x.view(batch_size, self.n_heads, -1, self.head_dim)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch_size, query_len, hid_dim]

        
        x = self.fc_o(x)

        # x = [batch_size, query_len, hid_dim]
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        '''Calculate the positionwiseFeedforward 

        Args:
          hid_dim : the output dimension of embedding for each word
          pf_dim: the dimension between two linear transformation, usually much bigger than hid_dim
          dropout: the dropout rate of the dropout layer
        '''
        super().__init__()

        self.fc_1 = Linear(hid_dim, pf_dim, w_init_gain="relu")
        self.fc_2 = Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x = [batch_size, seq_len, hid_dim]

        x = self.dropout(F.relu(self.fc_1(x)))

        # x = [batch_size, seq_len, pf_dim]

        x = self.fc_2(x)

        # x = [batch_size, seq_len, hid_dim]

        return x

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        '''One Encoder Layer

        Args:
          hid_dim: the dimension output from embedding for one word 
          n_heads: how many heads is chosen for multiheads attention 
          pd_dim: the hiding dimension in the positionwiseFeedforward 
          dropout: the dropout rate
          device: gpu or cpu 
        '''
        super(EncoderLayer,self).__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, device, dropout)
        
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_attn_mask=None, src_key_padding_mask=None):

        # src = [batch_size, src_len, hid_dim]
        # src_mask = [batch_size, src_len]

        _src, src_attention = self.self_attention(src, src, src, attn_mask=src_attn_mask, key_padding_mask=src_key_padding_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch_size, src_len, hid_dim]

        # posiionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch_size, src_len, hid_dim]

        return src, src_attention

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        '''One Encoder Layer

        Args:
          hid_dim: the dimension output from embedding for one word 
          n_heads: how many heads is chosen for multiheads attention 
          pd_dim: the hiding dimension in the positionwiseFeedforward 
          dropout: the dropout rate
          device: gpu or cpu 
        '''
        super(DecoderLayer,self).__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, device, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, device, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, src, tgt_attn_mask=None, src_attn_mask=None, tgt_key_padding_mask=None, src_key_padding_mask=None):

        # trg = [batch_size, trg_len, hid_dim]
        # enc_src = [batch_size, src_len, hid_dim]
        # trg_mask = [batch_size, trg_len]
        # src_mask = [batch_size, src_len]

        # self attention
        _tgt, tgt_attention = self.self_attention(tgt, tgt, tgt, attn_mask=tgt_attn_mask, key_padding_mask=tgt_key_padding_mask)

        tgt = self.self_attn_layer_norm(tgt + self.dropout(_tgt))

        # trg = [batch_size, trg_len, hid_dim]

        # arg : key query value
        _tgt, tgt_src_attention = self.encoder_attention(tgt, src, src, attn_mask=src_attn_mask, key_padding_mask=src_key_padding_mask)

        # dropout, residual connection and layer norm
        tgt = self.enc_attn_layer_norm(tgt + self.dropout(_tgt))

        # trg = [batch_size, trg_len, hid_dim]

        # positionwise feedforward
        _tgt = self.positionwise_feedforward(tgt)

        # dropout, residual and layer norm
        tgt = self.ff_layer_norm(tgt + self.dropout(_tgt))

        # trg = [batch_size, trg_len, hid_dim]
        # attention = [batch_size, n_heads, trg_len, src_len]

        return tgt, tgt_attention, tgt_src_attention

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim=2048,
                 dropout=0.1,
                 activation="relu"):

        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout)

        self.ff_linear1 = Linear(hid_dim, pf_dim, w_init_gain=activation)
        self.ff_linear2 = Linear(pf_dim, hid_dim)

        self.ff_norm1 = nn.LayerNorm(hid_dim)
        self.ff_norm2 = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_attn_mask=None, src_key_padding_mask=None):
        src2, src_align = self.self_attn(src, src, src, attn_mask=src_attn_mask, key_padding_mask=src_key_padding_mask)
        src = self.ff_norm1(src + self.dropout(src2))

        src2 = self.ff_linear2(self.dropout(F.relu(self.ff_linear1(src))))
        src = self.ff_norm2(src + self.dropout(src2))

        return src, src_align

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout)

        self.ff_linear1 = Linear(hid_dim, pf_dim, w_init_gain=activation)
        self.ff_linear2 = Linear(pf_dim, hid_dim)

        self.ff_norm1 = nn.LayerNorm(hid_dim)
        self.ff_norm2 = nn.LayerNorm(hid_dim)
        self.ff_norm3 = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, src, tgt_attn_mask=None, src_attn_mask=None, tgt_key_padding_mask=None, src_key_padding_mask=None):

       
        tgt2, tgt_align = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_attn_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.ff_norm1(tgt + self.dropout(tgt2))

        tgt2, tgt_src_align = self.cross_attn(tgt, src, src, attn_mask=src_attn_mask, key_padding_mask=src_key_padding_mask)
        tgt = self.ff_norm2(tgt + self.dropout(tgt2))

        tgt2 = self.ff_linear2(self.dropout(F.relu(self.ff_linear1(tgt))))
        tgt = self.ff_norm3(tgt + self.dropout(tgt2))

        return tgt, tgt_align, tgt_src_align
        
        
        
if __name__ == "__main__":

    # def make_key_mask(pos):
    #     # true will be -inf
    #     # false will be same
    #     max_len = torch.max(pos).item()
    #     ids = (torch.arange(0, max_len))
    #     mask = (pos.unsqueeze(1) <= ids).to(torch.bool)
    #     return mask

    # def make_attn_mask(mel, mask_future=True, num_neigbour=0):
    #     # true will be -inf
    #     # false will be same
    #     T = mel.size(1)
        
    #     if mask_future:
    #         past_mask = ~torch.triu(mel.new_ones(T,T)).transpose(0, 1).to(torch.bool)
    #     else:
    #         past_mask = torch.zeros(T,T).to(torch.bool)
    #     if num_neigbour > 0:
    #         neig_mask = np.zeros(shape=(T,T))
    #         for i in np.arange(-num_neigbour,num_neigbour+1):
    #             neig_mask += np.eye(T,T,k=i)
    #         neig_mask = ~torch.tensor(neig_mask).to(torch.bool)
    #     else:
    #         neig_mask = torch.zeros(T,T).to(torch.bool)
    #     # if training:
    #     #     diag_mask[diag_mask == 0] = -float('inf')
    #     # else:
    #     #     diag_mask[diag_mask == 0] = -1e9
    #     # diag_mask[diag_mask == 1] = 0

    #     pdb.set_trace()
    #     final_mask = past_mask | neig_mask
        
    #     return diag_mask

    # test = torch.rand((2,10,8))

    # test_2 = torch.rand((2,4,8))

    # test_fn = MultiHeadAttentionLayer(8, 4)

    # attn_mask = make_attn_mask(test)

    # pos = torch.tensor([2,3])
    # key_mask = make_key_mask(pos)


    # output, atten = test_fn(test_2, test, test, key_padding_mask=key_mask)

    # print(test_2.shape)
    # print(output.shape)
    # print(atten.shape)
    
    
    test = torch.rand((2,10,8))

    print(test)

    linear = Linear(8,10,bias=False)

    print(linear(test))
