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
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return F.tanh

class Linear(nn.Linear):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(Linear, self).__init__(in_dim, out_dim, bias)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(w_init_gain))

class Conv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels,kernel_size, stride=1, padding=None, dilation=1, bias=True, w_init_gain="linear"):
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.xavier_normal_(self.weight, gain=nn.init.calculate_gain(w_init_gain))

        
class Conv1dBatchNorm(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, dilation=1, bias=True, activation="relu", dropout=0.1):
        super(Conv1dBatchNorm,self).__init__()

        self.conv = Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, w_init_gain=activation)
        self.bn = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        out = self.dropout(x)
        return out 

class PosEmbeddingLayer(nn.Module):
    def __init__(self, num_pos, hid_dim, padding_idx=0):
        super(PosEmbeddingLayer, self).__init__()
        self.register_buffer('pe', self._get_pos_matrix(num_pos, hid_dim))
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x_len = x.shape[1]
        x = self.pe[:x_len,:]*self.alpha + x
        return x


    def _get_pos_matrix(self, num_pos, hid_dim):
        pe = torch.zeros(num_pos, hid_dim)
        position = torch.arange(0, num_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, hid_dim, 2).float() / hid_dim)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        return pe


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
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

        self.fc_q = Linear(hid_dim, hid_dim)
        self.fc_k = Linear(hid_dim, hid_dim)
        self.fc_v = Linear(hid_dim, hid_dim)

        self.fc_o = Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)


    def forward(self, query, key, value, mask=None):
        '''The forward calculation of the neural network

        Args:
          query: copy of the output of the word embedding + pos embedding
          key: copy of the output of the word embedding + pos embedding
          value: copy of the output of the word embedding + pos embedding
          mask: padding is masked by 0, others are 1 
        '''

        batch_size = query.shape[0]

        # query = [batch_size, query_len, hid_dim]
        # key = [batch_size, key_len, hid_dim]
        # value = [batch_size, value_len, hid_dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch_size, query_len, hid_dim]
        # K = [batch_size, key_len, hid_dim]
        # V = [batch_size, value_len, hid_dim]

        # -1 calculate the dimension given other dimension
        Q = Q.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute([0, 2, 1, 3])
        K = K.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute([0, 2, 1, 3])
        V = V.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute([0, 2, 1, 3])

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch_size, n_heads, query_len, key_len]
        if mask is not None:
            # masked_fill(mask, value) -> Tensor
            energy = energy.masked_fill(mask == 0, -inf)
        attention = torch.softmax(energy, dim=-1)

        # attentions = [batch_size, n_heads, query_len, key_len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch_size, n_heads, query_len, head_dim]

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
            hid_dim, n_heads, dropout, device)
        
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout)

        self.dropout_1 = nn.Dropout(dropout)

        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        # src = [batch_size, src_len, hid_dim]
        # src_mask = [batch_size, src_len]

        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout_1(_src))

        # src = [batch_size, src_len, hid_dim]

        # posiionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout_2(_src))

        # src = [batch_size, src_len, hid_dim]

        return src

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
        src_, src_align = self.self_attn(src, src, src, attn_mask=src_attn_mask, key_padding_mask=src_key_padding_mask)
        src = self.ff_norm1(src + self.dropout(src_))

        src_ = self.ff_linear2(self.dropout(F.relu(self.ff_linear1(src))))
        src = self.ff_norm2(src + self.dropout(src_))

        return src, src_align


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
        self.self_attention = MultiheadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        # trg = [batch_size, trg_len, hid_dim]
        # enc_src = [batch_size, src_len, hid_dim]
        # trg_mask = [batch_size, trg_len]
        # src_mask = [batch_size, src_len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        trg = self.self_attn_layer_norm(trg + self.dropout_1(_trg))

        # trg = [batch_size, trg_len, hid_dim]

        # arg : key query value
        _trg, attention = self.encoder_attention(
            trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout_2(_trg))

        # trg = [batch_size, trg_len, hid_dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout_3(_trg))

        # trg = [batch_size, trg_len, hid_dim]
        # attention = [batch_size, n_heads, trg_len, src_len]

        return trg, attention

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
        
        
        
