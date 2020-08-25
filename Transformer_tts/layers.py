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
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding=0, dilation=1, bias=True, activation="linear", dropout=0.1):
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


class MultiheadAtten(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super(MultiheadAtten, self).__init__()

        assert embed_dim % num_heads == 0

        self.dropout = dropout 

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.use_separate_weight = use_separate_weight

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        self.out_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.out_proj_bias = nn.Parameter(torch.empty(embed_dim))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        
        self._reset_parameters()

    def _reset_parameters(self):

        nn.init.xavier_uniform_(self.in_proj_weight)
            
        nn.init.kaiming_uniform_(self.out_proj_weight, a=sqrt(5)) # better than xavier_uniform_
        
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias,0.)
            nn.init.constant_(self.out_proj_bias,0.)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
            
        tgt_len, bsz, embed_dim = query.size()
            
        scaling = float(self.head_dim) ** -0.5

        if torch.equal(query, key) and torch.equal(key, value):
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            _b = in_proj_bias
            _start = 0
            _end = self.embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b_1 = _b[_start:_end, :]
            q = F.linear(query, _w, _b_1)

            _w = in_proj_weight[_end:,:]
            if _b is not None:
                _b_2 = _b[_start:]
            k, v = F.linear(key, _w, _b_2).chunk(2, dim=-1)

        q = q * scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0,1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1,2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

            
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn_output = F.linear(attn_output, self.out_proj_weight, self.out_proj_bias)
            
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            
        return attn_output, attn_output_weights
            
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim=2048,
                 dropout=0.1,
                 activation="relu"):

        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout)
        # self.self_attn = MultiheadAtten(hid_dim, n_heads, dropout=dropout)
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
    def __init__(self, hid_dim, n_heads, pf_dim=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout)
        # self.self_attn = MultiheadAtten(hid_dim, n_heads, dropout=dropout)
        # self.cross_attn = MultiheadAtten(hid_dim, n_heads, dropout=dropout)

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

    def make_key_mask(pos):
        # true will be -inf
        # false will be same
        max_len = torch.max(pos).item()
        ids = (torch.arange(0, max_len))
        mask = (pos.unsqueeze(1) <= ids).to(torch.bool)
        return mask

    def make_attn_mask(mel, mask_future=True, num_neigbour=0):
        # true will be -inf
        # false will be same
        T = mel.size(1)
        
        if mask_future:
            past_mask = ~torch.triu(mel.new_ones(T,T)).transpose(0, 1).to(torch.bool)
        else:
            past_mask = torch.zeros(T,T).to(torch.bool)
        if num_neigbour > 0:
            neig_mask = np.zeros(shape=(T,T))
            for i in np.arange(-num_neigbour,num_neigbour+1):
                neig_mask += np.eye(T,T,k=i)
            neig_mask = ~torch.tensor(neig_mask).to(torch.bool)
        else:
            neig_mask = torch.zeros(T,T).to(torch.bool)
        # if training:
        #     diag_mask[diag_mask == 0] = -float('inf')
        # else:
        #     diag_mask[diag_mask == 0] = -1e9
        # diag_mask[diag_mask == 1] = 0

        pdb.set_trace()
        final_mask = past_mask | neig_mask
        
        return diag_mask

    test = torch.rand((2,10,8))

    test_2 = torch.rand((2,4,8))

    test_fn = MultiHeadAttentionLayer(8, 4)

    attn_mask = make_attn_mask(test)

    pos = torch.tensor([2,3])
    key_mask = make_key_mask(pos)


    output, atten = test_fn(test_2, test, test, key_padding_mask=key_mask)

    print(test_2.shape)
    print(output.shape)
    print(atten.shape)
    
    
    # test = torch.rand((2,10,8))

    # print(test)

    # linear = Linear(8,10,bias=False)

    # print(linear(test))
