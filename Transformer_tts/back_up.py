class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout=0.1, bias=True):
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

        self.fc_o = Linear(hid_dim, hid_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)


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
        scaling = float(self.head_dim) ** -0.5
        Q = Q * scaling 
        
        energy = torch.bmm(Q, K.transpose(1,2))

        if attn_mask is not None:
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
        attention = attention.sum(dim=1)/self.n_heads
        # attention = [batch_size, n_heads, query_len, key_len]

        
        x = x.view(batch_size, self.n_heads, -1, self.head_dim)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch_size, query_len, hid_dim]

        
        x = self.fc_o(x)

        # x = [batch_size, query_len, hid_dim]
        return x, attention

class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout=0.1,
                 activation="relu"):
        '''One Encoder Layer

        Args:
          hid_dim: the dimension output from embedding for one word 
          n_heads: how many heads is chosen for multiheads attention 
          pd_dim: the hiding dimension in the positionwiseFeedforward 
          dropout: the dropout rate
          device: gpu or cpu 
        '''
        super(EncoderLayer,self).__init__()

        self.self_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout=dropout)

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

class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout=0.1,
                 activation="relu"):
        '''One Encoder Layer

        Args:
          hid_dim: the dimension output from embedding for one word 
          n_heads: how many heads is chosen for multiheads attention 
          pd_dim: the hiding dimension in the positionwiseFeedforward 
          dropout: the dropout rate
          device: gpu or cpu 
        '''
        super(DecoderLayer,self).__init__()

        self.self_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout=dropout)

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
    
