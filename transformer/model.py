import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt

class Transformer(nn.Module):
    # model inspired from Vaswani et. al. 2017
    # https://arxiv.org/pdf/1706.03762.pdf
    
    def __init__(self, vocab_size, d_model=512, n_layers=6, max_length=5000):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        # the position embedding is learned, not a bunch of sin and cos
        self.pos_embedding = nn.Embedding(max_length, d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model) for i in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model) for i in range(n_layers)])

        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, encoder_in, decoder_in):
        
        encoder_pos = torch.arange(encoder_in.shape[1]).repeat(encoder_in.shape[0], 1)
        decoder_pos = torch.arange(decoder_in.shape[1]).repeat(decoder_in.shape[0], 1)
        
        encoder_in = self.embedding(encoder_in) + self.pos_embedding(encoder_pos)
        decoder_in = self.embedding(decoder_in) + self.pos_embedding(decoder_pos)
        
        for i in range(self.n_layers):
            encoder_in = self.encoder_layers[i](encoder_in)
                
        for i in range(self.n_layers):
            decoder_in = self.decoder_layers[i](decoder_in=decoder_in, 
                                                encoder_out=encoder_in)

        out = self.fc(decoder_in)
        
        return out

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model=512):
        super(EncoderLayer, self).__init__()
        
        self.d_model = d_model
        
        self.attn = MultiHeadAttn()
        self.layernorm1 = nn.LayerNorm(d_model)
        
        self.positionwiseff = PositionWiseFeedForward()
        self.layernorm2 = nn.LayerNorm(d_model)
    
    def forward(self, encoder_in):
        
        encoder_in = self.layernorm1(encoder_in + self.attn(queries=encoder_in,
                                                            keys=encoder_in, 
                                                            values=encoder_in))
        encoder_in = self.layernorm2(encoder_in + self.positionwiseff(encoder_in))
        
        return encoder_in

class DecoderLayer(nn.Module):
    
    def __init__(self, d_model=512):
        super(DecoderLayer, self).__init__()
        
        self.d_model = d_model
        
        self.masked_attn = MultiHeadAttn()
        self.layernorm1 = nn.LayerNorm(d_model)
        
        self.attn = MultiHeadAttn()
        self.layernorm2 = nn.LayerNorm(d_model)
        
        self.positionwiseff = PositionWiseFeedForward()
        self.layernorm3 = nn.LayerNorm(d_model)
    
    def forward(self, decoder_in, encoder_out):
        
        seq_len = decoder_in.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len))
        
        decoder_in = self.layernorm1(decoder_in + self.masked_attn(queries=decoder_in, 
                                                                   keys=decoder_in, 
                                                                   values=decoder_in,
                                                                   mask=mask))
        decoder_in = self.layernorm2(decoder_in + self.attn(queries=decoder_in, 
                                                            keys=encoder_out, 
                                                            values=encoder_out))
        decoder_in = self.layernorm3(decoder_in + self.positionwiseff(decoder_in))
        
        return decoder_in

class MultiHeadAttn(nn.Module):
    
    def __init__(self, n_heads=8, d_model=512):
        super(MultiHeadAttn, self).__init__()
        
        # making sure that the dimensionality doesn't change before/after attn
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
                
        self.queries_fc = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for i in range(n_heads)])
        self.keys_fc = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for i in range(n_heads)])
        self.values_fc = nn.ModuleList([nn.Linear(self.d_model, self.d_v) for i in range(n_heads)])
        
        self.attn = ScaledDotProductAttn()
        
        self.head_fc = nn.Linear(self.d_model, self.d_model )
        
    def forward(self, queries, keys, values, mask=None):
        
        # only one GPU, we use a for loop instead...
        heads_list = []
        for i in range(self.n_heads):
        
            q = self.queries_fc[i](queries)
            k = self.keys_fc[i](keys)
            v = self.values_fc[i](values)
        
            heads_list.append(self.attn(q, k, v, mask=mask))
        
        head = torch.cat(heads_list, dim=-1)
        out = self.head_fc(head)
        
        return out

class ScaledDotProductAttn(nn.Module):
    
    def __init__(self, d_k=64, d_v=64):
        super(ScaledDotProductAttn, self).__init__()
        
        self.d_k = d_k
        self.d_v = d_v
    
    def forward(self, queries, keys, values, mask=None):
        
        # matmul
        x = torch.matmul(queries, torch.transpose(keys, -2, -1))
                
        # scale
        x = x / sqrt(self.d_k)
        
        # mask
        if mask is not None:
            x = mask * x
                    
        # softmax
        x = F.softmax(x, -1)
                
        # matmul
        out = torch.matmul(x, values)
                
        return out

class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, dmodel=512, dff=2046):
        super(PositionWiseFeedForward, self).__init__()
        
        self.dmodel = dmodel
        self.dff = dff
        
        self.fc1 = nn.Linear(dmodel, dff)
        self.fc2 = nn.Linear(dff, dmodel)
        
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
