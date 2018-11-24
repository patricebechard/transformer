import torch
from torch import nn
import torch.nn.functional as F

class Transformer(nn.Module):
    # model inspired from Vaswani et. al. 2017
    # https://arxiv.org/pdf/1706.03762.pdf
    
    def __init__(self, vocab_size, d_model=512, n_layers=6):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionEncoding()
        
#         self.encoder_layers = nn.ModuleList(
#             [EncoderLayer(d_model=d_model) for i in range(n_layers)])
#         self.decoder_layers = nn.ModuleList(
#             [DecoderLayer(d_model=d_model) for i in range(n_layers)])

        # easy way to share layers ?
        self.encoder_layer = EncoderLayer(d_model=d_model)
        self.decoder_layer = DecoderLayer(d_model=d_model)

        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, y):
        
#         x = self.embedding(x)
        x = self.pos_encoding(x)
                
#         y = self.embedding(y)
        y = self.pos_encoding(y)
        
        for i in range(self.n_layers):
            x = self.encoder_layer(x)
            y = self.decoder_layer(y, x)

        out = self.fc(y)
        
        return out

class PositionEncoding(nn.Module):
    
    def __init__(self, d_model=512, max_len=5000):
        super(PositionEncoding, self).__init__()
        
        # define empty array for position encoding
        # taken from annotated Transformer from HarvardNLP
        # http://nlp.seas.harvard.edu/2018/04/03/attention.html
        self.position_encoding = torch.zeros(max_len, d_model)
        numerator = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                         (np.log(10000.0) / d_model))
        
        self.position_encoding[:, 0::2] = torch.sin(numerator / denominator)
        self.position_encoding[:, 1::2] = torch.cos(numerator / denominator)
        
        self.position_encoding = self.position_encoding.unsqueeze(0)
        
    def forward(self, x):    
        return x + self.position_encoding[:, :x.shape[1]]


class EncoderLayer(nn.Module):
    
    def __init__(self, d_model=512):
        super(EncoderLayer, self).__init__()
        
        self.d_model = d_model
        
        self.attn = MultiHeadAttn()
        self.layernorm1 = nn.LayerNorm(d_model)
        
        self.positionwiseff = PositionWiseFeedForward()
        self.layernorm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        
        x = self.layernorm1(x + self.attn(x))
        x = self.layernorm2(x + self.positionwiseff(x))
        
        return x

class DecoderLayer(nn.Module):
    
    def __init__(self, d_model=512):
        super(DecoderLayer, self).__init__()
        
        self.d_model = d_model
        
        self.masked_attn = MultiHeadAttn(has_mask=True)
        self.layernorm1 = nn.LayerNorm(d_model)
        
        self.attn = MultiHeadAttn(from_encoder=True)
        self.layernorm2 = nn.LayerNorm(d_model)
        
        self.positionwiseff = PositionWiseFeedForward()
        self.layernorm3 = nn.LayerNorm(d_model)
    
    def forward(self, x, y):
        
        x = self.layernorm1(torch.add(x, self.masked_attn(x)))
        x = self.layernorm2(torch.add(x, self.attn(x, y)))
        x = self.layernorm3(torch.add(x, self.positionwiseff(x)))
        
        return x

class MultiHeadAttn(nn.Module):
    
    def __init__(self, n_heads=1, has_mask=False, from_encoder=False,
                 d_model=512, d_k=64, d_v=64):
        super(MultiHeadAttn, self).__init__()
        
        self.n_heads = n_heads
        self.has_mask = has_mask
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.from_encoder = from_encoder
                
        self.queries_fc = nn.Linear(self.d_model, self.d_k)
        self.keys_fc = nn.Linear(self.d_model, self.d_k)
        self.values_fc = nn.Linear(self.d_model, self.d_v)
        
        self.attn = ScaledDotProductAttn()
        
        self.head_fc = nn.Linear(self.d_v, self.d_model )
        
    def forward(self, x, y=None):
        
        if self.from_encoder:
            queries = self.queries_fc(y)
            keys = self.keys_fc(y)
        else:
            queries = self.queries_fc(x)
            keys = self.keys_fc(x)
        values = self.values_fc(x)
        
        head = self.attn(queries, keys, values)
        
#         head = torch.cat(head)

        out = self.head_fc(head)
        
        return out

class ScaledDotProductAttn(nn.Module):
    
    def __init__(self, has_mask=False):
        super(ScaledDotProductAttn, self).__init__()
        
        self.has_mask = has_mask
        if self.has_mask:
            self.mask = torch.ones(1)
    
    def forward(self, queries, keys, values):
        
        x = torch.matmul(torch.t(queries), keys)
        x = torch.div(x, np.sqrt(keys.shape[-1]))
        if self.has_mask:
            x = self.mask * x
        x = F.softmax(x, -1)
        out = torch.matmul(x, torch.t(values))
        
        return torch.t(out)

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

if __name__ == "__main__":

    data = torch.ones()
