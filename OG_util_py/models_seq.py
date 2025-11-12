import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len,1], pos向量
        # div_term [d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 10000^{2i/d_model}
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len,1,d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: [seq_len, batch_size, d_model]
        :return:
        '''
        x = x + self.pe[:x.size(0), :] 
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    :param seq_q: [batch_size, seq_len]
    :param seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    :return:
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    #eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask,args):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(args.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        self.W_Q = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_K = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_V = nn.Linear(args.d_model, args.d_v * args.n_heads, bias=False)
        self.fc = nn.Linear(args.n_heads * args.d_v, args.d_model, bias=False)

    def forward(self,input_Q, input_K, input_V, attn_mask):
        '''
        :param input_Q: [batch_size, len_q, d_model]
        :param input_K: [batch_size, len_k, d_model]
        :param input_V: [batch_size, len_v(=len_k), d_model]
        :param attn_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B,S,D) - proj -> (B,S,D_new) -split -> (B, S, H, W) -> trans -> (B,H,S,W)

        # 分解为MultiHead Attention
        Q = self.W_Q(input_Q).view(batch_size,-1, self.args.n_heads, self.args.d_k).transpose(1,2) # Q:[batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size,-1, self.args.n_heads, self.args.d_k).transpose(1,2) # K:[batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size,-1, self.args.n_heads, self.args.d_v).transpose(1,2) # V:[batch_size, n_heads, len_v(=len_k, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.n_heads, 1, 1) # attn_mask: [batch_size,n_heads, seq_len, seq_len]

        # [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q,K,V, attn_mask,self.args)
        context = context.transpose(1,2).reshape(batch_size, -1, self.args.n_heads * self.args.d_v)
        # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)

        return nn.LayerNorm(self.args.d_model).to(device)(output+residual),attn # Layer Normalization

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.args = args
        self.fc = nn.Sequential(
            nn.Linear(args.d_model,args.d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(args.d_ff, args.d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        :param inputs: [batch_size, seq_len, d_model]
        :return:
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.args.d_model).to(device)(output+residual) #[batch_size, seq_len, d_model]



class EncoderLayer(nn.Module):
    def __init__(self,args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self,enc_inputs, enc_self_attn_mask):
        '''
        :param enc_inputs: [batch_size, src_len, d_model]
        :param enc_self_attn_mask: [batch_size, src_len, src_len]
        :return:
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(args.src_vocab_size, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model)
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])

    def forward(self, enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :return:
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1) # [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Classifier(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        
        # More gradual decrease in layer sizes
        hidden_layer = [512, 384, 256, 128, args.dist_dim]  # More gradual size reduction
        
        self.e1 = nn.Linear(input_dim, hidden_layer[0])
        self.e2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.e3 = nn.Linear(hidden_layer[1], hidden_layer[2])
        self.e4 = nn.Linear(hidden_layer[2], hidden_layer[3])
        self.e5 = nn.Linear(hidden_layer[3], hidden_layer[4])

        # Projection layers for residual connections
        self.proj1 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.proj2 = nn.Linear(hidden_layer[1], hidden_layer[2])
        self.proj3 = nn.Linear(hidden_layer[2], hidden_layer[3])

        # Layer Normalization instead of Batch Normalization
        self.norm1 = nn.LayerNorm(hidden_layer[0])
        self.norm2 = nn.LayerNorm(hidden_layer[1])
        self.norm3 = nn.LayerNorm(hidden_layer[2])
        self.norm4 = nn.LayerNorm(hidden_layer[3])

        # Different dropout rates for different layers
        self.dropout1 = nn.Dropout(p=0.1)  # Less dropout in earlier layers
        self.dropout2 = nn.Dropout(p=0.15)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)

    def forward(self, dec_input):
        # First block
        h_1_pre = self.e1(dec_input)
        h_1 = self.norm1(h_1_pre)
        h_1 = F.gelu(h_1)  # GELU instead of LeakyReLU
        h_1 = self.dropout1(h_1)

        # Second block with residual
        h_2_pre = self.e2(h_1)
        h_2 = self.norm2(h_2_pre)
        h_2 = F.gelu(h_2)
        h_2 = self.dropout2(h_2)
        h_2 = h_2 + self.proj1(h_1)  # Residual connection
        
        # Third block with residual
        h_3_pre = self.e3(h_2)
        h_3 = self.norm3(h_3_pre)
        h_3 = F.gelu(h_3)
        h_3 = self.dropout3(h_3)
        h_3 = h_3 + self.proj2(h_2)  # Residual connection
        
        # Fourth block with residual
        h_4_pre = self.e4(h_3)
        h_4 = self.norm4(h_4_pre)
        h_4 = F.gelu(h_4)
        h_4 = self.dropout4(h_4)
        h_4 = h_4 + self.proj3(h_3)  # Residual connection

        # Output layer
        y = self.e5(h_4)

        return y

class Transformer(nn.Module):
    def __init__(self,args):
        super(Transformer,self).__init__()
        self.encoder = Encoder(args).to(device)
        self.decoder = Classifier(args.src_len*args.d_model,args).to(device)

    def forward(self,enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :param dec_inputs: [batch_size, tgt_len]
        :return:
        '''
        enc_outputs,_ = self.encoder(enc_inputs)
        # dec_inputs = enc_outputs[:,0,:].squeeze(1)
        dec_inputs = torch.reshape(enc_outputs,(enc_outputs.shape[0],-1))
        pred = self.decoder(dec_inputs)

        return pred

class LSTM(nn.Module):
    """
        Parameters
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, args, num_layers=1):
        super().__init__()
        self.args = args
        if args.task_type == 'Regression':
            last_layer = 1
        elif args.task_type == 'Classification':
            last_layer = 2
        self.src_emb = nn.Embedding(args.src_vocab_size, args.d_model)
        self.lstm = nn.LSTM(args.d_model, args.d_model, num_layers, batch_first=True) # utilize the LSTM model in torch.nn 
        self.forwardCalculation = nn.Linear(args.src_len*args.d_model, last_layer)

    def forward(self, inputs):
        inputs = self.src_emb(inputs)
        x, _ = self.lstm(inputs)  # _x is input, size (seq_len, batch, input_size)
        b, s, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.reshape(b, -1)
        x = self.forwardCalculation(x)
        
        if self.args.task_type == 'Classification':
            return F.log_softmax(x, dim=1)
        elif self.args.task_type == 'Regression':
            return x

class BidirectionalLSTM(nn.Module):
    """
        Parameters�?
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, args, num_layers=1):
        super().__init__()
        self.args = args
        if args.task_type == 'Regression':
            last_layer = 1
        elif args.task_type == 'Classification':
            last_layer = 2
        self.src_emb = nn.Embedding(args.src_vocab_size, args.d_model)
        self.lstm = nn.LSTM(args.d_model, args.d_model, num_layers,bidirectional= True, batch_first=True) # utilize the LSTM model in torch.nn 
        self.forwardCalculation = nn.Linear(2*args.src_len*args.d_model, last_layer)

    def forward(self, inputs):
        inputs = self.src_emb(inputs)
        x, _ = self.lstm(inputs)  # _x is input, size (seq_len, batch, input_size)
        b, s, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.reshape(b, -1)
        x = self.forwardCalculation(x)
        
        if self.args.task_type == 'Classification':
            return F.log_softmax(x, dim=1)
        elif self.args.task_type == 'Regression':
            return x

class RNN(nn.Module):
    """
        Parameters
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, args, num_layers=1):
        super().__init__()
        self.args = args
        if args.task_type == 'Regression':
            last_layer = 1
        elif args.task_type == 'Classification':
            last_layer = 2
        self.src_emb = nn.Embedding(args.src_vocab_size, args.d_model)
        self.lstm = nn.RNN(args.d_model, args.d_model, num_layers, batch_first=True)
        self.forwardCalculation = nn.Linear(args.src_len*args.d_model, last_layer)

    def forward(self, inputs):
        inputs = self.src_emb(inputs)
        x, _ = self.lstm(inputs)  # _x is input, size (seq_len, batch, input_size)
        b, s, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.reshape(b, -1)
        x = self.forwardCalculation(x)
        
        if self.args.task_type == 'Classification':
            return F.log_softmax(x, dim=1)
        elif self.args.task_type == 'Regression':
            return x