# Conditional Generative Model for Peptide Sequence Generation
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask, self.d_k)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return nn.LayerNorm(self.d_model).to(device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(inputs.size(-1)).to(device)(output + residual)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.dec_enc_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class ConditionalDecoder(nn.Module):
    """Conditional decoder that generates sequences based on dEdge value and sequence length"""
    def __init__(self, args):
        super(ConditionalDecoder, self).__init__()
        self.args = args
        # Condition embedding: dEdge value (1) + sequence length (1) = 2 features
        self.condition_embed = nn.Linear(2, args.d_model)
        self.tgt_emb = nn.Embedding(args.src_vocab_size, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model)
        self.layers = nn.ModuleList([DecoderLayer(args.d_model, args.d_ff, args.d_k, args.d_v, args.n_heads) 
                                     for _ in range(args.n_layers)])
        # Dummy encoder output for self-attention (we use condition as encoder)
        self.condition_proj = nn.Linear(args.d_model, args.d_model)
        self.projection = nn.Linear(args.d_model, args.src_vocab_size)

    def forward(self, dec_inputs, conditions, dec_self_attn_mask, dec_enc_attn_mask):
        # conditions: [batch_size, 2] where [:, 0] is dEdge, [:, 1] is seq_length
        condition_emb = self.condition_embed(conditions).unsqueeze(1)  # [batch_size, 1, d_model]
        condition_emb = condition_emb.repeat(1, dec_inputs.size(1), 1)  # [batch_size, seq_len, d_model]
        
        # Use condition as encoder output
        enc_outputs = self.condition_proj(condition_emb)
        
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)
        
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        
        dec_logits = self.projection(dec_outputs)  # [batch_size, tgt_len, vocab_size]
        return dec_logits, dec_self_attns, dec_enc_attns

class ConditionalGenerator(nn.Module):
    """Conditional generative model for peptide sequences"""
    def __init__(self, args):
        super(ConditionalGenerator, self).__init__()
        self.args = args
        self.decoder = ConditionalDecoder(args).to(device)
        
    def forward(self, dec_inputs, conditions, dec_self_attn_mask, dec_enc_attn_mask):
        dec_logits, _, _ = self.decoder(dec_inputs, conditions, dec_self_attn_mask, dec_enc_attn_mask)
        return dec_logits
    
    def generate(self, conditions, max_len, start_token=1, temperature=1.0):
        """Generate sequences given conditions (dEdge value and sequence length)"""
        batch_size = conditions.size(0)
        device = conditions.device
        
        # Start with start token
        dec_inputs = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        generated = dec_inputs.clone()
        
        for step in range(max_len - 1):
            dec_self_attn_mask = get_attn_subsequence_mask(dec_inputs).to(device)
            dec_enc_attn_mask = torch.zeros(batch_size, dec_inputs.size(1), 1).bool().to(device)
            
            dec_logits, _, _ = self.decoder(dec_inputs, conditions, dec_self_attn_mask, dec_enc_attn_mask)
            next_token_logits = dec_logits[:, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
            
            # Stop if we hit padding token (0)
            if (next_token == 0).all():
                break
                
            generated = torch.cat([generated, next_token], dim=1)
            dec_inputs = generated
        
        return generated

