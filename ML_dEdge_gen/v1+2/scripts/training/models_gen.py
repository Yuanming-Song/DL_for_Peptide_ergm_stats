import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L]


class ConditionalGenerator(nn.Module):
    """
    Proper encoder + decoder transformer for conditional generation.
    Condition (dEdge_norm, seq_length) is encoded by the encoder.
    Decoder autoregressively generates sequence using cross-attention.
    """

    def __init__(self, args):
        super().__init__()
        self.vocab_size = args.src_vocab_size
        self.d_model = args.d_model

        # Embeddings
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = PositionalEncoding(self.d_model)

        # Encode 2-dimensional condition into memory token
        self.cond_input = nn.Linear(2, self.d_model)

        # Encoder: takes condition token only
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=args.n_heads,
            dim_feedforward=args.d_ff,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="relu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

        # Decoder: autoregressive next-token generator
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=args.n_heads,
            dim_feedforward=args.d_ff,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="relu"
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=args.n_layers)

        # Final projection
        self.proj = nn.Linear(self.d_model, self.vocab_size)

        self.to(device)

    def _causal_mask(self, L, device):
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
        mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, 0.0)
        return mask

    def forward(self, dec_inputs, conditions, *unused):
        """
        dec_inputs: [B, L]
        conditions: [B, 2]
        """
        B, L = dec_inputs.shape

        # Encode condition → memory vector
        cond_token = self.cond_input(conditions).unsqueeze(1)
        memory = self.encoder(cond_token)   # [B, 1, d_model]

        # Decode tokens
        x = self.tok_emb(dec_inputs)
        x = self.pos_emb(x)

        tgt_mask = self._causal_mask(L, x.device)
        tgt_key_padding = dec_inputs.eq(0)  # PAD mask

        dec_out = self.decoder(
            x, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding
        )
        logits = self.proj(dec_out)
        return logits

    @torch.no_grad()
    def generate(self, conditions, max_len, start_token=1, temperature=1.0):
        """
        Autoregressive generation with sampling temperature.
        """
        self.eval()
        B = conditions.size(0)
        conditions = conditions.to(device)

        # Encode condition
        cond_token = self.cond_input(conditions).unsqueeze(1)
        memory = self.encoder(cond_token)

        # Start sequence
        generated = torch.full((B, 1), start_token, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            L = generated.size(1)
            x = self.tok_emb(generated)
            x = self.pos_emb(x)

            tgt_mask = self._causal_mask(L, x.device)
            tgt_key_padding = generated.eq(0)

            dec_out = self.decoder(
                x, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding
            )
            logits = self.proj(dec_out[:, -1, :]) / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_tok], dim=1)

        return generated