"""
Encoder-Decoder Transformer for the Taylor expansion Seequence Prediction.
A standard token-to-token seq2seq Transformer

Architecture
Encoder:
  - Token embedding + sinusoidal positional encoding
  - N standard Transformer encoder layers (self-attention + FFN)

Decoder:
  - Token embedding + sinusoidal positional encoding
  - N Transformer decoder layers (masked self-attn + cross-attn + FFN)
  - Linear classification head -> vocabulary logits

Both encoder and decoder share the same embedding matrix.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):     # x: (batch, seq_len, d_model)
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module): # Standard encoder-decoder Transformer for symbolic seq2seq tasks.
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_enc_layers=3, n_dec_layers=3, d_ff=256,
                 dropout=0.1, pad_id=0, max_len=256):
        super().__init__()
        self.pad_id  = pad_id
        self.d_model = d_model

        # Shared embedding for both encoder and decoder
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc   = SinusoidalPE(d_model, max_len, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers, enable_nested_tensor=False)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def _causal_mask(self, size, device):
        """Upper-triangular mask to prevent attending to future tokens."""
        return torch.triu(
            torch.ones(size, size, device=device), diagonal=1
        ).bool()

    def encode(self, src):
        """
        Encode source sequence of (batch, src_len) int
        Returns memory of (batch, src_len, d_model)
        """
        src_key_padding_mask = (src == self.pad_id)
        emb = self.pos_enc(self.embedding(src) * math.sqrt(self.d_model))
        return self.encoder(emb, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, src):
        """
        Decode one step (or full sequence during training)

        Parameters
        tgt    : (batch, tgt_len) int
        memory : (batch, src_len, d_model)
        src    : (batch, src_len) int

        Returns logits of (batch, tgt_len, vocab_size)
        """
        tgt_len = tgt.size(1)
        causal_mask         = self._causal_mask(tgt_len, tgt.device)
        tgt_key_padding_mask = (tgt == self.pad_id)
        mem_key_padding_mask = (src == self.pad_id)

        emb = self.pos_enc(self.embedding(tgt) * math.sqrt(self.d_model))
        out = self.decoder(
            tgt=emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=mem_key_padding_mask,
        )
        return self.fc_out(out)                              # (B, T, V)

    def forward(self, src, tgt):
        """Full teacher-forced forward pass."""
        memory = self.encode(src)
        return self.decode(tgt, memory, src)

    @torch.no_grad()
    def greedy_decode(self, src, bos_id, eos_id, max_len=60):
        """
        Greedy decoding for inference.
        Returns predictions : list of list of int
        """
        self.eval()
        memory = self.encode(src)
        B = src.size(0)
        device = src.device

        dec_input = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        predictions = [[] for _ in range(B)]
        finished    = [False] * B

        for _ in range(max_len):
            logits = self.decode(dec_input, memory, src)    # (B, t, V)
            next_token = logits[:, -1, :].argmax(dim=-1)    # (B,)

            for i in range(B):
                if finished[i]:
                    continue
                tok = next_token[i].item()
                if tok == eos_id:
                    finished[i] = True
                else:
                    predictions[i].append(tok)

            if all(finished):
                break
            dec_input = torch.cat([dec_input, next_token.unsqueeze(1)], dim=1)
        return predictions
