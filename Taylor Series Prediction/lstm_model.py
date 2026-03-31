"""
Encoder-Decoder LSTM for the Taylor expansion

Architecture:
Encoder:
  - Embedding layer: token IDs -> dense vectors
  - Bidirectional LSTM: reads the full input sequence and produces
    a fixed-size context (final hidden + cell states, both directions
    concatenated then projected down to hidden_size)

Decoder:
  - Embedding layer (shared weights with encoder by default)
  - Unidirectional LSTM: generates the target sequence token by token
  - Linear classification head: hidden state -> vocabulary logits

Training uses teacher forcing (the true previous token is fed at eachdecoder step).
Inference uses greedy decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLSTM(nn.Module): # Bidirectional LSTM encoder
    def __init__(self, vocab_size, embed_dim, hidden_size,
                 n_layers=2, dropout=0.3, pad_id=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        # Projectiing  bidirectional output (2 * hidden_size) back to hidden_size
        self.proj_h = nn.Linear(2 * hidden_size, hidden_size)
        self.proj_c = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, src):
        """
        src : (batch, src_len)

        Returns:
        hidden : (n_layers, batch, hidden_size)
        cell   : (n_layers, batch, hidden_size)
        """
        emb = self.embedding(src)                            # (B, S, E)
        _, (h, c) = self.lstm(emb)                           # h: (2*L, B, H)

        # Shape of h: (2*n_layers, batch, hidden_size) -> (n_layers, batch, 2*hidden_size) (for projecting)
        B = src.size(0)
        h = h.view(self.n_layers, 2, B, self.hidden_size)   # (L, 2, B, H)
        c = c.view(self.n_layers, 2, B, self.hidden_size)
        # Concatenate forward and backward directions
        h = torch.cat([h[:, 0], h[:, 1]], dim=-1)            # (L, B, 2H)
        c = torch.cat([c[:, 0], c[:, 1]], dim=-1)
        # Project back to hidden_size for the decoder
        h = torch.tanh(self.proj_h(h))                       # (L, B, H)
        c = torch.tanh(self.proj_c(c))

        return h, c



class DecoderLSTM(nn.Module): #Unidirectional LSTM decoder with classification head
    def __init__(self, vocab_size, embed_dim, hidden_size,
                 n_layers=2, dropout=0.3, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout  = nn.Dropout(dropout)
        self.fc_out   = nn.Linear(hidden_size, vocab_size)

    def forward_step(self, token_id, hidden, cell): # Single decoder's step
        """
        token_id : (batch,)
        hidden   : (n_layers, batch, hidden_size)
        cell     : (n_layers, batch, hidden_size)

        Returns:
        logits : (batch, vocab_size)
        hidden, cell: updated states
        """
        emb = self.embedding(token_id.unsqueeze(1))       # (B, 1, E)
        out, (hidden, cell) = self.lstm(emb, (hidden, cell))
        out = self.dropout(out.squeeze(1))                 # (B, H)
        logits = self.fc_out(out)                          # (B, V)
        return logits, hidden, cell


class Seq2SeqLSTM(nn.Module):
    """
    Complete encoder-decoder LSTM for Taylor expansion

    Both encoder and decoder share the same embedding matrix to reduce
    parameters and encourage consistent token representations.
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_size=256,
                 n_layers=2, dropout=0.3, pad_id=0):
        super().__init__()
        self.pad_id = pad_id

        self.encoder = EncoderLSTM(vocab_size, embed_dim, hidden_size, n_layers, dropout, pad_id)
        self.decoder = DecoderLSTM(vocab_size, embed_dim, hidden_size, n_layers, dropout, pad_id)
        self.decoder.embedding.weight = self.encoder.embedding.weight # Shared embedding weights between encoder and decoder

    def forward(self, src, tgt):
        """
        Teacher-forced forward pass.

        Parameters:
        src : (batch, src_len) int64
        tgt : (batch, tgt_len) int64   includes BOS and excludes EOS

        Returns logits : (batch, tgt_len, vocab_size)
        """
        hidden, cell = self.encoder(src)
        tgt_len = tgt.size(1)

        all_logits = []
        for t in range(tgt_len):
            logits, hidden, cell = self.decoder.forward_step(
                tgt[:, t], hidden, cell
            )
            all_logits.append(logits.unsqueeze(1))        # (B, 1, V)

        return torch.cat(all_logits, dim=1)               # (B, T, V)

    @torch.no_grad()
    def greedy_decode(self, src, bos_id, eos_id, max_len=60): # For Inference
        """
        src    : (batch, src_len)
        bos_id, eos_id, max_len : int

        Returns predictions : list of list of int  (one per batch item)
        """
        self.eval()
        hidden, cell = self.encoder(src)
        B = src.size(0)
        device = src.device

        current_token = torch.full((B,), bos_id, dtype=torch.long, device=device)
        predictions = [[] for _ in range(B)]
        finished    = [False] * B

        for _ in range(max_len):
            logits, hidden, cell = self.decoder.forward_step(current_token, hidden, cell)
            current_token = logits.argmax(dim=-1)       # (B,)

            for i in range(B):
                if finished[i]:
                    continue
                tok = current_token[i].item()
                if tok == eos_id:
                    finished[i] = True
                else:
                    predictions[i].append(tok)

            if all(finished):
                break
        return predictions