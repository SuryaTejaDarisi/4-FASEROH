"""
Full sequence-to-sequence Transformer for translating
histogram bin vectors into symbolic mathematical expressions.

Architecture:
Encoder:
  - Linear projection of histogram bins to d_model dimensions
  - Sinusoidal positional encoding
  - N stacked Transformer encoder layers (self-attention + FFN)

Decoder:
  - Token embedding for symbolic expression tokens
  - Trainable positional encoding
  - When a step's token is <CONST>, a small linear layer embeds the
    associated constant value and it is added to the token embedding
  - N stacked Transformer decoder layers (masked self-attention +
    cross-attention over encoder output + FFN)
  - Classification head: projects to vocab_size, predicts next token
  - Regression head: projects to 1, predicts the constant value when
    the predicted token is <CONST>

Loss
  L = CrossEntropy(symbol predictions, targets) + lambda_const * MSE(constant predictions, true constant values)
Only positions where the target is <CONST> contribute to the MSE term.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- Positional encodings -----
class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal encoding added to encoder inputs.
    Gives the model information about the order of histogram bins.
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1).float()          # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )                                                               # (d_model/2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as a buffer so it moves to GPU with .to(device)
        # but is not a trainable parameter
        self.register_buffer("pe", pe)                                  # (max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0)                        # x: (batch, seq_len, d_model)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embedding for decoder token positions. Unlike the fixed encoder version, 
    the decoder benefits from learning position representations alongside the token embeddings.
    """
    def __init__(self, d_model, max_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, seq_len)
        x = x + self.embedding(positions)
        return self.dropout(x)


# ----- ENCODER CLASS -----

class HistogramEncoder(nn.Module):
    """
    Encodes a normalised histogram (sequence of bin fractions) into a sequence of context vectors using a 
    Transformer encoder. The encoder doesn't collapse the sequence to a single vector, it keeps all 
    bin-level representations so the decoder can attend to different parts of the histogram via cross-attention.
    """
    def __init__(self, n_bins, d_model, n_heads, n_layers,
                 d_ff, dropout=0.1):
        super().__init__()

        # Project each bin value from 1 float to d_model floats
        self.input_projection = nn.Linear(1, d_model)

        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model, max_len=n_bins + 10, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,   # input shape: (batch, seq, d_model)
            norm_first=True,    # pre-norm is more stable than post-norm ? check !!!
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False   # suppresses warning when norm_first=True
        )

    def forward(self, histogram):
        """
        Parameters:- histogram -> (batch, n_bins) float tensor | Normalised bin fractions.

        Returns:- memory -> (batch, n_bins, d_model) float tensor | Per-bin context representations.
        """
        x = histogram.unsqueeze(-1) # Add feature dimension: (batch, n_bins) -> (batch, n_bins, 1)
        x = self.input_projection(x) # Linear projection: (batch, n_bins, 1) -> (batch, n_bins, d_model)
        x = self.pos_encoding(x) # Add positional info
        # Encode: each bin attends to all other bins
        memory = self.transformer_encoder(x)                # (batch, n_bins, d_model)
        return memory




# ----- Decoder -----
class SymbolicDecoder(nn.Module):
    """
    Autoregressively decodes the encoder's memory into a prefix-notation
    symbolic expression.

    At each decoding step:
      1. Embed the previous token
      2. If the previous token was <CONST>, add the constant's value embedding
      3. Apply learned positional encoding
      4. Run through Transformer decoder layers (with causal mask)
      5. Two output heads: symbol classification + constant regression
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers,
                 d_ff, const_id, dropout=0.1, max_seq=64):
        super().__init__()

        self.vocab_size = vocab_size
        self.const_id = const_id
        self.d_model = d_model

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Constant value embedding: maps a single float -> d_model
        self.const_embedding = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        self.pos_encoding = LearnedPositionalEncoding(
            d_model, max_len=max_seq, dropout=dropout
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )

        # Output heads
        self.classification_head = nn.Linear(d_model, vocab_size)
        self.regression_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        """Xavier-uniform init for linear layers; normal for embeddings."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _make_causal_mask(self, seq_len, device):
        """
        Create an upper-triangular causal attention mask.
        Positions that are True are blocked from attending.
        Shape: (seq_len, seq_len)
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def forward(self, decoder_input, memory, constants,
                tgt_key_padding_mask=None):
        """
        Parameters
        ----------
        decoder_input : (batch, seq_len) int64
            Token ids for the current (teacher-forced) sequence.
        memory : (batch, n_bins, d_model) float
            Encoder output.
        constants : (batch, MAX_CONSTANTS) float
            True constant values, used during training to embed <CONST>
            tokens.
        tgt_key_padding_mask : (batch, seq_len) bool, optional
            True at <PAD> positions.

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
            Symbol probability logits.
        const_preds : (batch, seq_len, 1)
            Predicted constant values at each position.
        """
        batch_size, seq_len = decoder_input.shape

        # Base token embeddings
        tok_emb = self.token_embedding(decoder_input)     # (B, S, d_model)

        # Inject constant values at positions where the token is <CONST>
        const_mask = (decoder_input == self.const_id)     # (B, S) bool
        if const_mask.any():
            # For each <CONST> position, pick the corresponding constant
            # value in order. We accumulate a running count per sample.
            const_vals = torch.zeros(
                batch_size, seq_len, 1, device=decoder_input.device
            )
            # Assign constants in order of their appearance in the sequence
            for b in range(batch_size):
                const_positions = const_mask[b].nonzero(as_tuple=True)[0]
                for j, pos in enumerate(const_positions):
                    if j < constants.shape[1]:
                        const_vals[b, pos, 0] = constants[b, j]

            const_emb = self.const_embedding(const_vals)  # (B, S, d_model)
            # Only add at actual <CONST> positions
            tok_emb = tok_emb + const_emb * const_mask.unsqueeze(-1).float()

        tok_emb = self.pos_encoding(tok_emb)

        # Causal mask prevents the decoder from peeking at future tokens
        causal_mask = self._make_causal_mask(seq_len, decoder_input.device)

        out = self.transformer_decoder(
            tgt=tok_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )                                                  # (B, S, d_model)

        logits = self.classification_head(out)             # (B, S, vocab_size)
        const_preds = self.regression_head(out)            # (B, S, 1)

        return logits, const_preds


# -----------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------

class HistoDecoder(nn.Module):
    """
    Full seq2seq model: HistogramEncoder + SymbolicDecoder.

    This is the complete model submitted for both the course project
    and the symbolic regression evaluation.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict with keys:
            n_bins, vocab_size, const_id, pad_id,
            d_model, n_heads, n_encoder_layers, n_decoder_layers,
            d_ff, dropout, max_seq
        """
        super().__init__()

        self.config = config
        self.pad_id = config["pad_id"]
        self.const_id = config["const_id"]

        self.encoder = HistogramEncoder(
            n_bins=config["n_bins"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_encoder_layers"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
        )

        self.decoder = SymbolicDecoder(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_decoder_layers"],
            d_ff=config["d_ff"],
            const_id=config["const_id"],
            dropout=config["dropout"],
            max_seq=config["max_seq"],
        )

    def forward(self, histogram, decoder_input, constants):
        """
        Full forward pass.

        Parameters
        ----------
        histogram : (batch, n_bins)
        decoder_input : (batch, seq_len)
        constants : (batch, MAX_CONSTANTS)

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        const_preds : (batch, seq_len, 1)
        """
        # Build padding mask for decoder: True at <PAD> positions
        tgt_padding_mask = (decoder_input == self.pad_id)   # (B, S)

        memory = self.encoder(histogram)
        logits, const_preds = self.decoder(
            decoder_input, memory, constants,
            tgt_key_padding_mask=tgt_padding_mask
        )
        return logits, const_preds

    def count_parameters(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------
# Loss function
# -----------------------------------------------------------------------

class SymbolicLoss(nn.Module):
    """
    Combined loss for symbolic expression generation.

      L = CE(symbol predictions, true symbols)
        + lambda_const * MSE(predicted constants, true constants)

    The CE term trains the symbol classification head.
    The MSE term trains the constant regression head, but only at
    positions where the target token is <CONST>.
    """

    def __init__(self, pad_id, const_id, lambda_const=0.5):
        super().__init__()
        self.pad_id = pad_id
        self.const_id = const_id
        self.lambda_const = lambda_const

        # Ignore <PAD> positions in the cross-entropy
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, logits, const_preds, decoder_target, true_constants):
        """
        Parameters
        ----------
        logits : (B, S, vocab_size)
        const_preds : (B, S, 1)
        decoder_target : (B, S)   true next-token ids
        true_constants : (B, MAX_CONSTANTS)  true constant values

        Returns
        -------
        total_loss : scalar
        ce_loss_val : scalar (for logging)
        mse_loss_val : scalar (for logging)
        """
        B, S, V = logits.shape

        # Cross-entropy over all non-PAD positions
        ce = self.ce_loss(
            logits.reshape(B * S, V),
            decoder_target.reshape(B * S)
        )

        # MSE only at <CONST> positions
        const_positions = (decoder_target == self.const_id)    # (B, S)
        mse = torch.tensor(0.0, device=logits.device)

        if const_positions.any():
            pred_vals = const_preds[const_positions].squeeze(-1)    # (N_consts,)

            # Gather the corresponding true values
            # For each sample, the j-th <CONST> in the sequence maps to
            # true_constants[b, j]
            true_vals_list = []
            for b in range(B):
                positions_b = const_positions[b].nonzero(as_tuple=True)[0]
                for j, _ in enumerate(positions_b):
                    if j < true_constants.shape[1]:
                        true_vals_list.append(true_constants[b, j])

            if true_vals_list:
                true_vals = torch.stack(true_vals_list).to(logits.device)
                n = min(len(pred_vals), len(true_vals))
                mse = F.mse_loss(pred_vals[:n], true_vals[:n])

        total = ce + self.lambda_const * mse
        return total, ce.detach(), mse.detach()