"""
ann_transformer.py
==================
Experiment 2: one network with two branches that are trained together.

  fingerprint (2053 numbers)  ->  ANN branch        -> 128 numbers  \
                                                                     concatenate -> small network -> 12 logits
  SMILES text  -> tokens      ->  Transformer branch -> 64 numbers  /

This is a FEATURE-LEVEL hybrid, not an average of two separate models. Both
branches are updated by the same loss in the same backward pass, so the ANN
branch can learn to rely on the Transformer branch and the other way round.

The ANN branch is the existing MultiTaskToxicNet from Member 2. We reuse it
instead of writing a second copy, and we call extract_latent() to take its
128-dim bottleneck rather than its own 12 outputs.

How the Transformer branch works, in plain English:
  1. Each token gets a learned vector (nn.Embedding).
  2. Each position gets a learned vector too, so the model knows token order.
  3. Self-attention lets every token look at every other token in the molecule
     and decide which ones matter. Two encoder layers do this twice.
  4. We average the token vectors (ignoring padding) into one vector for the
     whole molecule.
"""

import torch
import torch.nn as nn

from phase2.deep_learning.src.model import MultiTaskToxicNet


class AnnTransformerHybrid(nn.Module):
    """
    ANN branch + Transformer branch + a fusion head, trained end to end.

    use_fingerprints / use_smiles let us switch a branch off. That gives us
    honest ablations: the same training code produces the ANN-only model, the
    Transformer-only model and the hybrid, so any difference comes from the
    architecture and not from a different training script.
    """

    def __init__(
        self,
        vocab_size,
        max_length=128,
        fingerprint_dim=2053,
        ann_latent_dim=128,
        d_model=64,
        num_heads=4,
        num_layers=2,
        num_endpoints=12,
        dropout=0.35,
        use_fingerprints=True,
        use_smiles=True,
    ):
        super().__init__()
        if not use_fingerprints and not use_smiles:
            raise ValueError("At least one branch must be switched on.")

        self.use_fingerprints = use_fingerprints
        self.use_smiles = use_smiles
        self.max_length = max_length

        fused_dim = 0

        if use_fingerprints:
            # Member 2's network, reused. We only use its trunk + latent layer.
            self.ann = MultiTaskToxicNet(
                input_dim=fingerprint_dim,
                hidden_dims=[512, 256],
                latent_dim=ann_latent_dim,
                num_endpoints=num_endpoints,
                dropout=dropout,
            )
            fused_dim += ann_latent_dim

        if use_smiles:
            self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.position_embedding = nn.Embedding(max_length, d_model)
            self.embedding_dropout = nn.Dropout(dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers, enable_nested_tensor=False
            )
            self.transformer_norm = nn.LayerNorm(d_model)
            fused_dim += d_model

        # The fusion head. Small on purpose: the branches do the real work.
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_endpoints),
        )

    def encode_smiles(self, tokens):
        """Turns a batch of token id sequences into one vector per molecule."""
        batch_size, length = tokens.shape
        positions = torch.arange(length, device=tokens.device).unsqueeze(0).expand(batch_size, length)

        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)

        # True where the position is padding, so attention skips it.
        padding_mask = tokens == 0
        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)
        hidden = self.transformer_norm(hidden)

        # Average over real tokens only. Padding must not drag the average
        # towards zero, so we divide by the real token count, not the length.
        keep = (~padding_mask).unsqueeze(-1).float()
        summed = (hidden * keep).sum(dim=1)
        counts = keep.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(self, fingerprints, tokens):
        parts = []
        if self.use_fingerprints:
            parts.append(self.ann.extract_latent(fingerprints))
        if self.use_smiles:
            parts.append(self.encode_smiles(tokens))

        fused = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.fusion(fused)
