"""
loss.py
=======
Masked and class-weighted multi-task loss for Tox21 molecular toxicity.
Handles missing labels strictly via boolean masks and penalizes positive misclassifications
using training-set class weights.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedWeightedBCEWithLogitsLoss(nn.Module):
    """
    Masked Binary Cross Entropy with Logits Loss for Multi-Task Learning with Missing Labels.

    For each compound i and endpoint j:
      loss_ij = BCEWithLogits(logits_ij, y_ij, pos_weight=w_j) * mask_ij

    Total loss:
      L = sum(loss_ij) / sum(mask_ij)

    Parameters
    ----------
    pos_weight : torch.Tensor of shape (num_endpoints,)
        Positive class weights w_j = N_neg / N_pos calculated strictly on train set.
    eps : float, default 1e-8
        Numerical stability epsilon to prevent zero-division when batch mask has 0 observed.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scalar masked loss across the batch.

        Parameters
        ----------
        logits : torch.Tensor of shape (B, num_endpoints)
            Raw model outputs before sigmoid.
        targets : torch.Tensor of shape (B, num_endpoints)
            Ground truth labels (0.0 or 1.0).
        mask : torch.Tensor of shape (B, num_endpoints)
            Boolean or 0/1 observation mask (1=observed, 0=unobserved/NaN).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        mask_f = mask.float()

        # Compute unreduced element-wise loss
        if self.pos_weight is not None:
            pos_w = self.pos_weight.to(logits.device)
            unreduced = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_w, reduction="none"
            )
        else:
            unreduced = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )

        # Apply observation mask
        masked_loss = unreduced * mask_f

        # Normalize strictly by the count of observed labels
        total_observed = mask_f.sum()
        if total_observed > 0:
            return masked_loss.sum() / total_observed
        else:
            return masked_loss.sum() * 0.0

    def compute_per_endpoint_losses(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-endpoint average loss for logging and diagnostic purposes.

        Returns
        -------
        torch.Tensor of shape (num_endpoints,)
        """
        mask_f = mask.float()
        if self.pos_weight is not None:
            pos_w = self.pos_weight.to(logits.device)
            unreduced = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_w, reduction="none"
            )
        else:
            unreduced = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )

        masked_loss = unreduced * mask_f
        obs_per_endpoint = mask_f.sum(dim=0).clamp(min=1.0)
        return masked_loss.sum(dim=0) / obs_per_endpoint
