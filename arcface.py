from __future__ import annotations
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize(t: torch.Tensor, dim: int = 1, eps: float = 1e-5) -> torch.Tensor:
    return F.normalize(t, p=2, dim=dim, eps=eps)


class SimpleArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin, L4 from the paper).
    Computes logits and CE loss with a single center per class.

    Args:
        num_classes: number of classes.
        embedding_dim: feature dimension
        s: scale factor applied to logits.
        m: angular margin in radians.
        easy_margin: use easy margin as in ArcFace.
        reduction: reduction for CE loss.
        label_smoothing: optional label smoothing epsilon.

    Forward:
        inputs:
            embeddings: tensor of shape [B, embedding_dim]
            labels: tensor of shape [B] with class indices (long)
        returns:
            (loss, logits)
    """
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        s: float = 64.0,
        m: float = 0.5,
        easy_margin: bool = False,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        assert num_classes > 1 and embedding_dim > 0
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = bool(easy_margin)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

        # Class centers (weights)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 2 or embeddings.size(1) != self.embedding_dim:
            raise ValueError(f"embeddings must be [B, {self.embedding_dim}]")
        if labels.ndim != 1 or labels.size(0) != embeddings.size(0):
            raise ValueError("labels must be [B] and match batch size")

        x = _normalize(embeddings, dim=1)
        W = _normalize(self.weight, dim=1)

        # Cosine similarity logits
        cos = F.linear(x, W)  # [B, C]
        cos = cos.clamp(-1.0, 1.0)
        
        # Angular margin transform for target class
        # Same thing as phi = torch.cos(torch.acos(cos).clamp(theta, math.pi) + self.m) but arccos is SLOW so replaced with 
        # cos(theta + m) = cos theta · cos m − sin theta · sin m
        sin = torch.sqrt(torch.clamp(1.0 - cos * cos, min=0.0))
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        phi = cos * cos_m - sin * sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cos > 0, phi, cos)
        else:
            th = math.cos(math.pi - self.m)
            mm = math.sin(math.pi - self.m) * self.m
            phi = torch.where(cos > th, phi, cos - mm)

        # Replace target-class logits with phi
        logits = cos.clone()
        logits.scatter_(1, labels.view(-1, 1), phi.gather(1, labels.view(-1, 1)))

        logits = logits * self.s

        # Cross-entropy with optional label smoothing
        if self.label_smoothing > 0.0:
            # Smooth targets: (1 - eps) on true class, eps/C on others
            eps = self.label_smoothing
            with torch.no_grad():
                true_dist = torch.full_like(logits, fill_value=eps / self.num_classes)
                true_dist.scatter_(1, labels.view(-1, 1), 1.0 - eps + (eps / self.num_classes))
            log_probs = F.log_softmax(logits, dim=1)
            loss = (-true_dist * log_probs).sum(dim=1)
            loss = loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss
        else:
            loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss, logits


class SubCenterArcFaceLoss(nn.Module):
    """
    Sub-center ArcFace (L7).
    Each class has K sub-centers. For each class we take the maximum similarity
    over its K sub-centers, then apply ArcFace margin to the target class logit.

    Args:
        num_classes: number of classes.
        embedding_dim: feature dimension.
        k: number of sub-centers per class (>1).
        s: scale factor applied to logits.
        m: angular margin in radians.
        easy_margin: use easy margin variant.
        reduction: reduction for CE loss.
        label_smoothing: optional label smoothing epsilon.

    Forward:
        inputs:
            embeddings: [B, embedding_dim]
            labels: [B]
        returns:
            (loss, logits) where logits are [B, num_classes]
    """
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        k: int = 3,
        s: float = 64.0,
        m: float = 0.5,
        easy_margin: bool = False,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        assert num_classes > 1 and embedding_dim > 0
        assert k >= 1
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.k = int(k)
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = bool(easy_margin)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

        # Sub-centers for all classes flattened as [C*K, D]
        self.weight = nn.Parameter(torch.empty(num_classes * self.k, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 2 or embeddings.size(1) != self.embedding_dim:
            raise ValueError(f"embeddings must be [B, {self.embedding_dim}]")
        if labels.ndim != 1 or labels.size(0) != embeddings.size(0):
            raise ValueError("labels must be [B] and match batch size")

        x = _normalize(embeddings, dim=1)
        W = _normalize(self.weight, dim=1)

        # Cosine to all sub-centers: [B, C*K]
        cos_all = F.linear(x, W).clamp(-1.0, 1.0)
        # Reshape to [B, C, K] and take max over sub-centers per class
        B = cos_all.size(0)
        cos_ck = cos_all.view(B, self.num_classes, self.k)
        cos, _ = cos_ck.max(dim=2)  # [B, C], gradient flows to the chosen sub-center

        sin = torch.sqrt((1.0 - cos.square()).clamp_min(0.0))
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        phi = cos * cos_m - sin * sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cos > 0, phi, cos)
        else:
            th = math.cos(math.pi - self.m)
            mm = math.sin(math.pi - self.m) * self.m
            phi = torch.where(cos > th, phi, cos - mm)

        # Replace target-class logits with phi
        logits = cos.clone()
        logits.scatter_(1, labels.view(-1, 1), phi.gather(1, labels.view(-1, 1)))
        logits = logits * self.s

        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            with torch.no_grad():
                true_dist = torch.full_like(logits, fill_value=eps / self.num_classes)
                true_dist.scatter_(1, labels.view(-1, 1), 1.0 - eps + (eps / self.num_classes))
            log_probs = F.log_softmax(logits, dim=1)
            loss = (-true_dist * log_probs).sum(dim=1)
            loss = loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss
        else:
            loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss, logits
