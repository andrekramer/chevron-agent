"""Learnable assent gates with explicit A/N comparison paths."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AssentGateOutput:
    logits: Tensor
    assent: Tensor
    mismatch: Tensor
    threshold: Tensor
    slope: Tensor


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class ProjectedCosineAssent(nn.Module):
    """Compare evidence and retained content after independent projections.

    The gate learns an alignment between an A evidence space and an N retained
    space.  Its scalar threshold remains in [0, 1] and its slope is positive by
    construction.
    """

    def __init__(
        self,
        evidence_dim: int,
        retained_dim: int,
        comparison_dim: int,
        *,
        initial_threshold: float = 0.20,
        initial_slope: float = 10.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if evidence_dim <= 0 or retained_dim <= 0 or comparison_dim <= 0:
            raise ValueError("all dimensions must be positive")
        if not 0.0 < initial_threshold < 1.0:
            raise ValueError("initial_threshold must be strictly inside (0, 1)")
        if initial_slope <= 0.0:
            raise ValueError("initial_slope must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.evidence_projection = nn.Linear(evidence_dim, comparison_dim, bias=False)
        self.retained_projection = nn.Linear(retained_dim, comparison_dim, bias=False)
        self.raw_threshold = nn.Parameter(
            torch.tensor(math.log(initial_threshold / (1.0 - initial_threshold)))
        )
        self.raw_slope = nn.Parameter(torch.tensor(_inverse_softplus(initial_slope)))
        self.eps = eps

    @property
    def threshold(self) -> Tensor:
        return torch.sigmoid(self.raw_threshold)

    @property
    def slope(self) -> Tensor:
        return F.softplus(self.raw_slope) + self.eps

    def compare(self, evidence: Tensor, retained: Tensor) -> Tensor:
        if evidence.ndim != 2:
            raise ValueError("evidence must be [batch, dim]")
        if retained.ndim == 2:
            retained = retained.unsqueeze(1)
        if retained.ndim != 3 or retained.shape[0] != evidence.shape[0]:
            raise ValueError("retained must be [batch, slots, dim]")

        evidence_projected = F.normalize(
            self.evidence_projection(evidence), dim=-1, eps=self.eps
        )
        retained_projected = F.normalize(
            self.retained_projection(retained), dim=-1, eps=self.eps
        )
        cosine = torch.einsum(
            "bd,bsd->bs", evidence_projected, retained_projected
        ).clamp(-1.0, 1.0)
        return 0.5 * (1.0 - cosine)

    def forward(self, evidence: Tensor, retained: Tensor) -> AssentGateOutput:
        mismatch = self.compare(evidence, retained)
        logits = self.slope * (self.threshold - mismatch)
        return AssentGateOutput(
            logits=logits,
            assent=torch.sigmoid(logits),
            mismatch=mismatch,
            threshold=self.threshold,
            slope=self.slope,
        )

    def assent_with_margin(
        self, evidence: Tensor, retained: Tensor, *, threshold_margin: float
    ) -> Tensor:
        """Return a stricter gate while preserving a positive threshold margin."""

        if threshold_margin <= 0.0:
            raise ValueError("threshold_margin must be positive")
        mismatch = self.compare(evidence, retained)
        write_threshold = torch.clamp(self.threshold - threshold_margin, min=0.0)
        return torch.sigmoid(self.slope * (write_threshold - mismatch))
