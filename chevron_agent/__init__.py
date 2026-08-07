"""Core components for Chevron Agent experiments."""

from .attention import (
    ChevronAttentionConfig,
    ChevronAttentionOutput,
    apply_convex_write,
    chevron_attention,
    normalized_cosine_mismatch,
)
from .buffer import BoundedProvisionalBuffer, ProvisionalEntry
from .gates import (
    AssentGateOutput,
    DirectDecisionOutput,
    DirectPairMLP,
    ProjectedCosineAssent,
)

__all__ = [
    "ChevronAttentionConfig",
    "ChevronAttentionOutput",
    "apply_convex_write",
    "chevron_attention",
    "normalized_cosine_mismatch",
    "AssentGateOutput",
    "ProjectedCosineAssent",
    "DirectDecisionOutput",
    "DirectPairMLP",
    "BoundedProvisionalBuffer",
    "ProvisionalEntry",
]
