"""Refined Chevron Attention mathematical kernel.

This module deliberately contains no RL policy.  It implements the smallest
testable object from ``maths2.md``: retrieval, asymmetric assent, admitted read
mass, attributed residual mass, conservative write permission, and a
provisional-allocation signal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class ChevronAttentionConfig:
    """Numerical choices for the diagnostic Chevron Attention kernel."""

    read_threshold: float = 0.15
    write_threshold: float = 0.08
    read_slope: float = 40.0
    write_slope: float = 50.0
    allocation_residual_threshold: float = 0.80
    allocation_admitted_threshold: float = 0.25
    top_k: int | None = None
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if not 0.0 <= self.write_threshold < self.read_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= write_threshold "
                "< read_threshold <= 1"
            )
        if self.read_slope <= 0.0 or self.write_slope <= 0.0:
            raise ValueError("read and write slopes must be positive")
        if not 0.0 <= self.allocation_residual_threshold <= 1.0:
            raise ValueError("allocation_residual_threshold must be in [0, 1]")
        if not 0.0 <= self.allocation_admitted_threshold <= 1.0:
            raise ValueError("allocation_admitted_threshold must be in [0, 1]")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when supplied")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


@dataclass(frozen=True)
class ChevronAttentionOutput:
    """All observable quantities needed for causal diagnostics."""

    alpha: Tensor
    mismatch: Tensor
    read_assent: Tensor
    read_mass: Tensor
    slot_residual: Tensor
    total_residual: Tensor
    read_output: Tensor
    write_assent: Tensor
    write_gate: Tensor
    selected_mask: Tensor
    allocate_provisional: Tensor


def normalized_cosine_mismatch(evidence: Tensor, retained: Tensor, eps: float = 1e-8) -> Tensor:
    """Return cosine distance scaled to the declared interval [0, 1].

    ``evidence`` has shape ``[batch, dim]`` and ``retained`` has shape
    ``[batch, slots, dim]``.  The explicit scaling makes read and write
    thresholds comparable across experiments.
    """

    if evidence.ndim != 2 or retained.ndim != 3:
        raise ValueError("evidence must be [batch, dim] and retained [batch, slots, dim]")
    if evidence.shape[0] != retained.shape[0] or evidence.shape[-1] != retained.shape[-1]:
        raise ValueError("evidence and retained content dimensions must match")

    evidence_unit = F.normalize(evidence, dim=-1, eps=eps)
    retained_unit = F.normalize(retained, dim=-1, eps=eps)
    cosine = torch.einsum("bd,bsd->bs", evidence_unit, retained_unit).clamp(-1.0, 1.0)
    return 0.5 * (1.0 - cosine)


def _ensure_batch(vector: Tensor, name: str) -> Tensor:
    if vector.ndim == 1:
        return vector.unsqueeze(0)
    if vector.ndim != 2:
        raise ValueError(f"{name} must be [dim] or [batch, dim]")
    return vector


def _expand_memory(memory: Tensor, batch: int, name: str) -> Tensor:
    if memory.ndim == 2:
        return memory.unsqueeze(0).expand(batch, -1, -1)
    if memory.ndim != 3 or memory.shape[0] != batch:
        raise ValueError(f"{name} must be [slots, dim] or [batch, slots, dim]")
    return memory


def chevron_attention(
    address_query: Tensor,
    evidence: Tensor,
    address_memory: Tensor,
    content_memory: Tensor,
    *,
    eligibility: Tensor | None = None,
    config: ChevronAttentionConfig | None = None,
) -> ChevronAttentionOutput:
    """Apply refined Chevron Attention to an explicit A/N memory.

    Address features participate only in retrieval.  Evidence and retained
    content participate only in assent.  This informational separation is the
    first safeguard against assent collapsing into retrieval twice.
    """

    cfg = config or ChevronAttentionConfig()
    address_query = _ensure_batch(address_query, "address_query")
    evidence = _ensure_batch(evidence, "evidence")
    if address_query.shape[0] != evidence.shape[0]:
        raise ValueError("address_query and evidence batch dimensions must match")

    batch = address_query.shape[0]
    address_memory_b = _expand_memory(address_memory, batch, "address_memory")
    content_memory_b = _expand_memory(content_memory, batch, "content_memory")

    if address_memory_b.shape[1] != content_memory_b.shape[1]:
        raise ValueError("address and content memories must contain the same number of slots")
    if address_query.shape[-1] != address_memory_b.shape[-1]:
        raise ValueError("address query and address memory dimensions must match")
    if evidence.shape[-1] != content_memory_b.shape[-1]:
        raise ValueError("evidence and content memory dimensions must match")

    slots = address_memory_b.shape[1]
    value_dim = content_memory_b.shape[-1]
    device = address_query.device
    dtype = address_query.dtype

    if slots == 0:
        empty = torch.empty((batch, 0), device=device, dtype=dtype)
        empty_bool = torch.empty((batch, 0), device=device, dtype=torch.bool)
        ones = torch.ones((batch,), device=device, dtype=dtype)
        return ChevronAttentionOutput(
            alpha=empty,
            mismatch=empty,
            read_assent=empty,
            read_mass=empty,
            slot_residual=empty,
            total_residual=ones,
            read_output=torch.zeros((batch, value_dim), device=device, dtype=dtype),
            write_assent=empty,
            write_gate=empty,
            selected_mask=empty_bool,
            allocate_provisional=torch.ones((batch,), device=device, dtype=torch.bool),
        )

    logits = torch.einsum("bd,bsd->bs", address_query, address_memory_b)
    logits = logits / math.sqrt(address_query.shape[-1])
    alpha = torch.softmax(logits, dim=-1)

    if cfg.top_k is None or cfg.top_k >= slots:
        selected_mask = torch.ones_like(alpha, dtype=torch.bool)
    else:
        indices = torch.topk(alpha, k=cfg.top_k, dim=-1).indices
        selected_mask = torch.zeros_like(alpha, dtype=torch.bool)
        selected_mask.scatter_(dim=-1, index=indices, value=True)
    selected = selected_mask.to(dtype=dtype)

    mismatch = normalized_cosine_mismatch(evidence, content_memory_b, eps=cfg.eps)
    read_assent = torch.sigmoid(cfg.read_slope * (cfg.read_threshold - mismatch))
    read_mass = alpha * read_assent * selected

    # This definition includes all dropped top-k mass in the residual.  It is
    # equivalent to alpha * (1-r) when every slot is selected.
    slot_residual = alpha - read_mass
    total_residual = slot_residual.sum(dim=-1)
    read_output = torch.einsum("bs,bsd->bd", read_mass, content_memory_b)

    write_assent = torch.sigmoid(cfg.write_slope * (cfg.write_threshold - mismatch))
    if eligibility is None:
        eligibility_b = torch.ones_like(alpha)
    else:
        eligibility_b = eligibility
        if eligibility_b.ndim == 1:
            eligibility_b = eligibility_b.unsqueeze(0)
        if eligibility_b.shape != alpha.shape:
            raise ValueError("eligibility must have shape [slots] or [batch, slots]")
        if torch.any((eligibility_b < 0.0) | (eligibility_b > 1.0)).item():
            raise ValueError("eligibility values must be in [0, 1]")
        eligibility_b = eligibility_b.to(device=device, dtype=dtype)

    write_gate = alpha * write_assent * eligibility_b * selected
    maximum_admitted = read_mass.max(dim=-1).values
    allocate_provisional = (
        (total_residual > cfg.allocation_residual_threshold)
        & (maximum_admitted < cfg.allocation_admitted_threshold)
    )

    return ChevronAttentionOutput(
        alpha=alpha,
        mismatch=mismatch,
        read_assent=read_assent,
        read_mass=read_mass,
        slot_residual=slot_residual,
        total_residual=total_residual,
        read_output=read_output,
        write_assent=write_assent,
        write_gate=write_gate,
        selected_mask=selected_mask,
        allocate_provisional=allocate_provisional,
    )


def apply_convex_write(
    address_memory: Tensor,
    content_memory: Tensor,
    address_target: Tensor,
    content_target: Tensor,
    write_gate: Tensor,
    *,
    eta_address: float,
    eta_content: float,
    detach_state: bool = False,
) -> tuple[Tensor, Tensor]:
    """Apply a single observation's write gate to A and N memory slots.

    The same permission gates both updates, preventing a rejected observation
    from moving an established address toward content that N does not encode.
    """

    if address_memory.ndim != 2 or content_memory.ndim != 2:
        raise ValueError("memory tensors must be [slots, dim]")
    if address_memory.shape[0] != content_memory.shape[0]:
        raise ValueError("address and content memories must have equal slot counts")
    if address_target.ndim != 1 or address_target.shape[0] != address_memory.shape[-1]:
        raise ValueError("address_target must match the address dimension")
    if content_target.ndim != 1 or content_target.shape[0] != content_memory.shape[-1]:
        raise ValueError("content_target must match the content dimension")
    if write_gate.ndim == 2:
        if write_gate.shape[0] != 1:
            raise ValueError("state writes currently accept one observation at a time")
        write_gate = write_gate.squeeze(0)
    if write_gate.ndim != 1 or write_gate.shape[0] != address_memory.shape[0]:
        raise ValueError("write_gate must contain one value per memory slot")
    if torch.any((write_gate < 0.0) | (write_gate > 1.0)).item():
        raise ValueError("write_gate values must be in [0, 1]")
    if not 0.0 <= eta_address <= 1.0 or not 0.0 <= eta_content <= 1.0:
        raise ValueError("write rates must be in [0, 1]")

    gate_address = (eta_address * write_gate).unsqueeze(-1)
    gate_content = (eta_content * write_gate).unsqueeze(-1)
    proposed_address = (1.0 - gate_address) * address_memory + gate_address * address_target
    proposed_content = (1.0 - gate_content) * content_memory + gate_content * content_target

    # Preserve the protection invariant exactly when permission is zero.
    protected = write_gate.unsqueeze(-1) == 0.0
    updated_address = torch.where(protected, address_memory, proposed_address)
    updated_content = torch.where(protected, content_memory, proposed_content)

    if detach_state:
        updated_address = updated_address.detach()
        updated_content = updated_content.detach()
    return updated_address, updated_content
