# Experiment 010a findings: pre-consolidation identity revalidation

## Outcome

The correction passed every frozen criterion across 100 untouched paired
lifetimes.

Experiment 010a changed one rule only. When an aggregated provisional
new-identity candidate reached its delayed support requirement, it reran
identity assent against established memory before permanent allocation. A
candidate that now matched N was reconciled rather than written as a new slot.

No threshold, buffer capacity, support count, reward process, task mixture, or
retrospective policy rule changed.

## Fresh-seed result

| Condition | Return | Clean accuracy | Retention | Reversed probe | Novel probe | Duplicates | Reconciliations |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original protected Chevron | 0.5308 | 0.8347 | 0.9483 | 0.980 | 0.915 | 0.01 | 0.00 |
| Revalidated protected Chevron | 0.5308 | 0.8348 | 0.9483 | 0.980 | 0.915 | 0.00 | 0.01 |

The original path created one duplicate on seed `102000012`. The revalidated
path intercepted exactly that candidate and recorded one reconciliation. It
made no duplicate allocation, established-memory overwrite, or under-supported
write.

Revalidated Chevron still promoted 3.74 of four novel identities and revised
3.93 of four changed policies per lifetime.

## Cost of the correction

The correction was behaviourally neutral within measurement precision:

- return difference: -0.000025, approximate paired 95% interval -0.000074 to
  +0.000024;
- clean-accuracy difference: +0.000013, interval -0.000012 to +0.000037;
- retention difference: -0.000050, interval -0.000148 to +0.000048;
- reversed probe difference: exactly 0; and
- novel probe difference: exactly 0.

All five non-inferiority criteria passed by margins much wider than required.

## Architectural result

The provisional bank does not merely delay a write. It changes the evidence
being evaluated: multiple observations are averaged, outcomes accumulate, and
the candidate may become more recognisable while it waits.

Therefore the allocation decision cannot be treated as permanently settled at
buffer entry.

```text
entry assent decides whether evidence is provisionally unresolved

promotion assent decides whether the accumulated candidate is still novel
```

This gives Chevron a two-boundary consolidation rule:

1. withhold permanent judgement when current evidence fails assent; and
2. rerun assent after evidence matures, immediately before changing N.

The second boundary caught the exact rare failure for which it was introduced,
without reducing final acquisition or retention.

## Strongest defensible claim

> In this compact delayed stochastic-reward task, promotion-time identity
> revalidation eliminated duplicate allocation across 100 untouched lifetimes
> by reconciling a matured provisional candidate with established memory. It
> preserved the retrospective-policy mechanism's return, retention, novel
> acquisition, and reversal learning within tight paired bounds.

Experiment 010 remains a failed 13-of-14 confirmation and should continue to be
reported that way. Experiment 010a is the successful new audit that closes its
identified protection gap.

## Next step

The fixed mechanism now has both entry-time and promotion-time identity
protection. The next experiment can move to separately learned identity
geometry while keeping retrospective policy evidence, the shared typed bank,
and both consolidation boundaries fixed.
