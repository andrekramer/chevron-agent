# Experiment 009b confirmation findings

## Outcome

The shared-capacity result is confirmed. Across 100 untouched lifetimes, a
capacity-eight typed provisional bank decisively outperformed capacity four
while preserving every protection invariant. It was also non-inferior overall
to the simpler identity-only adaptation control under all predeclared margins.

All fourteen capacity criteria and all four control criteria passed.

## Fresh-seed result

| Condition | Return | Stable | Reversed | Novel | New promotions | Revisions | Evictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dual shared 4 | 0.594 | 0.972 | 0.787 | 0.627 | 2.90 | 3.76 | 75.07 |
| Dual shared 8 | 0.750 | 0.965 | 0.952 | 0.906 | 3.90 | 4.48 | 17.49 |
| Identity only shared 4 | 0.760 | 0.955 | 0.972 | 0.853 | 3.86 | 0.00 | 16.92 |

Shared 8 versus Shared 4:

- return improved by 0.1557, approximate paired 95% interval +0.1332 to
  +0.1782;
- reversed-context accuracy improved by 0.1644, interval +0.1268 to +0.2019;
- novel-context accuracy improved by 0.2783, interval +0.2339 to +0.3227;
- eviction count fell by 57.58, interval -63.09 to -52.07; and
- mean evictions fell from 75.07 to 17.49, a 76.7% reduction.

Shared 8 consolidated 3.90 of four novel identities and 4.48 policy revisions
per lifetime on average. Revision count can exceed four because later noisy
evidence may trigger a further coherent revision of an already changed policy.

It made no premature permanent writes, no established-memory overwrites, and
no duplicate identity allocations.

## Comparison with identity-only adaptation

Shared-8 dual Chevron reached 0.750 return versus 0.760 for identity-only. The
paired difference was -0.0100, interval -0.0230 to +0.0030, comfortably within
the predeclared -0.05 non-inferiority margin.

The conditions showed a meaningful tradeoff:

- dual Chevron was 0.0206 lower on reversed-context accuracy, interval -0.0313
  to -0.0099;
- dual Chevron was 0.0529 higher on novel-context accuracy, interval +0.0299
  to +0.0760; and
- stable-context performance was statistically indistinguishable.

Direct value adaptation remains slightly faster for deterministic policy
reversals. The protected dual mechanism acquires novel identities more
reliably, while matching overall return within the declared margin.

## Strongest defensible claim

> In this compact delayed novelty-and-reversal task, a typed shared provisional
> bank with capacity for the combined unresolved traffic enabled dual identity
> and policy assent to preserve stable knowledge, revise familiar policies, and
> acquire novel identities without premature writes or duplicate memories.
> Capacity eight decisively outperformed capacity four across 100 untouched
> lifetimes and was non-inferior overall to rapid identity-only value adaptation.

This confirms capacity for this fixed traffic distribution, not the universal
choice of eight entries. The task supplies meaningful identity and policy
geometries, deterministic binary reward, four actions, and twelve compact
contexts. It does not yet establish learned dual representations, visual
agency, or a persistent core self.

## Design conclusion

The evidence now supports:

- separate identity and policy residuals;
- typed candidates with different consolidation destinations;
- one flexible shared provisional bank rather than rigid partitions; and
- capacity determined by combined unresolved traffic rather than novelty alone.

A useful adaptive version could let sustained occupancy, eviction pressure, or
the joint identity/policy residual field regulate effective capacity. That is a
future hypothesis. The next conservative step is not another capacity sweep:
it is learning the two relations separately while retaining this confirmed
shared-buffer mechanism.
