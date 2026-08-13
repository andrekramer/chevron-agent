# Experiment 009a development findings

## Outcome

Experiment 009's buffer bottleneck was caused by insufficient total capacity,
not by identity and policy candidates sharing a queue. Separate queues did not
improve on a single capacity-matched shared queue. The primary routing gate
failed, so confirmation was not run.

## Main result

| Layout | Return | Stable | Reversed | Novel | New promotions | Revisions | Evictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shared 4 | 0.587 | 0.951 | 0.826 | 0.598 | 2.85 | 4.20 | 78.85 |
| Split 2+2 | 0.566 | 0.953 | 0.765 | 0.614 | 2.40 | 3.75 | 87.75 |
| Shared 8 | 0.725 | 0.960 | 0.944 | 0.864 | 3.85 | 4.20 | 22.75 |
| Split 4+4 | 0.721 | 0.963 | 0.946 | 0.836 | 3.65 | 4.30 | 26.05 |
| Identity only, shared 4 | 0.746 | 0.956 | 0.973 | 0.802 | 3.75 | 0.00 | 19.90 |

## Capacity result

Increasing one shared queue from four to eight entries improved return by
0.1375, with an approximate paired 95% interval from +0.0928 to +0.1822. It
improved final novel accuracy by 0.2663, interval +0.1755 to +0.3570, and
reversed accuracy by 0.1182, interval +0.0632 to +0.1731.

Mean evictions fell from 78.85 to 22.75, a reduction of 56.10 per lifetime.
Shared capacity eight recovered 3.85 of four novel identities and all four
policy revisions on average while preserving stable contexts.

This cleanly explains most of Experiment 009's novelty deficit. A four-entry
buffer was adequate when it carried novelty alone, but not when policy revision
traffic was added.

## Routing result

At equal total capacity eight, Split 4+4 did not improve on Shared 8:

- return difference: -0.0033, interval -0.0242 to +0.0176;
- novel-accuracy difference: -0.0283, interval -0.0765 to +0.0200;
- eviction difference: +3.30, interval +2.18 to +4.42.

Rigid separation slightly increased evictions because spare capacity in one
queue could not absorb a temporary burst in the other. The equal-total-capacity
Split 2+2 layout was also mildly harmful: its return non-inferiority interval
fell just beyond the predeclared -0.05 margin, and it completed fewer policy
revisions.

The evidence therefore argues against separate fixed queues. Identity and
policy candidates should retain distinct types and consolidation destinations,
but they can share flexible provisional capacity.

## Comparison with identity-only adaptation

Shared-8 dual Chevron reached 0.725 return versus 0.746 for identity-only. The
paired difference was -0.0208, with an interval from -0.0551 to +0.0135, so
there was no clear overall performance difference on these twenty development
lifetimes. Dual Chevron remained 0.029 lower on reversed-context accuracy, but
0.063 higher on novel accuracy; only the reversal difference excluded zero.

Removing buffer pressure therefore made the conservative dual mechanism
competitive with simple value adaptation, but did not establish superiority.
The deterministic reward task still favours rapid retrospective correction.

## Conclusion

The answer to the Experiment 009 question is:

> Keep identity and policy residuals logically separate, but do not give them
> rigidly separate queues. Give the typed candidates enough shared provisional
> capacity, or allow capacity to respond to unresolved traffic.

The capacity result was a predeclared secondary diagnostic, not a confirmed
fresh-seed result. A narrowly scoped confirmation should compare Shared 8,
Shared 4, and identity-only on untouched lifetimes before changing the task or
learning representations. If confirmed, the architecture can use one typed
candidate bank whose effective capacity is governed by sustained unresolved
mass rather than fixed partitions.

## Subsequent confirmation

Experiment 009b performed that narrow confirmation across 100 untouched
lifetimes. Every capacity and non-inferiority criterion passed. Shared 8
improved return over Shared 4 by +0.1557 and novel accuracy by +0.2783, while
reducing evictions by 76.7%. It was non-inferior overall to identity-only
adaptation. The capacity conclusion is therefore no longer development-only.
