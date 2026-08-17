# Experiment 011 development findings

## Result

The learned identity path worked, but the proposed hard-persistence refinement
did not. The frozen development gate failed, so fresh-seed confirmation was not
run.

The fixed nonlinear sensor badly distorted the identity decision. It admitted
99.8% of repeat views, but rejected only 16.1% of confusable identity changes at
the frozen 0.62 boundary. Consequently, raw-sensor Chevron almost never learned
the four new identities: its novel probe accuracy was 0.088 and return was
0.331.

All three temporal objectives repaired most of this boundary:

| Representation | Same admitted | Confusable rejected | Balanced decision | Return | Novel probe |
|---|---:|---:|---:|---:|---:|
| Raw sensor | 0.998 | 0.161 | 0.580 | 0.331 | 0.088 |
| Pairwise temporal | 0.986 | 0.824 | 0.905 | 0.491 | 0.863 |
| Multi-view temporal | 0.984 | 0.829 | 0.907 | 0.479 | 0.788 |
| Hard persistence | 0.985 | 0.824 | 0.904 | 0.470 | 0.738 |

Hard-persistence protected Chevron improved return over the raw sensor by
+0.139, with an approximate paired 95% interval from +0.106 to +0.172. It
improved novel probe accuracy by +0.650. It retained 0.922 stable accuracy,
promoted 3.5 of four novel identities, recovered 3.8 of four reversed policies,
and made no duplicate allocations, established-memory overwrites, or
under-supported writes.

Those are useful absolute results, but they do not support the proposed extra
curriculum. Hard persistence trailed the simpler pairwise temporal encoder by
0.021 return, interval -0.043 to +0.001, and by 0.125 novel-probe accuracy,
interval -0.248 to -0.003. It also missed the 0.75 novel-probe floor by 0.0125.
Its return gap to the oracle was -0.095, with interval -0.128 to -0.061, outside
the frozen -0.08 non-inferiority margin.

## What the experiment shows

Temporal self-supervision can learn a useful identity comparison for the
Chevron gate without seeing the RL categories, actions, rewards, or policies.
The strongest development condition was the simplest one: two noisy views of a
newly sampled persistent identity, trained with pairwise contrastive learning.
It reached 0.802 latent-cosine correlation, 0.905 balanced identity-decision
accuracy, 0.491 return, and 0.863 novel-probe accuracy.

The more identity-specific loss did not add useful information. Four-view
windows and deliberately confusable changes produced nearly the same held-out
identity-boundary accuracy as pairwise training, then slightly worse downstream
control. The apparent hard negatives were not a scarce training signal: with
new random identities on every batch, ordinary contrastive negatives were
already sufficient to learn the useful correction.

The representation diagnostics were necessary but not sufficient. All three
learned encoders scored near 0.905 on the balanced identity decision, while
their downstream novel probes ranged from 0.738 to 0.863. A single clean-template
pair test does not capture candidate clustering, noisy online retrieval, and
delayed promotion over a lifetime.

## Architectural implication

Do not add the hard-persistence curriculum to the current architecture. Keep
the small residual identity encoder and the simple pairwise temporal objective
as the leading learned-identity candidate. The downstream Chevron mechanism
does not need to change: promotion-time revalidation preserved all tested
protection invariants, and the protected policy path remained within its frozen
non-inferiority margins against direct value adaptation.

This experiment does not confirm the pairwise encoder, because it was a control
inside a development study whose predeclared confirmation trigger named the
hard-persistence condition. The clean next step is a narrow Experiment 011a:
freeze the simpler pairwise learner exactly as run here and test it on untouched
encoder and RL seeds. That is a simplification selected by the development
result, not a post-hoc rescue of the failed hard-persistence condition.

## Strongest defensible claim

On this compact delayed-context task, temporal contrastive learning recovered
an identity geometry that made a fixed Chevron gate substantially better than
using the distorted sensor directly, while preserving the established memory
protection invariants. The more elaborate persistence-window and hard-negative
objectives did not outperform simple temporal pairing. This is development
evidence, not yet a fresh-seed confirmation or a comparison with standard
attention in a visual RL environment.
