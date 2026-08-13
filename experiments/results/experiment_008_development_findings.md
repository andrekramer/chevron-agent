# Experiment 008 development findings

## Outcome

Dense counterfactual consequence supervision did not produce a better Chevron
comparison geometry than temporal similarity or one-step action prediction.
The frozen confirmation gate failed and confirmation seeds were not used.

## Representation result

The consequence-metric objective did change the representation in the intended
direction. Held-out correlation between embedding cosine and consequence-
signature cosine rose from 0.107 in the distorted raw sensor to 0.488, a gain
of 0.380. The 812-parameter encoder's loss fell from approximately 1.13 to
0.31 over 500 updates.

It nevertheless missed the predeclared 0.85 correlation target. Dense access
to all four action consequences was therefore not enough for the compact
encoder to reproduce the declared metric accurately.

## Downstream result

| Condition | Return | Final old | Final new | New probe | q calibration | Promotions |
|---|---:|---:|---:|---:|---:|---:|
| Oracle latent Chevron | 0.833 | 0.973 | 0.885 | 0.975 | 0.216 | 3.90 |
| Raw-sensor Chevron | 0.397 | 0.679 | 0.408 | 0.500 | 0.028 | 0.35 |
| Temporal-contrastive Chevron | 0.699 | 0.935 | 0.656 | 0.775 | 0.163 | 3.00 |
| Action-predictive Chevron | 0.678 | 0.928 | 0.608 | 0.750 | 0.143 | 3.00 |
| Consequence-metric Chevron | 0.611 | 0.893 | 0.540 | 0.662 | 0.154 | 2.45 |
| Consequence-metric content attention | 0.627 | 0.875 | 0.587 | 0.725 | 0.395 | 2.50 |

Consequence-metric Chevron was substantially better than raw sensor geometry,
but worse than temporal Chevron by 0.0875 return, with an approximate paired
95% interval from -0.1229 to -0.0521. It was also worse than action-predictive
Chevron by 0.0663 return, interval -0.1072 to -0.0255.

## Ideal-target audit

Because the learned encoder recovered only part of its target relation, a
post-development causal diagnostic supplied the exact four-dimensional
consequence signature directly to the unchanged Chevron gate. This diagnostic
did not alter the failed development decision or reopen confirmation.

The ideal consequence geometry itself reached only 0.494 return, 0.804 old
accuracy, 0.554 novel accuracy, 0.675 novel-probe accuracy, 0.102 q calibration,
and 2.80 promotions. Oracle latent geometry on the same twenty lifetimes
reached 0.833 return and 0.885 novel accuracy.

The target audit explains why. Although the three contexts within each address
family always required different optimal actions, 13.8% of competing context
pairs still had consequence-signature cosine at or above the frozen 0.62 assent
boundary. Their maximum similarity was 0.959. In the other direction, 15.4% of
noisy observations fell below the same boundary relative to their own retained
prototype. A consequence profile can be similar even when two memories must
remain distinct, and it can change under noise while contextual identity has
not changed.

## Interpretation

Behavioural consequence similarity is not identical to memory identity. A
single consequence metric is too coarse for both jobs: it can group situations
with similar affordances even when the agent must retain their distinct
histories, and observation noise can perturb an outcome signature independently
of contextual identity.

This result does not argue for tuning the sigmoid gate. Nor does it justify
proceeding to a sparse sampled-consequence learner, because even the exact dense
target failed. It instead suggests that assent may require at least two
relations:

1. an identity or persistence relation answering whether this is the same kind
   of situation or memory; and
2. a consequence relation answering whether the same action or policy remains
   appropriate.

A conservative next hypothesis would combine them as distinct assent factors
rather than forcing both meanings into one cosine space. Temporal identity
could retrieve or admit a candidate, while consequence incompatibility could
veto policy reuse or writing. That is a genuine architectural hypothesis and
should be tested in the compact task before any spatial demonstration.
