# Experiment 006a: frozen self-calibrated gate protocol

## Question

Can label-free calibration of the confirmed Chevron gate recover the remaining
performance lost when temporal contrastive learning changes the cosine scale?

Experiment 006 learned useful geometry but inherited the oracle space's cosine
threshold 0.62 and mismatch slope 40. It narrowly missed the downstream gate.
This experiment retrains the same encoder and changes only how the monotone gate
is calibrated.

## Calibration data

For 4096 fresh unlabelled temporal pairs, the frozen encoder produces cosine
similarities for:

- positive pairs: two adjacent views of the same persisting state;
- negative pairs: a deterministic one-position permutation of the second
  views, breaking the temporal pairing.

No context identity, action, reward, compatibility, memory state, or downstream
audit metric enters calibration.

## Frozen calibration rule

Let `p10` be the tenth percentile of positive similarity and `n95` the
ninety-fifth percentile of negative similarity:

```text
similarity_threshold = 0.5 * (p10 + n95)
separation = max(p10 - n95, 0.02)
mismatch_slope = clamp(4 * logit(0.9) / separation, 20, 120)
```

The factor four accounts for both the midpoint and conversion from cosine
similarity to half-cosine mismatch. Thus the gate aims for approximately 0.9
assent at `p10` and 0.1 at `n95` when the clamp is inactive.

Separate write permission retains the fixed 0.05 mismatch-threshold margin.
All encoder, sensor, memory, buffer, and task settings remain those of
Experiment 006.

## Conditions

1. Oracle geometric Chevron in latent space.
2. Temporal geometric Chevron with the inherited 0.62/40 gate.
3. Temporal geometric Chevron with label-free calibration.
4. Temporal content attention with the calibrated similarity threshold.
5. Calibrated temporal Chevron with immediate writing.

## Development and confirmation

Development retrains encoder seeds 0 and 1 exactly as in Experiment 006 and
uses ten fresh paired lifetimes per seed.

Confirmation is triggered only if calibrated temporal Chevron:

- reaches 0.95 final old accuracy;
- reaches 0.75 final novel and clean novel-probe accuracy;
- reaches 0.15 residual calibration and three promotions;
- has a paired return interval with lower bound above zero versus the inherited
  gate;
- has paired return and novel-accuracy lower bounds above zero versus immediate
  writing;
- has paired return and novel-accuracy lower bounds above -0.05 versus the
  oracle; and
- has a return lower bound above -0.05 versus calibrated content attention.

If triggered, confirmation trains untouched encoder seeds 700 through 709 and
evaluates twenty lifetimes per seed without changing the rule.

## Interpretation

Passing would complete the compact bridge from unlabelled temporal experience
to learned comparison geometry, calibrated vigilance, and protected category
acquisition. Failure would show that the representation needs an objective more
closely connected to predictive consequences, not merely scale calibration.

## Development outcome

The positive tenth percentile averaged 0.582 and the negative ninety-fifth
percentile 0.539. Their narrow separation forced the mismatch slope to its
predeclared maximum of 120; the calibrated similarity threshold was 0.561.

Calibration did not improve the inherited gate. Return changed from 0.709 to
0.686, final old accuracy from 0.936 to 0.911, final novel accuracy from 0.693
to 0.679, and promotions from 3.35 to 2.70. Residual calibration moved only
from 0.137 to 0.145. The calibrated buffer still decisively beat immediate
writing, but the central calibration and oracle-noninferiority criteria failed.
Confirmation was not run.

This rules out simple monotone rescaling as the missing component. The learned
space needs a denser objective tied to predicted transitions or consequences,
not another threshold percentile, slope, or training-duration sweep.
