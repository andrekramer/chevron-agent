# Experiment 005 development findings

## Outcome

Experiment 005 does not justify moving this Chevron configuration into a
spatial game. The retrospective reward objective was correctly connected to
both A/N comparison paths, but it produced no detectable improvement over the
policy-only gate.

This is the bounded stopping result proposed after Experiment 004. The frozen
confirmation seeds were not used.

## What changed

The environment, delayed reward, provisional promotion rule, permanent memory,
and observation interface remained fixed. Buffer capacity was four, matching
the number of novel contexts and the declared Experiment 004 diagnostic.

Two changes were tested:

1. Delayed reward retrospectively scored whether admitted memory supported the
   selected action and correctly predicted its success or failure.
2. A projected bilinear slot-or-null attention model replaced the underfitting
   direct MLP. It received the same information, the same retrospective loss,
   and the same 314-parameter budget as Chevron.

Neither training objective received a context identity, correct action,
compatibility label, or target memory slot.

## Main result

Across two training seeds and 20 fresh evaluation lifetimes per condition:

| Condition | Return | Final old | Final new | New probe | q calibration | Promotions |
|---|---:|---:|---:|---:|---:|---:|
| Content attention + buffer | 0.788 | 0.982 | 0.821 | 0.887 | 0.457 | 3.55 |
| Bilinear + retrospective + buffer | 0.458 | 0.749 | 0.327 | 0.400 | 0.067 | 0.00 |
| Chevron + retrospective + buffer | 0.512 | 0.806 | 0.357 | 0.438 | 0.030 | 0.55 |
| Chevron + policy only + buffer | 0.513 | 0.805 | 0.362 | 0.412 | 0.032 | 0.55 |
| Chevron + retrospective + immediate | 0.554 | 0.869 | 0.354 | 0.475 | 0.044 | 0.00 |
| Chevron + retrospective + coupled write | 0.503 | 0.804 | 0.331 | 0.350 | 0.033 | 0.50 |

Retrospective Chevron minus policy-only Chevron was -0.0005 return per
decision. Its approximate paired 95% interval was -0.0155 to +0.0145. Final
old accuracy, final novel accuracy, calibration, and promotion count were also
effectively unchanged. The auxiliary outcome-consistency target did not solve
the credit problem.

Chevron did outperform projected bilinear attention by 0.0548 return, with an
approximate paired interval from +0.0204 to +0.0893. This is a real development
comparison, but it is not enough to support a broad claim: both learned models
failed the acquisition and calibration thresholds, while the fixed controller
solved the task substantially better.

## Protection mechanics

The protected memory path continued to behave as specified:

- buffered Chevron made no permanent writes before delayed evidence;
- immediate Chevron made premature writes on 1.0% of decisions;
- separate write assent preserved a positive read/write margin;
- separate write assent reduced established-memory drift from 0.150 for the
  coupled gate to 0.135.

However, immediate writing achieved higher return and old accuracy than the
buffered model in this development run. Protection is therefore demonstrated
as an invariant, not as a performance advantage on this task.

## Why this result matters

The fixed content controller reached 0.982 old accuracy, 0.821 novel accuracy,
and promoted 3.55 of four novel contexts. The task is solvable, the buffer can
hold the required candidates, and delayed positive evidence can drive useful
promotion. The failure is more specific: neither sparse policy reward nor the
selected-action consistency target learned the comparison geometry needed to
turn q into a reliable vigilance signal.

This also explains why simply adding a spatial environment would be premature.
It would add observation encoding, exploration, temporal credit, and value
learning before the isolated comparison mechanism works.

## Conclusion

The strongest defensible conclusion is:

> Explicit provisional storage and stricter write permission provide testable
> memory-protection semantics, but the current differentiable Chevron assent
> gate has not learned useful vigilance from delayed scalar reward.

Further scaling of this exact configuration is not recommended. Continuing the
research would require a conceptual change rather than another threshold or
training-duration sweep—for example, self-supervised predictive comparison,
contrastive temporal consistency, or a learned representation whose objective
exposes stable context structure independently of sparse task reward.

Any such work should be treated as a new hypothesis. The present experiment
has completed its intended go/no-go decision.

## Subsequent geometric isolation

Experiment 005a subsequently held the gate formula fixed but removed the
reward-trained projections. Parameter-free cosine mismatch passed its
100-lifetime fresh-seed confirmation, including strong buffering and
write-protection ablations. This does not reverse Experiment 005's negative
learning result. It localises it: the current gate and buffer are viable when
the comparison geometry is informative, while learning that geometry from
sparse reward remains unsuccessful.
