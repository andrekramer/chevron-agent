# Experiment 006a development findings

## Outcome

Label-free threshold and slope calibration did not close Experiment 006's
remaining gap. Confirmation was not run.

The calibrated gate used only temporal-positive and permuted-negative cosine
distributions. Across two encoder seeds:

- positive tenth percentile: 0.582;
- negative ninety-fifth percentile: 0.539;
- calibrated similarity threshold: 0.561;
- calibrated mismatch slope: 120, the frozen upper bound.

The small separation between conservative positives and hard negatives is
itself diagnostic: the temporal representation does not provide a clean
monotone compatibility boundary.

## Downstream comparison

| Condition | Return | Final old | Final new | New probe | q calibration | Promotions |
|---|---:|---:|---:|---:|---:|---:|
| Oracle geometric Chevron | 0.819 | 0.970 | 0.896 | 1.000 | 0.211 | 4.00 |
| Temporal Chevron, inherited gate | 0.709 | 0.936 | 0.693 | 0.813 | 0.137 | 3.35 |
| Temporal Chevron, calibrated gate | 0.686 | 0.911 | 0.679 | 0.763 | 0.145 | 2.70 |
| Temporal content attention, calibrated | 0.661 | 0.896 | 0.620 | 0.750 | 0.337 | 2.35 |
| Temporal Chevron, calibrated immediate | 0.599 | 0.934 | 0.365 | 0.463 | 0.193 | 0.00 |

Calibration made the Chevron result slightly worse rather than better. It did
not pass the causal comparison with the inherited gate, the absolute retention
and acquisition thresholds, or oracle non-inferiority.

The provisional buffer continued to matter: calibrated buffered Chevron
substantially exceeded immediate writing on novel acquisition, while immediate
writing made premature updates on 7.46% of decisions. This preserves the
Experiment 005a mechanism result even though the learned representation remains
insufficient.

## Conclusion

The missing component is not the sigmoid gate and is not merely its numerical
scale. Temporal instance consistency learns useful invariance but leaves
positive and hard-negative similarity distributions too entangled.

A further experiment would need a genuinely different signal: prediction of
action-conditioned transitions or consequences. Adjusting quantiles, slope
caps, encoder duration, or gate thresholds would be parameter tuning around a
failed hypothesis rather than progress toward a general agent.
