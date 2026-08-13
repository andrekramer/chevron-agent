# Experiment 009: development dual-relation assent

- Seeds: 90000000–90000019

| Condition | Return | Stable | Reversed | Novel | q identity | q policy | New promotions | Revisions | Duplicates | Premature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dual relation + buffer | 0.587 +/- 0.096 | 0.951 +/- 0.046 | 0.826 +/- 0.119 | 0.598 +/- 0.179 | 0.273 +/- 0.030 | 0.991 +/- 0.015 | 2.850 +/- 1.014 | 4.200 +/- 1.030 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Collapsed relation + buffer | 0.540 +/- 0.048 | 0.963 +/- 0.023 | 0.615 +/- 0.082 | 0.583 +/- 0.090 | 0.150 +/- 0.044 | nan +/- nan | 4.700 +/- 1.229 | 0.000 +/- 0.000 | 5.850 +/- 1.459 | 0.000 +/- 0.000 |
| Identity only + buffer | 0.746 +/- 0.071 | 0.956 +/- 0.020 | 0.973 +/- 0.020 | 0.802 +/- 0.162 | 0.277 +/- 0.023 | nan +/- nan | 3.750 +/- 0.622 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Dual relation + immediate revision | 0.573 +/- 0.065 | 0.872 +/- 0.075 | 0.957 +/- 0.045 | 0.311 +/- 0.108 | nan +/- nan | nan +/- nan | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.118 +/- 0.020 |

## Frozen gate

- PASS: `stable_accuracy_at_least_0.95`
- PASS: `reversed_accuracy_at_least_0.75`
- FAIL: `novel_accuracy_at_least_0.75`
- PASS: `identity_calibration_at_least_0.15`
- PASS: `policy_calibration_at_least_0.15`
- FAIL: `new_promotions_at_least_3`
- PASS: `revision_promotions_at_least_3`
- PASS: `no_premature_writes`
- PASS: `no_established_overwrites`
- PASS: `no_duplicate_allocations`
- PASS: `return_better_than_collapsed`
- FAIL: `return_better_than_identity_only`
- FAIL: `return_better_than_immediate`
- FAIL: `novel_better_than_collapsed`

Confirmation triggered: **False**

## Paired diagnostics

```json
{
  "dual_buffer_minus_collapsed_buffer": {
    "return_per_decision_mean": 0.047,
    "return_per_decision_approx_95ci_low": 0.005543181340902044,
    "return_per_decision_approx_95ci_high": 0.08845681865909796,
    "return_per_decision_wins": 14,
    "final_stable_accuracy_mean": -0.012341106420301224,
    "final_stable_accuracy_approx_95ci_low": -0.03244485865993464,
    "final_stable_accuracy_approx_95ci_high": 0.007762645819332196,
    "final_stable_accuracy_wins": 8,
    "final_reversed_accuracy_mean": 0.21065823564567826,
    "final_reversed_accuracy_approx_95ci_low": 0.16042822582385532,
    "final_reversed_accuracy_approx_95ci_high": 0.2608882454675012,
    "final_reversed_accuracy_wins": 19,
    "final_novel_accuracy_mean": 0.015483163210683446,
    "final_novel_accuracy_approx_95ci_low": -0.0796791028270287,
    "final_novel_accuracy_approx_95ci_high": 0.1106454292483956,
    "final_novel_accuracy_wins": 8
  },
  "dual_buffer_minus_identity_only_buffer": {
    "return_per_decision_mean": -0.15833333333333335,
    "return_per_decision_approx_95ci_low": -0.2092587703303129,
    "return_per_decision_approx_95ci_high": -0.10740789633635381,
    "return_per_decision_wins": 2,
    "final_stable_accuracy_mean": -0.005096330807943461,
    "final_stable_accuracy_approx_95ci_low": -0.0255607872996493,
    "final_stable_accuracy_approx_95ci_high": 0.015368125683762376,
    "final_stable_accuracy_wins": 10,
    "final_reversed_accuracy_mean": -0.14714737201101075,
    "final_reversed_accuracy_approx_95ci_low": -0.19846457133283055,
    "final_reversed_accuracy_approx_95ci_high": -0.09583017268919095,
    "final_reversed_accuracy_wins": 2,
    "final_novel_accuracy_mean": -0.20343158117339005,
    "final_novel_accuracy_approx_95ci_low": -0.29572827000684454,
    "final_novel_accuracy_approx_95ci_high": -0.11113489233993559,
    "final_novel_accuracy_wins": 3
  },
  "dual_buffer_minus_dual_immediate": {
    "return_per_decision_mean": 0.01433333333333334,
    "return_per_decision_approx_95ci_low": -0.026842478444402806,
    "return_per_decision_approx_95ci_high": 0.05550914511106948,
    "return_per_decision_wins": 10,
    "final_stable_accuracy_mean": 0.07880514088086735,
    "final_stable_accuracy_approx_95ci_low": 0.036688590733007106,
    "final_stable_accuracy_approx_95ci_high": 0.12092169102872759,
    "final_stable_accuracy_wins": 16,
    "final_reversed_accuracy_mean": -0.13066613244146735,
    "final_reversed_accuracy_approx_95ci_low": -0.18387563418179778,
    "final_reversed_accuracy_approx_95ci_high": -0.07745663070113692,
    "final_reversed_accuracy_wins": 2,
    "final_novel_accuracy_mean": 0.2867848480032685,
    "final_novel_accuracy_approx_95ci_low": 0.21078462389568658,
    "final_novel_accuracy_approx_95ci_high": 0.36278507211085037,
    "final_novel_accuracy_wins": 18
  }
}
```
