# Experiment 009b: shared-capacity confirmation

- Fresh seeds: 92000000–92000099

| Condition | Return | Stable | Reversed | Novel | New promotions | Revisions | Evictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dual shared 4 | 0.594 +/- 0.115 | 0.972 +/- 0.030 | 0.787 +/- 0.180 | 0.627 +/- 0.222 | 2.900 +/- 1.162 | 3.760 +/- 0.873 | 75.070 +/- 29.819 |
| Dual shared 8 | 0.750 +/- 0.056 | 0.965 +/- 0.043 | 0.952 +/- 0.052 | 0.906 +/- 0.094 | 3.900 +/- 0.332 | 4.480 +/- 0.900 | 17.490 +/- 6.051 |
| Identity only shared 4 | 0.760 +/- 0.049 | 0.955 +/- 0.035 | 0.972 +/- 0.028 | 0.853 +/- 0.097 | 3.860 +/- 0.375 | 0.000 +/- 0.000 | 16.920 +/- 5.846 |

## Capacity gate

- PASS: `stable_accuracy_at_least_0.95`
- PASS: `reversed_accuracy_at_least_0.80`
- PASS: `novel_accuracy_at_least_0.80`
- PASS: `identity_calibration_at_least_0.15`
- PASS: `policy_calibration_at_least_0.15`
- PASS: `new_promotions_at_least_3`
- PASS: `revision_promotions_at_least_3`
- PASS: `no_premature_writes`
- PASS: `no_established_overwrites`
- PASS: `no_duplicate_allocations`
- PASS: `return_better_than_shared_4`
- PASS: `reversed_better_than_shared_4`
- PASS: `novel_better_than_shared_4`
- PASS: `evictions_reduced_by_50_percent`

## Identity-only non-inferiority gate

- PASS: `return_noninferior_to_identity_only`
- PASS: `stable_noninferior_to_identity_only`
- PASS: `reversed_noninferior_to_identity_only`
- PASS: `novel_noninferior_to_identity_only`

Capacity confirmed: **True**
Competitive with identity-only: **True**

## Paired diagnostics

```json
{
  "dual_shared_8_minus_dual_shared_4": {
    "return_per_decision_mean": 0.1557,
    "return_per_decision_population_sd": 0.11477155570959208,
    "return_per_decision_approx_95ci_low": 0.13320477508091996,
    "return_per_decision_approx_95ci_high": 0.17819522491908005,
    "return_per_decision_wins": 93,
    "final_stable_accuracy_mean": -0.006922698418470552,
    "final_stable_accuracy_population_sd": 0.049105844780803605,
    "final_stable_accuracy_approx_95ci_low": -0.01654744399550806,
    "final_stable_accuracy_approx_95ci_high": 0.0027020471585669547,
    "final_stable_accuracy_wins": 27,
    "final_reversed_accuracy_mean": 0.16436642034676827,
    "final_reversed_accuracy_population_sd": 0.19157642828535096,
    "final_reversed_accuracy_approx_95ci_low": 0.1268174404028395,
    "final_reversed_accuracy_approx_95ci_high": 0.20191540029069704,
    "final_reversed_accuracy_wins": 76,
    "final_novel_accuracy_mean": 0.2782779220828803,
    "final_novel_accuracy_population_sd": 0.22642982026225045,
    "final_novel_accuracy_approx_95ci_low": 0.23389767731147923,
    "final_novel_accuracy_approx_95ci_high": 0.3226581668542814,
    "final_novel_accuracy_wins": 87,
    "buffer_evictions_mean": -57.58,
    "buffer_evictions_population_sd": 28.087427792519556,
    "buffer_evictions_approx_95ci_low": -63.08513584733383,
    "buffer_evictions_approx_95ci_high": -52.074864152666166,
    "buffer_evictions_wins": 0
  },
  "dual_shared_8_minus_identity_only_shared_4": {
    "return_per_decision_mean": -0.010000000000000004,
    "return_per_decision_population_sd": 0.06621178142898738,
    "return_per_decision_approx_95ci_low": -0.02297750916008153,
    "return_per_decision_approx_95ci_high": 0.0029775091600815224,
    "return_per_decision_wins": 45,
    "final_stable_accuracy_mean": 0.00988289695019118,
    "final_stable_accuracy_population_sd": 0.05099544216081335,
    "final_stable_accuracy_approx_95ci_low": -0.00011220971332823702,
    "final_stable_accuracy_approx_95ci_high": 0.019878003613710594,
    "final_stable_accuracy_wins": 55,
    "final_reversed_accuracy_mean": -0.020575275561158082,
    "final_reversed_accuracy_population_sd": 0.05456052081194723,
    "final_reversed_accuracy_approx_95ci_low": -0.03126913764029974,
    "final_reversed_accuracy_approx_95ci_high": -0.009881413482016425,
    "final_reversed_accuracy_wins": 19,
    "final_novel_accuracy_mean": 0.05294225198884615,
    "final_novel_accuracy_population_sd": 0.11780326915752197,
    "final_novel_accuracy_approx_95ci_low": 0.02985281123397185,
    "final_novel_accuracy_approx_95ci_high": 0.07603169274372046,
    "final_novel_accuracy_wins": 69,
    "buffer_evictions_mean": 0.57,
    "buffer_evictions_population_sd": 5.824525731765634,
    "buffer_evictions_approx_95ci_low": -0.5716070434260644,
    "buffer_evictions_approx_95ci_high": 1.7116070434260644,
    "buffer_evictions_wins": 50
  }
}
```
