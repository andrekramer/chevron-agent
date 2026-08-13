# Experiment 009a: development split-queue diagnostic

- Seeds: 90000000–90000019

| Condition | Return | Stable | Reversed | Novel | New promotions | Revisions | Evictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dual shared 4 | 0.587 +/- 0.096 | 0.951 +/- 0.046 | 0.826 +/- 0.119 | 0.598 +/- 0.179 | 2.850 +/- 1.014 | 4.200 +/- 1.030 | 78.850 +/- 25.156 |
| Dual split 2+2 | 0.566 +/- 0.092 | 0.953 +/- 0.055 | 0.765 +/- 0.182 | 0.614 +/- 0.230 | 2.400 +/- 1.281 | 3.750 +/- 0.829 | 87.750 +/- 22.378 |
| Dual shared 8 | 0.725 +/- 0.060 | 0.960 +/- 0.018 | 0.944 +/- 0.067 | 0.864 +/- 0.106 | 3.850 +/- 0.572 | 4.200 +/- 0.600 | 22.750 +/- 8.746 |
| Dual split 4+4 | 0.721 +/- 0.066 | 0.963 +/- 0.019 | 0.946 +/- 0.066 | 0.836 +/- 0.133 | 3.650 +/- 0.654 | 4.300 +/- 0.714 | 26.050 +/- 9.097 |
| Identity only shared 4 | 0.746 +/- 0.071 | 0.956 +/- 0.020 | 0.973 +/- 0.020 | 0.802 +/- 0.162 | 3.750 +/- 0.622 | 0.000 +/- 0.000 | 19.900 +/- 7.049 |

## Primary routing gate

- PASS: `stable_accuracy_at_least_0.95`
- PASS: `reversed_accuracy_at_least_0.75`
- PASS: `novel_accuracy_at_least_0.75`
- PASS: `identity_calibration_at_least_0.15`
- PASS: `policy_calibration_at_least_0.15`
- PASS: `new_promotions_at_least_3`
- PASS: `revision_promotions_at_least_3`
- PASS: `no_premature_writes`
- PASS: `no_established_overwrites`
- PASS: `no_duplicate_allocations`
- FAIL: `return_better_than_shared_8`
- FAIL: `novel_better_than_shared_8`
- FAIL: `evictions_reduced_by_25_percent`

## Secondary diagnostics

- PASS: `shared_8_return_better_than_shared_4`
- PASS: `shared_8_novel_better_than_shared_4`
- FAIL: `split_2_2_return_noninferior_to_shared_4`
- FAIL: `split_2_2_novel_noninferior_to_shared_4`

Confirmation triggered: **False**

## Paired diagnostics

```json
{
  "dual_split_4_4_minus_dual_shared_8": {
    "return_per_decision_mean": -0.0033333333333333214,
    "return_per_decision_approx_95ci_low": -0.02423489521011393,
    "return_per_decision_approx_95ci_high": 0.01756822854344729,
    "return_per_decision_wins": 6,
    "final_stable_accuracy_mean": 0.002751243997620817,
    "final_stable_accuracy_approx_95ci_low": -0.0032041185111020337,
    "final_stable_accuracy_approx_95ci_high": 0.008706606506343667,
    "final_stable_accuracy_wins": 6,
    "final_reversed_accuracy_mean": 0.0014347901301200005,
    "final_reversed_accuracy_approx_95ci_low": -0.018996637974586315,
    "final_reversed_accuracy_approx_95ci_high": 0.021866218234826314,
    "final_reversed_accuracy_wins": 5,
    "final_novel_accuracy_mean": -0.028252195430909715,
    "final_novel_accuracy_approx_95ci_low": -0.07649289094224661,
    "final_novel_accuracy_approx_95ci_high": 0.01998850008042718,
    "final_novel_accuracy_wins": 5,
    "buffer_evictions_mean": 3.3,
    "buffer_evictions_approx_95ci_low": 2.1817688968732805,
    "buffer_evictions_approx_95ci_high": 4.418231103126719,
    "buffer_evictions_wins": 20
  },
  "dual_shared_8_minus_dual_shared_4": {
    "return_per_decision_mean": 0.1375,
    "return_per_decision_approx_95ci_low": 0.09283886016780236,
    "return_per_decision_approx_95ci_high": 0.18216113983219767,
    "return_per_decision_wins": 17,
    "final_stable_accuracy_mean": 0.009358858945657167,
    "final_stable_accuracy_approx_95ci_low": -0.006895780191768912,
    "final_stable_accuracy_approx_95ci_high": 0.025613498083083247,
    "final_stable_accuracy_wins": 6,
    "final_reversed_accuracy_mean": 0.1181593067383335,
    "final_reversed_accuracy_approx_95ci_low": 0.06318818774004842,
    "final_reversed_accuracy_approx_95ci_high": 0.17313042573661858,
    "final_reversed_accuracy_wins": 14,
    "final_novel_accuracy_mean": 0.2662603026969255,
    "final_novel_accuracy_approx_95ci_low": 0.17552157381282485,
    "final_novel_accuracy_approx_95ci_high": 0.3569990315810262,
    "final_novel_accuracy_wins": 17,
    "buffer_evictions_mean": -56.1,
    "buffer_evictions_approx_95ci_low": -65.32752096719373,
    "buffer_evictions_approx_95ci_high": -46.87247903280627,
    "buffer_evictions_wins": 0
  },
  "dual_split_2_2_minus_dual_shared_4": {
    "return_per_decision_mean": -0.021166666666666657,
    "return_per_decision_approx_95ci_low": -0.0584647193712725,
    "return_per_decision_approx_95ci_high": 0.016131386037939188,
    "return_per_decision_wins": 7,
    "final_stable_accuracy_mean": 0.0016147296442201088,
    "final_stable_accuracy_approx_95ci_low": -0.028325488677560302,
    "final_stable_accuracy_approx_95ci_high": 0.03155494796600052,
    "final_stable_accuracy_wins": 7,
    "final_reversed_accuracy_mean": -0.06116553530715518,
    "final_reversed_accuracy_approx_95ci_low": -0.14779322953191545,
    "final_reversed_accuracy_approx_95ci_high": 0.025462158917605088,
    "final_reversed_accuracy_wins": 8,
    "final_novel_accuracy_mean": 0.015522427990169924,
    "final_novel_accuracy_approx_95ci_low": -0.08313361447419723,
    "final_novel_accuracy_approx_95ci_high": 0.11417847045453708,
    "final_novel_accuracy_wins": 7,
    "buffer_evictions_mean": 8.9,
    "buffer_evictions_approx_95ci_low": 1.2435716420774892,
    "buffer_evictions_approx_95ci_high": 16.556428357922513,
    "buffer_evictions_wins": 13
  },
  "dual_split_4_4_minus_identity_only_shared_4": {
    "return_per_decision_mean": -0.02416666666666667,
    "return_per_decision_approx_95ci_low": -0.062435147314728864,
    "return_per_decision_approx_95ci_high": 0.014101813981395525,
    "return_per_decision_wins": 10,
    "final_stable_accuracy_mean": 0.007013772135334523,
    "final_stable_accuracy_approx_95ci_low": -0.00015323981492988865,
    "final_stable_accuracy_approx_95ci_high": 0.014180784085598934,
    "final_stable_accuracy_wins": 10,
    "final_reversed_accuracy_mean": -0.027553275142557267,
    "final_reversed_accuracy_approx_95ci_low": -0.05303770653739382,
    "final_reversed_accuracy_approx_95ci_high": -0.0020688437477207125,
    "final_reversed_accuracy_wins": 2,
    "final_novel_accuracy_mean": 0.03457652609262574,
    "final_novel_accuracy_approx_95ci_low": -0.0561095191194987,
    "final_novel_accuracy_approx_95ci_high": 0.12526257130475016,
    "final_novel_accuracy_wins": 11,
    "buffer_evictions_mean": 6.15,
    "buffer_evictions_approx_95ci_low": 3.5084500383297694,
    "buffer_evictions_approx_95ci_high": 8.791549961670231,
    "buffer_evictions_wins": 19
  }
}
```
