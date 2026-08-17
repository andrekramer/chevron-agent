# Experiment 011a: pairwise-identity confirmation

- Encoder seeds: [1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209]
- Paired RL lifetimes: 200
- Representation learner: **frozen pairwise temporal contrastive**
- Downstream identity threshold: **fixed at cosine 0.62**

| Condition | Return | Clean accuracy | Retention | Reversed probe | Novel probe | Identity calibration | False revisions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle protected Chevron | 0.546 +/- 0.058 | 0.842 +/- 0.033 | 0.948 +/- 0.022 | 0.986 +/- 0.057 | 0.922 +/- 0.136 | 0.279 +/- 0.033 | 0.090 +/- 0.415 |
| Raw-sensor protected Chevron | 0.360 +/- 0.059 | 0.726 +/- 0.033 | 0.905 +/- 0.067 | 0.949 +/- 0.104 | 0.146 +/- 0.172 | -0.103 +/- 0.142 | 3.060 +/- 2.183 |
| Pairwise-temporal protected Chevron | 0.481 +/- 0.072 | 0.801 +/- 0.042 | 0.926 +/- 0.029 | 0.931 +/- 0.148 | 0.771 +/- 0.236 | 0.223 +/- 0.065 | 0.195 +/- 0.638 |
| Pairwise-temporal direct adaptation | 0.539 +/- 0.050 | 0.837 +/- 0.028 | 0.928 +/- 0.030 | 0.973 +/- 0.078 | 0.667 +/- 0.231 | 0.229 +/- 0.064 | 0.000 +/- 0.000 |

## Learned representation

- Same-identity admission: 0.988 +/- 0.001
- Confusable-change rejection: 0.817 +/- 0.010
- Balanced identity accuracy: 0.902 +/- 0.005
- Latent cosine correlation: 0.794 +/- 0.008

## Frozen confirmation gate

- PASS: `retention_accuracy_at_least_0.90`
- PASS: `reversed_probe_at_least_0.75`
- PASS: `novel_probe_at_least_0.75`
- PASS: `new_promotions_at_least_3`
- PASS: `unique_revisions_at_least_3`
- PASS: `identity_calibration_at_least_0.10`
- PASS: `policy_calibration_at_least_0.10`
- PASS: `false_stable_revisions_at_most_0.25`
- PASS: `no_duplicate_allocations`
- PASS: `no_established_overwrites`
- PASS: `no_under_supported_writes`
- PASS: `same_identity_admission_at_least_0.90`
- PASS: `confusable_change_rejection_at_least_0.80`
- PASS: `balanced_identity_accuracy_at_least_0.85`
- PASS: `return_better_than_raw_sensor`
- PASS: `novel_probe_better_than_raw_sensor`
- PASS: `return_noninferior_to_oracle`
- PASS: `clean_accuracy_noninferior_to_oracle`
- PASS: `retention_noninferior_to_oracle`
- PASS: `return_noninferior_to_direct`
- PASS: `clean_accuracy_noninferior_to_direct`
- PASS: `retention_noninferior_to_direct`

Confirmation passed: **True**

## Paired diagnostics

```json
{
  "pairwise_temporal_protected_minus_raw_sensor_protected": {
    "return_per_decision_mean": 0.120825,
    "return_per_decision_population_sd": 0.07957665722433935,
    "return_per_decision_approx_95ci_low": 0.10979623798629239,
    "return_per_decision_approx_95ci_high": 0.1318537620137076,
    "return_per_decision_wins": 183,
    "clean_accuracy_mean": 0.074575,
    "clean_accuracy_population_sd": 0.04728412127765514,
    "clean_accuracy_approx_95ci_low": 0.0680217551316161,
    "clean_accuracy_approx_95ci_high": 0.08112824486838391,
    "clean_accuracy_wins": 188,
    "retention_accuracy_mean": 0.020200000000000006,
    "retention_accuracy_population_sd": 0.0683974414726165,
    "retention_accuracy_approx_95ci_low": 0.010720598242504969,
    "retention_accuracy_approx_95ci_high": 0.029679401757495043,
    "retention_accuracy_wins": 100,
    "reversed_probe_accuracy_mean": -0.0175,
    "reversed_probe_accuracy_population_sd": 0.16490527584040482,
    "reversed_probe_accuracy_approx_95ci_low": -0.04035470520483693,
    "reversed_probe_accuracy_approx_95ci_high": 0.005354705204836923,
    "reversed_probe_accuracy_wins": 26,
    "novel_probe_accuracy_mean": 0.625,
    "novel_probe_accuracy_population_sd": 0.2850438562747845,
    "novel_probe_accuracy_approx_95ci_low": 0.5854949370333371,
    "novel_probe_accuracy_approx_95ci_high": 0.6645050629666629,
    "novel_probe_accuracy_wins": 189
  },
  "pairwise_temporal_protected_minus_oracle_protected": {
    "return_per_decision_mean": -0.064825,
    "return_per_decision_population_sd": 0.06892908946881571,
    "return_per_decision_approx_95ci_low": -0.07437808441054511,
    "return_per_decision_approx_95ci_high": -0.055271915589454886,
    "return_per_decision_wins": 32,
    "clean_accuracy_mean": -0.04102499999999999,
    "clean_accuracy_population_sd": 0.041646721059406346,
    "clean_accuracy_approx_95ci_low": -0.04679694105955699,
    "clean_accuracy_approx_95ci_high": -0.03525305894044299,
    "clean_accuracy_wins": 31,
    "retention_accuracy_mean": -0.022124999999999985,
    "retention_accuracy_population_sd": 0.021380116346736738,
    "retention_accuracy_approx_95ci_low": -0.025088132949261627,
    "retention_accuracy_approx_95ci_high": -0.019161867050738343,
    "retention_accuracy_wins": 20,
    "reversed_probe_accuracy_mean": -0.055,
    "reversed_probe_accuracy_population_sd": 0.1439618004888797,
    "reversed_probe_accuracy_approx_95ci_low": -0.07495208761007228,
    "reversed_probe_accuracy_approx_95ci_high": -0.035047912389927716,
    "reversed_probe_accuracy_wins": 2,
    "novel_probe_accuracy_mean": -0.15125,
    "novel_probe_accuracy_population_sd": 0.25111936902596743,
    "novel_probe_accuracy_approx_95ci_low": -0.18605336891020752,
    "novel_probe_accuracy_approx_95ci_high": -0.11644663108979247,
    "novel_probe_accuracy_wins": 15
  },
  "pairwise_temporal_protected_minus_pairwise_temporal_direct": {
    "return_per_decision_mean": -0.0578875,
    "return_per_decision_population_sd": 0.05285067259505786,
    "return_per_decision_approx_95ci_low": -0.06521222952051815,
    "return_per_decision_approx_95ci_high": -0.05056277047948186,
    "return_per_decision_wins": 22,
    "clean_accuracy_mean": -0.035706249999999995,
    "clean_accuracy_population_sd": 0.03213948542583562,
    "clean_accuracy_approx_95ci_low": -0.04016055542533709,
    "clean_accuracy_approx_95ci_high": -0.0312519445746629,
    "clean_accuracy_wins": 20,
    "retention_accuracy_mean": -0.002150000000000001,
    "retention_accuracy_population_sd": 0.02640790601316203,
    "retention_accuracy_approx_95ci_low": -0.0058099490460934,
    "retention_accuracy_approx_95ci_high": 0.0015099490460933986,
    "retention_accuracy_wins": 83,
    "reversed_probe_accuracy_mean": -0.04125,
    "reversed_probe_accuracy_population_sd": 0.16351127636955196,
    "reversed_probe_accuracy_approx_95ci_low": -0.06391150673499006,
    "reversed_probe_accuracy_approx_95ci_high": -0.01858849326500994,
    "reversed_probe_accuracy_wins": 15,
    "novel_probe_accuracy_mean": 0.10375,
    "novel_probe_accuracy_population_sd": 0.25651206111994035,
    "novel_probe_accuracy_approx_95ci_low": 0.0681992420966866,
    "novel_probe_accuracy_approx_95ci_high": 0.1393007579033134,
    "novel_probe_accuracy_wins": 89
  }
}
```
