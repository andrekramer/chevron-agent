# Experiment 011: persistence-derived identity development

- Training seeds: [0, 1]
- RL lifetimes per encoder: 10
- Downstream gate threshold: **fixed at cosine 0.62**
- Policy mechanism: **frozen protected retrospective Chevron**

| Condition | Return | Clean accuracy | Retention | Reversed probe | Novel probe | Identity calibration | New IDs | Revisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oracle protected Chevron | 0.565 +/- 0.067 | 0.855 +/- 0.035 | 0.948 +/- 0.033 | 1.000 +/- 0.000 | 0.963 +/- 0.089 | 0.288 +/- 0.026 | 3.850 +/- 0.357 | 4.000 +/- 0.000 |
| Raw-sensor protected Chevron | 0.331 +/- 0.047 | 0.711 +/- 0.028 | 0.903 +/- 0.057 | 0.963 +/- 0.089 | 0.087 +/- 0.143 | -0.151 +/- 0.090 | 0.300 +/- 0.458 | 3.950 +/- 0.218 |
| Pairwise-temporal protected Chevron | 0.491 +/- 0.050 | 0.808 +/- 0.024 | 0.928 +/- 0.037 | 0.938 +/- 0.108 | 0.863 +/- 0.201 | 0.234 +/- 0.069 | 3.600 +/- 0.663 | 3.800 +/- 0.400 |
| Multi-view-temporal protected Chevron | 0.479 +/- 0.049 | 0.801 +/- 0.026 | 0.927 +/- 0.040 | 0.963 +/- 0.119 | 0.787 +/- 0.198 | 0.231 +/- 0.069 | 3.550 +/- 0.805 | 3.900 +/- 0.300 |
| Hard-persistence protected Chevron | 0.470 +/- 0.057 | 0.795 +/- 0.033 | 0.921 +/- 0.043 | 0.925 +/- 0.179 | 0.738 +/- 0.201 | 0.232 +/- 0.069 | 3.500 +/- 0.975 | 3.800 +/- 0.510 |
| Hard-persistence direct adaptation | 0.531 +/- 0.052 | 0.833 +/- 0.026 | 0.927 +/- 0.036 | 0.963 +/- 0.089 | 0.713 +/- 0.213 | 0.236 +/- 0.073 | 3.900 +/- 0.539 | 0.000 +/- 0.000 |

## Representation diagnostics

| Representation | Same admitted | Confusable rejected | Balanced accuracy | Gap | Latent correlation |
|---|---:|---:|---:|---:|---:|
| raw_sensor | 0.998 +/- 0.000 | 0.161 +/- 0.002 | 0.580 +/- 0.001 | 0.186 +/- 0.000 | 0.477 +/- 0.011 |
| pairwise | 0.986 +/- 0.001 | 0.824 +/- 0.011 | 0.905 +/- 0.005 | 0.431 +/- 0.004 | 0.802 +/- 0.004 |
| multiview | 0.984 +/- 0.001 | 0.829 +/- 0.001 | 0.907 +/- 0.000 | 0.434 +/- 0.002 | 0.778 +/- 0.005 |
| hard_persistence | 0.985 +/- 0.002 | 0.824 +/- 0.002 | 0.904 +/- 0.000 | 0.435 +/- 0.000 | 0.792 +/- 0.005 |

## Frozen gate

- PASS: `retention_accuracy_at_least_0.90`
- PASS: `reversed_probe_at_least_0.75`
- FAIL: `novel_probe_at_least_0.75`
- PASS: `new_promotions_at_least_3`
- PASS: `unique_revisions_at_least_3`
- PASS: `identity_calibration_at_least_0.10`
- PASS: `policy_calibration_at_least_0.10`
- PASS: `no_duplicate_allocations`
- PASS: `no_established_overwrites`
- PASS: `no_under_supported_writes`
- PASS: `same_identity_admission_at_least_0.90`
- PASS: `confusable_change_rejection_at_least_0.80`
- PASS: `balanced_identity_accuracy_at_least_0.85`
- PASS: `return_better_than_raw_sensor`
- FAIL: `return_better_than_pairwise_temporal`
- FAIL: `return_noninferior_to_multiview`
- FAIL: `return_noninferior_to_oracle`
- PASS: `clean_accuracy_noninferior_to_oracle`
- PASS: `retention_noninferior_to_oracle`
- PASS: `return_noninferior_to_direct`
- PASS: `clean_accuracy_noninferior_to_direct`
- PASS: `retention_noninferior_to_direct`

Overall development result: **FAIL**

## Paired diagnostics

```json
{
  "hard_persistence_protected_minus_raw_sensor_protected": {
    "return_per_decision_mean": 0.139125,
    "return_per_decision_population_sd": 0.0760890719814613,
    "return_per_decision_approx_95ci_low": 0.10577749389009727,
    "return_per_decision_approx_95ci_high": 0.17247250610990272,
    "return_per_decision_wins": 20,
    "clean_accuracy_mean": 0.08343750000000001,
    "clean_accuracy_population_sd": 0.043719408947857474,
    "clean_accuracy_approx_95ci_low": 0.06427662421266972,
    "clean_accuracy_approx_95ci_high": 0.1025983757873303,
    "clean_accuracy_wins": 19,
    "retention_accuracy_mean": 0.018999999999999996,
    "retention_accuracy_population_sd": 0.0636513943287969,
    "retention_accuracy_approx_95ci_low": -0.00889645353803956,
    "retention_accuracy_approx_95ci_high": 0.04689645353803955,
    "retention_accuracy_wins": 10,
    "reversed_probe_accuracy_mean": -0.0375,
    "reversed_probe_accuracy_population_sd": 0.19803724397193576,
    "reversed_probe_accuracy_approx_95ci_low": -0.1242936489611999,
    "reversed_probe_accuracy_approx_95ci_high": 0.049293648961199914,
    "reversed_probe_accuracy_wins": 2,
    "novel_probe_accuracy_mean": 0.65,
    "novel_probe_accuracy_population_sd": 0.24238399287081644,
    "novel_probe_accuracy_approx_95ci_low": 0.5437705313954739,
    "novel_probe_accuracy_approx_95ci_high": 0.7562294686045261,
    "novel_probe_accuracy_wins": 19
  },
  "hard_persistence_protected_minus_pairwise_temporal_protected": {
    "return_per_decision_mean": -0.020750000000000005,
    "return_per_decision_population_sd": 0.050392831831521434,
    "return_per_decision_approx_95ci_low": -0.04283563232058345,
    "return_per_decision_approx_95ci_high": 0.0013356323205834346,
    "return_per_decision_wins": 4,
    "clean_accuracy_mean": -0.01324999999999999,
    "clean_accuracy_population_sd": 0.03175196844291705,
    "clean_accuracy_approx_95ci_low": -0.027165913732126963,
    "clean_accuracy_approx_95ci_high": 0.0006659137321269829,
    "clean_accuracy_wins": 6,
    "retention_accuracy_mean": -0.007000000000000001,
    "retention_accuracy_population_sd": 0.01873499399519519,
    "retention_accuracy_approx_95ci_low": -0.015210973145736136,
    "retention_accuracy_approx_95ci_high": 0.0012109731457361338,
    "retention_accuracy_wins": 7,
    "reversed_probe_accuracy_mean": -0.0125,
    "reversed_probe_accuracy_population_sd": 0.14737282653189496,
    "reversed_probe_accuracy_approx_95ci_low": -0.07708898899967392,
    "reversed_probe_accuracy_approx_95ci_high": 0.05208898899967393,
    "reversed_probe_accuracy_wins": 2,
    "novel_probe_accuracy_mean": -0.125,
    "novel_probe_accuracy_population_sd": 0.2795084971874737,
    "novel_probe_accuracy_approx_95ci_low": -0.2475,
    "novel_probe_accuracy_approx_95ci_high": -0.002500000000000016,
    "novel_probe_accuracy_wins": 4
  },
  "hard_persistence_protected_minus_multiview_temporal_protected": {
    "return_per_decision_mean": -0.009000000000000011,
    "return_per_decision_population_sd": 0.053117793628877315,
    "return_per_decision_approx_95ci_low": -0.03227989948431909,
    "return_per_decision_approx_95ci_high": 0.01427989948431907,
    "return_per_decision_wins": 7,
    "clean_accuracy_mean": -0.006749999999999978,
    "clean_accuracy_population_sd": 0.02998020180052163,
    "clean_accuracy_approx_95ci_low": -0.019889402764205054,
    "clean_accuracy_approx_95ci_high": 0.006389402764205099,
    "clean_accuracy_wins": 6,
    "retention_accuracy_mean": -0.006,
    "retention_accuracy_population_sd": 0.020772578077840987,
    "retention_accuracy_approx_95ci_low": -0.015103983743394975,
    "retention_accuracy_approx_95ci_high": 0.003103983743394975,
    "retention_accuracy_wins": 8,
    "reversed_probe_accuracy_mean": -0.0375,
    "reversed_probe_accuracy_population_sd": 0.11924240017711821,
    "reversed_probe_accuracy_approx_95ci_low": -0.08976028606886877,
    "reversed_probe_accuracy_approx_95ci_high": 0.014760286068868776,
    "reversed_probe_accuracy_wins": 1,
    "novel_probe_accuracy_mean": -0.05,
    "novel_probe_accuracy_population_sd": 0.2806243040080456,
    "novel_probe_accuracy_approx_95ci_low": -0.1729890239005091,
    "novel_probe_accuracy_approx_95ci_high": 0.0729890239005091,
    "novel_probe_accuracy_wins": 5
  },
  "hard_persistence_protected_minus_oracle_protected": {
    "return_per_decision_mean": -0.094625,
    "return_per_decision_population_sd": 0.07571028249187821,
    "return_per_decision_approx_95ci_low": -0.12780649429652016,
    "return_per_decision_approx_95ci_high": -0.06144350570347984,
    "return_per_decision_wins": 2,
    "clean_accuracy_mean": -0.06018749999999999,
    "clean_accuracy_population_sd": 0.04484221469497241,
    "clean_accuracy_approx_95ci_low": -0.07984046710264125,
    "clean_accuracy_approx_95ci_high": -0.04053453289735873,
    "clean_accuracy_wins": 0,
    "retention_accuracy_mean": -0.026249999999999996,
    "retention_accuracy_population_sd": 0.01687268502639695,
    "retention_accuracy_approx_95ci_low": -0.033644780253665414,
    "retention_accuracy_approx_95ci_high": -0.01885521974633458,
    "retention_accuracy_wins": 0,
    "reversed_probe_accuracy_mean": -0.075,
    "reversed_probe_accuracy_population_sd": 0.17853571071357124,
    "reversed_probe_accuracy_approx_95ci_low": -0.15324672517108942,
    "reversed_probe_accuracy_approx_95ci_high": 0.0032467251710894263,
    "reversed_probe_accuracy_wins": 0,
    "novel_probe_accuracy_mean": -0.225,
    "novel_probe_accuracy_population_sd": 0.22220486043288973,
    "novel_probe_accuracy_approx_95ci_low": -0.3223855738803238,
    "novel_probe_accuracy_approx_95ci_high": -0.12761442611967622,
    "novel_probe_accuracy_wins": 1
  },
  "hard_persistence_protected_minus_hard_persistence_direct": {
    "return_per_decision_mean": -0.061250000000000006,
    "return_per_decision_population_sd": 0.037253355553560547,
    "return_per_decision_approx_95ci_low": -0.07757700293991522,
    "return_per_decision_approx_95ci_high": -0.04492299706008479,
    "return_per_decision_wins": 1,
    "clean_accuracy_mean": -0.038375,
    "clean_accuracy_population_sd": 0.022227868206375526,
    "clean_accuracy_approx_95ci_low": -0.04811679276365495,
    "clean_accuracy_approx_95ci_high": -0.02863320723634505,
    "clean_accuracy_wins": 0,
    "retention_accuracy_mean": -0.005250000000000005,
    "retention_accuracy_population_sd": 0.02736215452043204,
    "retention_accuracy_approx_95ci_low": -0.017241992953633687,
    "retention_accuracy_approx_95ci_high": 0.006741992953633678,
    "retention_accuracy_wins": 8,
    "reversed_probe_accuracy_mean": -0.0375,
    "reversed_probe_accuracy_population_sd": 0.16345871038277526,
    "reversed_probe_accuracy_approx_95ci_low": -0.10913893843434588,
    "reversed_probe_accuracy_approx_95ci_high": 0.03413893843434588,
    "reversed_probe_accuracy_wins": 2,
    "novel_probe_accuracy_mean": 0.025,
    "novel_probe_accuracy_population_sd": 0.27271780286589287,
    "novel_probe_accuracy_approx_95ci_low": -0.09452384699297459,
    "novel_probe_accuracy_approx_95ci_high": 0.14452384699297458,
    "novel_probe_accuracy_wins": 8
  }
}
```
