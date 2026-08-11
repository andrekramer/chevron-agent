# Experiment 006a: development self-calibrated gate

- Encoder seeds: 0–1
- Evaluation lifetimes per seed: 10
- Calibrated similarity threshold: 0.561 +/- 0.002
- Calibrated mismatch slope: 120.000 +/- 0.000
- Calibration labels: none

| Method | Return | Final old | Final new | New probe | q calibration | Promotions | N drift | Premature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oracle geometric Chevron | 0.819 +/- 0.038 | 0.970 +/- 0.013 | 0.896 +/- 0.065 | 1.000 +/- 0.000 | 0.211 +/- 0.046 | 4.000 +/- 0.000 | 0.044 +/- 0.003 | 0.000 +/- 0.000 |
| Temporal Chevron, inherited gate | 0.709 +/- 0.062 | 0.936 +/- 0.032 | 0.693 +/- 0.166 | 0.812 +/- 0.222 | 0.137 +/- 0.084 | 3.350 +/- 0.654 | 0.065 +/- 0.010 | 0.000 +/- 0.000 |
| Temporal Chevron, calibrated gate | 0.685 +/- 0.085 | 0.911 +/- 0.048 | 0.679 +/- 0.210 | 0.762 +/- 0.216 | 0.145 +/- 0.081 | 2.700 +/- 0.843 | 0.084 +/- 0.017 | 0.000 +/- 0.000 |
| Temporal content attention, calibrated | 0.661 +/- 0.075 | 0.896 +/- 0.058 | 0.620 +/- 0.193 | 0.750 +/- 0.262 | 0.337 +/- 0.186 | 2.350 +/- 1.014 | 0.109 +/- 0.019 | 0.000 +/- 0.000 |
| Temporal Chevron, calibrated immediate | 0.599 +/- 0.064 | 0.934 +/- 0.045 | 0.365 +/- 0.193 | 0.463 +/- 0.213 | 0.193 +/- 0.070 | 0.000 +/- 0.000 | 0.070 +/- 0.013 | 0.075 +/- 0.025 |

## Frozen confirmation gate

- FAIL: `old_accuracy_at_least_0.95`
- FAIL: `new_accuracy_at_least_0.75`
- PASS: `new_probe_at_least_0.75`
- FAIL: `q_calibration_at_least_0.15`
- FAIL: `promotions_at_least_3`
- FAIL: `return_better_than_inherited`
- PASS: `return_better_than_immediate`
- PASS: `novel_better_than_immediate`
- FAIL: `return_noninferior_to_oracle`
- FAIL: `novel_noninferior_to_oracle`
- PASS: `return_noninferior_to_content`

Confirmation triggered: **False**

## Paired diagnostics

```json
{
  "calibrated_temporal_geometric_chevron_minus_inherited_temporal_geometric_chevron": {
    "return_per_decision_mean": -0.02366666666666667,
    "return_per_decision_sd": 0.08365831369500822,
    "return_per_decision_wins": 7,
    "return_per_decision_approx_95ci_low": -0.0603315392224545,
    "return_per_decision_approx_95ci_high": 0.012998205889121164,
    "final_old_accuracy_mean": -0.02415703618665524,
    "final_old_accuracy_sd": 0.03946959995290677,
    "final_old_accuracy_wins": 7,
    "final_old_accuracy_approx_95ci_low": -0.04145535106038196,
    "final_old_accuracy_approx_95ci_high": -0.0068587213129285216,
    "final_new_accuracy_mean": -0.013970829645676506,
    "final_new_accuracy_sd": 0.19135238874724378,
    "final_new_accuracy_wins": 9,
    "final_new_accuracy_approx_95ci_low": -0.09783471162925388,
    "final_new_accuracy_approx_95ci_high": 0.06989305233790087,
    "residual_calibration_mean": 0.007686551758117421,
    "residual_calibration_sd": 0.06260281536560183,
    "residual_calibration_wins": 13,
    "residual_calibration_approx_95ci_low": -0.01975034178699197,
    "residual_calibration_approx_95ci_high": 0.035123445303226815
  },
  "calibrated_temporal_geometric_chevron_minus_calibrated_temporal_immediate": {
    "return_per_decision_mean": 0.08699999999999998,
    "return_per_decision_sd": 0.08895034425279118,
    "return_per_decision_wins": 16,
    "return_per_decision_approx_95ci_low": 0.04801579279123522,
    "return_per_decision_approx_95ci_high": 0.12598420720876474,
    "final_old_accuracy_mean": -0.022143414294136698,
    "final_old_accuracy_sd": 0.03234966286958318,
    "final_old_accuracy_wins": 5,
    "final_old_accuracy_approx_95ci_low": -0.03632127915835212,
    "final_old_accuracy_approx_95ci_high": -0.007965549429921275,
    "final_new_accuracy_mean": 0.31492985092748704,
    "final_new_accuracy_sd": 0.23274600477200955,
    "final_new_accuracy_wins": 18,
    "final_new_accuracy_approx_95ci_low": 0.21292441684779312,
    "final_new_accuracy_approx_95ci_high": 0.41693528500718097,
    "residual_calibration_mean": -0.047887355647792625,
    "residual_calibration_sd": 0.06532234990302455,
    "residual_calibration_wins": 5,
    "residual_calibration_approx_95ci_low": -0.0765161377550978,
    "residual_calibration_approx_95ci_high": -0.01925857354048745
  },
  "calibrated_temporal_geometric_chevron_minus_oracle_geometric_chevron": {
    "return_per_decision_mean": -0.13366666666666666,
    "return_per_decision_sd": 0.07752682991535993,
    "return_per_decision_wins": 0,
    "return_per_decision_approx_95ci_low": -0.16764429797374522,
    "return_per_decision_approx_95ci_high": -0.09968903535958809,
    "final_old_accuracy_mean": -0.05872208364021621,
    "final_old_accuracy_sd": 0.04906996450607628,
    "final_old_accuracy_wins": 2,
    "final_old_accuracy_approx_95ci_low": -0.08022794379287754,
    "final_old_accuracy_approx_95ci_high": -0.03721622348755487,
    "final_new_accuracy_mean": -0.21657857214394477,
    "final_new_accuracy_sd": 0.2075239637534122,
    "final_new_accuracy_wins": 2,
    "final_new_accuracy_approx_95ci_low": -0.30752995936685984,
    "final_new_accuracy_approx_95ci_high": -0.1256271849210297,
    "residual_calibration_mean": -0.06547051689812444,
    "residual_calibration_sd": 0.07124981967903635,
    "residual_calibration_wins": 3,
    "residual_calibration_approx_95ci_low": -0.09669712717476224,
    "residual_calibration_approx_95ci_high": -0.034243906621486644
  },
  "calibrated_temporal_geometric_chevron_minus_calibrated_temporal_content_attention": {
    "return_per_decision_mean": 0.024833333333333332,
    "return_per_decision_sd": 0.06833418912600614,
    "return_per_decision_wins": 12,
    "return_per_decision_approx_95ci_low": -0.005115445512989692,
    "return_per_decision_approx_95ci_high": 0.05478211217965635,
    "final_old_accuracy_mean": 0.015516119465398631,
    "final_old_accuracy_sd": 0.05062533910885108,
    "final_old_accuracy_wins": 11,
    "final_old_accuracy_approx_95ci_low": -0.006671413662349818,
    "final_old_accuracy_approx_95ci_high": 0.03770365259314708,
    "final_new_accuracy_mean": 0.05919115526957523,
    "final_new_accuracy_sd": 0.13871122283978946,
    "final_new_accuracy_wins": 14,
    "final_new_accuracy_approx_95ci_low": -0.001601718538755327,
    "final_new_accuracy_approx_95ci_high": 0.11998402907790578,
    "residual_calibration_mean": -0.19235416906671796,
    "residual_calibration_sd": 0.1266944135704762,
    "residual_calibration_wins": 1,
    "residual_calibration_approx_95ci_low": -0.24788044400487705,
    "residual_calibration_approx_95ci_high": -0.13682789412855886
  }
}
```

Confirmation was not run because the frozen development gate failed. See
`experiment_006a_development_findings.md` for interpretation.
