# Experiment 008: ideal consequence-target audit

This is a post-development causal diagnostic, not a reopened confirmation gate.
It asks whether the exact target geometry would support the frozen Chevron mechanism.

| Geometry | Return | Final old | Final new | New probe | q calibration | Promotions |
|---|---:|---:|---:|---:|---:|---:|
| Oracle latent | 0.833 +/- 0.033 | 0.973 +/- 0.014 | 0.885 +/- 0.103 | 0.975 +/- 0.075 | 0.216 +/- 0.058 | 3.900 +/- 0.300 |
| Oracle consequence signature | 0.494 +/- 0.101 | 0.804 +/- 0.060 | 0.554 +/- 0.118 | 0.675 +/- 0.225 | 0.102 +/- 0.088 | 2.800 +/- 1.166 |

## Geometry audit

- Competing within-family mean similarity: -0.152
- Competing within-family maximum similarity: 0.959
- Competing contexts above the 0.62 assent boundary: 13.8%
- Matching noisy-event mean similarity: 0.792
- Matching noisy events below the assent boundary: 15.4%

The exact consequence signature is therefore not a sufficient memory-identity geometry.
It sometimes treats distinct contexts as mutually assenting and sometimes moves a noisy
observation too far from its own retained prototype.

## Paired diagnostics

```json
{
  "oracle_consequence_chevron_minus_oracle_latent_chevron": {
    "return_per_decision_mean": -0.3395,
    "return_per_decision_sd": 0.10732949214004944,
    "return_per_decision_wins": 0,
    "return_per_decision_approx_95ci_low": -0.3865392239214733,
    "return_per_decision_approx_95ci_high": -0.2924607760785268,
    "final_old_accuracy_mean": -0.1692308088332391,
    "final_old_accuracy_sd": 0.06301383771451805,
    "final_old_accuracy_wins": 0,
    "final_old_accuracy_approx_95ci_low": -0.19684784086518836,
    "final_old_accuracy_approx_95ci_high": -0.14161377680128984,
    "final_new_accuracy_mean": -0.33183382557499147,
    "final_new_accuracy_sd": 0.18588418701821452,
    "final_new_accuracy_wins": 1,
    "final_new_accuracy_approx_95ci_low": -0.4133011624855337,
    "final_new_accuracy_approx_95ci_high": -0.2503664886644492,
    "residual_calibration_mean": -0.11441265854759912,
    "residual_calibration_sd": 0.08981647048940436,
    "residual_calibration_wins": 1,
    "residual_calibration_approx_95ci_low": -0.15377646231622785,
    "residual_calibration_approx_95ci_high": -0.07504885477897039
  }
}
```
