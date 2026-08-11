# Experiment 005a: fresh-seed geometric confirmation

- Lifetime seeds: 60000000–60000099
- Fresh lifetimes: 100
- Additional training: none
- Formula and thresholds: frozen from the development diagnostic

| Method | Return | Final old | Final new | New probe | q calibration | Promotions | N drift | Premature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Content attention + buffer | 0.794 +/- 0.082 | 0.981 +/- 0.013 | 0.802 +/- 0.199 | 0.887 +/- 0.188 | 0.457 +/- 0.178 | 3.530 +/- 0.768 | 0.055 +/- 0.010 | 0.000 +/- 0.000 |
| Geometric Chevron + buffer | 0.823 +/- 0.055 | 0.972 +/- 0.014 | 0.896 +/- 0.109 | 0.978 +/- 0.080 | 0.193 +/- 0.068 | 3.920 +/- 0.306 | 0.044 +/- 0.004 | 0.000 +/- 0.000 |
| Geometric Chevron + immediate | 0.648 +/- 0.052 | 0.970 +/- 0.015 | 0.352 +/- 0.124 | 0.295 +/- 0.241 | 0.186 +/- 0.058 | 0.000 +/- 0.000 | 0.041 +/- 0.003 | 0.086 +/- 0.015 |
| Geometric Chevron + coupled write | 0.816 +/- 0.061 | 0.970 +/- 0.016 | 0.878 +/- 0.130 | 0.963 +/- 0.102 | 0.196 +/- 0.069 | 3.860 +/- 0.400 | 0.050 +/- 0.007 | 0.000 +/- 0.000 |

## Frozen confirmation gate

- PASS: `old_accuracy_at_least_0.95`
- PASS: `new_accuracy_at_least_0.75`
- PASS: `new_probe_at_least_0.75`
- PASS: `q_calibration_at_least_0.15`
- PASS: `promotions_at_least_3`
- PASS: `no_premature_writes`
- PASS: `positive_read_write_margin`
- PASS: `buffer_return_better_than_immediate`
- PASS: `buffer_novel_better_than_immediate`
- PASS: `return_noninferior_to_content`
- PASS: `novel_noninferior_to_content`

Confirmed: **True**

## Paired diagnostics

```json
{
  "geometric_chevron_buffer_minus_content_attention_buffer": {
    "return_per_decision_mean": 0.029833333333333337,
    "return_per_decision_sd": 0.07342261935226568,
    "return_per_decision_wins": 67,
    "return_per_decision_approx_95ci_low": 0.015442499940289263,
    "return_per_decision_approx_95ci_high": 0.04422416672637741,
    "final_old_accuracy_mean": -0.008982971565926012,
    "final_old_accuracy_sd": 0.01351991892830811,
    "final_old_accuracy_wins": 17,
    "final_old_accuracy_approx_95ci_low": -0.0116328756758744,
    "final_old_accuracy_approx_95ci_high": -0.006333067455977623,
    "final_new_accuracy_mean": 0.0943023556563419,
    "final_new_accuracy_sd": 0.17343205027397612,
    "final_new_accuracy_wins": 61,
    "final_new_accuracy_approx_95ci_low": 0.060309673802642585,
    "final_new_accuracy_approx_95ci_high": 0.12829503751004123,
    "residual_calibration_mean": -0.2635465820120935,
    "residual_calibration_sd": 0.13461151576780472,
    "residual_calibration_wins": 3,
    "residual_calibration_approx_95ci_low": -0.28993043910258326,
    "residual_calibration_approx_95ci_high": -0.23716272492160378
  },
  "geometric_chevron_buffer_minus_geometric_chevron_immediate": {
    "return_per_decision_mean": 0.17550000000000002,
    "return_per_decision_sd": 0.0680568439375057,
    "return_per_decision_wins": 99,
    "return_per_decision_approx_95ci_low": 0.1621608585882489,
    "return_per_decision_approx_95ci_high": 0.18883914141175112,
    "final_old_accuracy_mean": 0.0018885049021628098,
    "final_old_accuracy_sd": 0.012900658759960139,
    "final_old_accuracy_wins": 44,
    "final_old_accuracy_approx_95ci_low": -0.0006400242147893776,
    "final_old_accuracy_approx_95ci_high": 0.004417034019114997,
    "final_new_accuracy_mean": 0.5438684450248426,
    "final_new_accuracy_sd": 0.14898895932212644,
    "final_new_accuracy_wins": 100,
    "final_new_accuracy_approx_95ci_low": 0.5146666089977058,
    "final_new_accuracy_approx_95ci_high": 0.5730702810519793,
    "residual_calibration_mean": 0.006997099674635365,
    "residual_calibration_sd": 0.054989494456229915,
    "residual_calibration_wins": 53,
    "residual_calibration_approx_95ci_low": -0.003780841238785698,
    "residual_calibration_approx_95ci_high": 0.01777504058805643
  },
  "geometric_chevron_buffer_minus_geometric_chevron_coupled_write": {
    "return_per_decision_mean": 0.0072999999999999975,
    "return_per_decision_sd": 0.03221941819913744,
    "return_per_decision_wins": 46,
    "return_per_decision_approx_95ci_low": 0.0009849940329690584,
    "return_per_decision_approx_95ci_high": 0.013615005967030937,
    "final_old_accuracy_mean": 0.0023544133551778577,
    "final_old_accuracy_sd": 0.009582851479572002,
    "final_old_accuracy_wins": 34,
    "final_old_accuracy_approx_95ci_low": 0.00047617446518174534,
    "final_old_accuracy_approx_95ci_high": 0.00423265224517397,
    "final_new_accuracy_mean": 0.018458871746856686,
    "final_new_accuracy_sd": 0.08936514959828681,
    "final_new_accuracy_wins": 34,
    "final_new_accuracy_approx_95ci_low": 0.000943302425592471,
    "final_new_accuracy_approx_95ci_high": 0.0359744410681209,
    "residual_calibration_mean": -0.0029411343082337183,
    "residual_calibration_sd": 0.01876328419205722,
    "residual_calibration_wins": 24,
    "residual_calibration_approx_95ci_low": -0.006618738009876933,
    "residual_calibration_approx_95ci_high": 0.0007364693934094964
  }
}
```
