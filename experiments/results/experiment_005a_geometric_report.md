# Experiment 005a: geometric gate isolation

- Evaluation: exact 20 Experiment 005 development lifetimes
- Additional training: none
- Cosine similarity threshold: 0.62
- Half-cosine mismatch threshold: 0.19
- Gate slope: 40.0
- Write threshold margin: 0.05

| Method | Return | Final old | Final new | Old probe | New probe | q calibration | Promotions | N drift | Premature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Content attention + buffer | 0.788 +/- 0.051 | 0.982 +/- 0.013 | 0.821 +/- 0.156 | 1.000 +/- 0.000 | 0.887 +/- 0.167 | 0.457 +/- 0.137 | 3.550 +/- 0.669 | 0.054 +/- 0.007 | 0.000 +/- 0.000 |
| Learned Chevron + retrospective + buffer | 0.512 +/- 0.083 | 0.806 +/- 0.054 | 0.357 +/- 0.205 | 0.812 +/- 0.093 | 0.438 +/- 0.236 | 0.030 +/- 0.049 | 0.550 +/- 0.589 | 0.135 +/- 0.020 | 0.000 +/- 0.000 |
| Geometric Chevron + buffer | 0.828 +/- 0.054 | 0.976 +/- 0.012 | 0.912 +/- 0.086 | 1.000 +/- 0.000 | 0.975 +/- 0.109 | 0.201 +/- 0.069 | 3.900 +/- 0.436 | 0.043 +/- 0.004 | 0.000 +/- 0.000 |
| Geometric Chevron + immediate | 0.636 +/- 0.036 | 0.974 +/- 0.015 | 0.329 +/- 0.129 | 1.000 +/- 0.000 | 0.263 +/- 0.216 | 0.200 +/- 0.060 | 0.000 +/- 0.000 | 0.039 +/- 0.003 | 0.089 +/- 0.014 |
| Geometric Chevron + coupled write | 0.832 +/- 0.047 | 0.975 +/- 0.016 | 0.916 +/- 0.057 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.209 +/- 0.065 | 4.000 +/- 0.000 | 0.048 +/- 0.006 | 0.000 +/- 0.000 |

## Frozen diagnostic gate

- PASS: `old_accuracy_at_least_0.95`
- PASS: `new_accuracy_at_least_0.75`
- PASS: `new_probe_at_least_0.75`
- PASS: `q_calibration_at_least_0.15`
- PASS: `promotions_at_least_3`
- PASS: `no_premature_writes`
- PASS: `positive_read_write_margin`
- PASS: `return_within_0.05_of_content`

Geometric gate viable: **True**

## Paired diagnostics

```json
{
  "geometric_chevron_buffer_minus_content_attention_buffer": {
    "return_per_decision_mean": 0.03916666666666667,
    "return_per_decision_sd": 0.08151973893639079,
    "return_per_decision_wins": 13,
    "return_per_decision_approx_95ci_low": 0.0034390658237846497,
    "return_per_decision_approx_95ci_high": 0.0748942675095487,
    "final_old_accuracy_mean": -0.006079958615245578,
    "final_old_accuracy_sd": 0.012134100830206367,
    "final_old_accuracy_wins": 1,
    "final_old_accuracy_approx_95ci_low": -0.011397962778472479,
    "final_old_accuracy_approx_95ci_high": -0.0007619544520186773,
    "final_new_accuracy_mean": 0.09133742000732555,
    "final_new_accuracy_sd": 0.1800786979919881,
    "final_new_accuracy_wins": 12,
    "final_new_accuracy_approx_95ci_low": 0.012414450845416453,
    "final_new_accuracy_approx_95ci_high": 0.17026038916923464,
    "residual_calibration_mean": -0.25549019508183834,
    "residual_calibration_sd": 0.11327868918594644,
    "residual_calibration_wins": 0,
    "residual_calibration_approx_95ci_low": -0.30513676956852026,
    "residual_calibration_approx_95ci_high": -0.20584362059515643
  },
  "geometric_chevron_buffer_minus_chevron_retrospective_buffer": {
    "return_per_decision_mean": 0.3151666666666667,
    "return_per_decision_sd": 0.10264613032245999,
    "return_per_decision_wins": 20,
    "return_per_decision_approx_95ci_low": 0.2701800165611154,
    "return_per_decision_approx_95ci_high": 0.360153316772218,
    "final_old_accuracy_mean": 0.1703736815655869,
    "final_old_accuracy_sd": 0.0529081479651805,
    "final_old_accuracy_wins": 20,
    "final_old_accuracy_approx_95ci_low": 0.1471856633444898,
    "final_old_accuracy_approx_95ci_high": 0.19356169978668403,
    "final_new_accuracy_mean": 0.5554053995254464,
    "final_new_accuracy_sd": 0.2555760404454251,
    "final_new_accuracy_wins": 19,
    "final_new_accuracy_approx_95ci_low": 0.44339426115363,
    "final_new_accuracy_approx_95ci_high": 0.6674165378972629,
    "residual_calibration_mean": 0.17135348630521544,
    "residual_calibration_sd": 0.06226338825847731,
    "residual_calibration_wins": 20,
    "residual_calibration_approx_95ci_low": 0.1440653532487536,
    "residual_calibration_approx_95ci_high": 0.19864161936167726
  },
  "geometric_chevron_buffer_minus_geometric_chevron_immediate": {
    "return_per_decision_mean": 0.1915,
    "return_per_decision_sd": 0.0639670510945361,
    "return_per_decision_wins": 20,
    "return_per_decision_approx_95ci_low": 0.16346520378475335,
    "return_per_decision_approx_95ci_high": 0.21953479621524666,
    "final_old_accuracy_mean": 0.002276697478430073,
    "final_old_accuracy_sd": 0.010348475063412224,
    "final_old_accuracy_wins": 7,
    "final_old_accuracy_approx_95ci_low": -0.0022587216877991575,
    "final_old_accuracy_approx_95ci_high": 0.006812116644659303,
    "final_new_accuracy_mean": 0.5836199152484919,
    "final_new_accuracy_sd": 0.16734718756979827,
    "final_new_accuracy_wins": 20,
    "final_new_accuracy_approx_95ci_low": 0.5102767765475944,
    "final_new_accuracy_approx_95ci_high": 0.6569630539493894,
    "residual_calibration_mean": 0.0009222284285824889,
    "residual_calibration_sd": 0.04373291150492662,
    "residual_calibration_wins": 11,
    "residual_calibration_approx_95ci_low": -0.018244565115301224,
    "residual_calibration_approx_95ci_high": 0.020089021972466203
  },
  "geometric_chevron_buffer_minus_geometric_chevron_coupled_write": {
    "return_per_decision_mean": -0.004333333333333323,
    "return_per_decision_sd": 0.030913953396805846,
    "return_per_decision_wins": 8,
    "return_per_decision_approx_95ci_low": -0.017881970778042926,
    "return_per_decision_approx_95ci_high": 0.009215304111376282,
    "final_old_accuracy_mean": 0.0011084949035835678,
    "final_old_accuracy_sd": 0.012721733011587001,
    "final_old_accuracy_wins": 4,
    "final_old_accuracy_approx_95ci_low": -0.004467050418296715,
    "final_old_accuracy_approx_95ci_high": 0.006684040225463851,
    "final_new_accuracy_mean": -0.003205044486280284,
    "final_new_accuracy_sd": 0.06638338042841105,
    "final_new_accuracy_wins": 5,
    "final_new_accuracy_approx_95ci_low": -0.0322988437242549,
    "final_new_accuracy_approx_95ci_high": 0.02588875475169433,
    "residual_calibration_mean": -0.008260869349464956,
    "residual_calibration_sd": 0.0222622514878175,
    "residual_calibration_wins": 2,
    "residual_calibration_approx_95ci_low": -0.018017731250620284,
    "residual_calibration_approx_95ci_high": 0.0014959925516903737
  }
}
```
