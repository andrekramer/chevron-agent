# Experiment 006: development temporal-contrastive geometry

- Encoder seeds: 0–1
- Contrastive steps per seed: 500
- Evaluation lifetimes per seed: 10
- Encoder parameters: 812
- Downstream Chevron parameters: zero

| Method | Return | Final old | Final new | New probe | q calibration | Promotions | N drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle geometric Chevron | 0.819 +/- 0.050 | 0.972 +/- 0.016 | 0.890 +/- 0.102 | 0.963 +/- 0.089 | 0.189 +/- 0.058 | 3.850 +/- 0.357 | 0.046 +/- 0.004 |
| Raw-sensor geometric Chevron | 0.346 +/- 0.100 | 0.685 +/- 0.088 | 0.327 +/- 0.189 | 0.388 +/- 0.279 | -0.004 +/- 0.061 | 0.500 +/- 0.592 | 0.135 +/- 0.015 |
| Random-encoder geometric Chevron | -0.045 +/- 0.055 | 0.423 +/- 0.065 | 0.164 +/- 0.103 | 0.150 +/- 0.166 | -0.001 +/- 0.008 | 0.000 +/- 0.000 | 0.078 +/- 0.019 |
| Temporal-contrastive geometric Chevron | 0.732 +/- 0.058 | 0.937 +/- 0.025 | 0.727 +/- 0.162 | 0.887 +/- 0.147 | 0.135 +/- 0.097 | 3.650 +/- 0.654 | 0.064 +/- 0.007 |
| Temporal-contrastive content attention | 0.732 +/- 0.085 | 0.942 +/- 0.036 | 0.702 +/- 0.171 | 0.775 +/- 0.208 | 0.385 +/- 0.181 | 3.000 +/- 0.894 | 0.085 +/- 0.015 |

## Representation diagnostics

- Temporal positive cosine: 0.760 +/- 0.001
- Temporal negative cosine: 0.008 +/- 0.002
- Temporal cosine gap: 0.752 +/- 0.003
- Raw/latent cosine correlation: 0.453 +/- 0.004
- Encoded/latent cosine correlation: 0.784 +/- 0.005

## Frozen confirmation gate

- FAIL: `old_accuracy_at_least_0.95`
- FAIL: `new_accuracy_at_least_0.75`
- PASS: `new_probe_at_least_0.75`
- FAIL: `q_calibration_at_least_0.15`
- PASS: `promotions_at_least_3`
- PASS: `return_better_than_raw`
- PASS: `return_better_than_random`
- FAIL: `return_noninferior_to_oracle`
- FAIL: `novel_noninferior_to_oracle`
- PASS: `return_noninferior_to_content`
- PASS: `temporal_cosine_gap_at_least_0.30`
- PASS: `latent_cosine_correlation_at_least_0.60`

Confirmation triggered: **False**

## Paired diagnostics

```json
{
  "temporal_geometric_chevron_minus_raw_sensor_geometric_chevron": {
    "return_per_decision_mean": 0.38583333333333336,
    "return_per_decision_sd": 0.11511943962735816,
    "return_per_decision_wins": 20,
    "return_per_decision_approx_95ci_low": 0.335380014395796,
    "return_per_decision_approx_95ci_high": 0.4362866522708707,
    "final_old_accuracy_mean": 0.25125297036357974,
    "final_old_accuracy_sd": 0.09439562808241843,
    "final_old_accuracy_wins": 20,
    "final_old_accuracy_approx_95ci_low": 0.2098822622940489,
    "final_old_accuracy_approx_95ci_high": 0.2926236784331106,
    "final_new_accuracy_mean": 0.39947637316667867,
    "final_new_accuracy_sd": 0.20266112266622702,
    "final_new_accuracy_wins": 19,
    "final_new_accuracy_approx_95ci_low": 0.3106562200177697,
    "final_new_accuracy_approx_95ci_high": 0.4882965263155876,
    "residual_calibration_mean": 0.13975280527305942,
    "residual_calibration_sd": 0.12324459271456172,
    "residual_calibration_wins": 17,
    "residual_calibration_approx_95ci_low": 0.0857384809879285,
    "residual_calibration_approx_95ci_high": 0.19376712955819034
  },
  "temporal_geometric_chevron_minus_random_encoder_geometric_chevron": {
    "return_per_decision_mean": 0.7766666666666666,
    "return_per_decision_sd": 0.08469415219012867,
    "return_per_decision_wins": 20,
    "return_per_decision_approx_95ci_low": 0.7395478178742739,
    "return_per_decision_approx_95ci_high": 0.8137855154590593,
    "final_old_accuracy_mean": 0.5133531668893714,
    "final_old_accuracy_sd": 0.07010636729390607,
    "final_old_accuracy_wins": 20,
    "final_old_accuracy_approx_95ci_low": 0.48262769671612193,
    "final_old_accuracy_approx_95ci_high": 0.5440786370626208,
    "final_new_accuracy_mean": 0.562824689106347,
    "final_new_accuracy_sd": 0.19994648287313246,
    "final_new_accuracy_wins": 20,
    "final_new_accuracy_approx_95ci_low": 0.47519427930334796,
    "final_new_accuracy_approx_95ci_high": 0.650455098909346,
    "residual_calibration_mean": 0.1366883600344012,
    "residual_calibration_sd": 0.1009007683784201,
    "residual_calibration_wins": 19,
    "residual_calibration_approx_95ci_low": 0.09246664852748393,
    "residual_calibration_approx_95ci_high": 0.1809100715413185
  },
  "temporal_geometric_chevron_minus_oracle_geometric_chevron": {
    "return_per_decision_mean": -0.08649999999999998,
    "return_per_decision_sd": 0.08076733605300732,
    "return_per_decision_wins": 4,
    "return_per_decision_approx_95ci_low": -0.12189784574011439,
    "return_per_decision_approx_95ci_high": -0.051102154259885575,
    "final_old_accuracy_mean": -0.03586579551045972,
    "final_old_accuracy_sd": 0.024413000499824748,
    "final_old_accuracy_wins": 0,
    "final_old_accuracy_approx_95ci_low": -0.046565264726319236,
    "final_old_accuracy_approx_95ci_high": -0.025166326294600205,
    "final_new_accuracy_mean": -0.1632158331341747,
    "final_new_accuracy_sd": 0.21574605527205476,
    "final_new_accuracy_wins": 3,
    "final_new_accuracy_approx_95ci_low": -0.25777071084546,
    "final_new_accuracy_approx_95ci_high": -0.06866095542288941,
    "residual_calibration_mean": -0.05330858647759036,
    "residual_calibration_sd": 0.09597011770367794,
    "residual_calibration_wins": 7,
    "residual_calibration_approx_95ci_low": -0.09536934504843003,
    "residual_calibration_approx_95ci_high": -0.011247827906750697
  },
  "temporal_geometric_chevron_minus_temporal_content_attention": {
    "return_per_decision_mean": 0.0,
    "return_per_decision_sd": 0.10744235500547634,
    "return_per_decision_wins": 8,
    "return_per_decision_approx_95ci_low": -0.04708868825316234,
    "return_per_decision_approx_95ci_high": 0.04708868825316234,
    "final_old_accuracy_mean": -0.00556004643875993,
    "final_old_accuracy_sd": 0.03656872311915898,
    "final_old_accuracy_wins": 8,
    "final_old_accuracy_approx_95ci_low": -0.021586995984742223,
    "final_old_accuracy_approx_95ci_high": 0.010466903107222363,
    "final_new_accuracy_mean": 0.02471454701345244,
    "final_new_accuracy_sd": 0.259176933601704,
    "final_new_accuracy_wins": 8,
    "final_new_accuracy_approx_95ci_low": -0.08887475236628603,
    "final_new_accuracy_approx_95ci_high": 0.1383038463931909,
    "residual_calibration_mean": -0.249319318426882,
    "residual_calibration_sd": 0.13407971232491397,
    "residual_calibration_wins": 0,
    "residual_calibration_approx_95ci_low": -0.3080823432546583,
    "residual_calibration_approx_95ci_high": -0.1905562935991057
  }
}
```

Confirmation was not run because the frozen development gate failed. See
`experiment_006_development_findings.md` for interpretation.
