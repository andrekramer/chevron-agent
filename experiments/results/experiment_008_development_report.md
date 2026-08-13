# Experiment 008: development consequence geometry

- Encoder seeds: 0–1
- Training steps per objective: 500
- Evaluation lifetimes per seed: 10
- Downstream encoder parameters: 812

| Method | Return | Final old | Final new | New probe | q calibration | Promotions | N drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle geometric Chevron | 0.833 +/- 0.033 | 0.973 +/- 0.014 | 0.885 +/- 0.103 | 0.975 +/- 0.075 | 0.216 +/- 0.058 | 3.900 +/- 0.300 | 0.043 +/- 0.003 |
| Raw-sensor geometric Chevron | 0.397 +/- 0.088 | 0.679 +/- 0.057 | 0.408 +/- 0.161 | 0.500 +/- 0.209 | 0.028 +/- 0.066 | 0.350 +/- 0.477 | 0.131 +/- 0.017 |
| Temporal-contrastive geometric Chevron | 0.699 +/- 0.093 | 0.935 +/- 0.029 | 0.656 +/- 0.215 | 0.775 +/- 0.249 | 0.163 +/- 0.086 | 3.000 +/- 1.095 | 0.067 +/- 0.011 |
| Action-predictive geometric Chevron | 0.678 +/- 0.066 | 0.928 +/- 0.042 | 0.608 +/- 0.162 | 0.750 +/- 0.194 | 0.143 +/- 0.051 | 3.000 +/- 0.837 | 0.069 +/- 0.010 |
| Consequence-metric geometric Chevron | 0.611 +/- 0.090 | 0.893 +/- 0.057 | 0.540 +/- 0.203 | 0.662 +/- 0.253 | 0.154 +/- 0.064 | 2.450 +/- 0.973 | 0.080 +/- 0.015 |
| Consequence-metric content attention | 0.627 +/- 0.094 | 0.875 +/- 0.074 | 0.587 +/- 0.201 | 0.725 +/- 0.249 | 0.395 +/- 0.177 | 2.500 +/- 0.922 | 0.104 +/- 0.020 |

## Consequence diagnostics

- Raw consequence-cosine correlation: 0.107 +/- 0.018
- Encoded consequence-cosine correlation: 0.488 +/- 0.005
- Correlation gain: 0.380 +/- 0.013

## Frozen confirmation gate

- FAIL: `old_accuracy_at_least_0.95`
- FAIL: `new_accuracy_at_least_0.75`
- FAIL: `new_probe_at_least_0.75`
- PASS: `q_calibration_at_least_0.15`
- FAIL: `promotions_at_least_3`
- FAIL: `return_better_than_temporal`
- FAIL: `return_better_than_action_predictive`
- FAIL: `return_noninferior_to_oracle`
- FAIL: `novel_noninferior_to_oracle`
- PASS: `return_noninferior_to_content`
- FAIL: `consequence_correlation_at_least_0.85`
- PASS: `consequence_correlation_gain_at_least_0.20`

Confirmation triggered: **False**

## Paired diagnostics

```json
{
  "consequence_geometric_chevron_minus_temporal_geometric_chevron": {
    "return_per_decision_mean": -0.0875,
    "return_per_decision_sd": 0.08074850857710561,
    "return_per_decision_wins": 2,
    "return_per_decision_approx_95ci_low": -0.12288959423498605,
    "return_per_decision_approx_95ci_high": -0.05211040576501393,
    "final_old_accuracy_mean": -0.042291378105291794,
    "final_old_accuracy_sd": 0.04097470891661757,
    "final_old_accuracy_wins": 2,
    "final_old_accuracy_approx_95ci_low": -0.06024933606647323,
    "final_old_accuracy_approx_95ci_high": -0.024333420144110357,
    "final_new_accuracy_mean": -0.1159987785136932,
    "final_new_accuracy_sd": 0.17954461744556124,
    "final_new_accuracy_wins": 7,
    "final_new_accuracy_approx_95ci_low": -0.19468767655577723,
    "final_new_accuracy_approx_95ci_high": -0.03730988047160916,
    "residual_calibration_mean": -0.009628041657780067,
    "residual_calibration_sd": 0.06770459904741459,
    "residual_calibration_wins": 7,
    "residual_calibration_approx_95ci_low": -0.039300890486219826,
    "residual_calibration_approx_95ci_high": 0.020044807170659693
  },
  "consequence_geometric_chevron_minus_action_predictive_geometric_chevron": {
    "return_per_decision_mean": -0.06633333333333333,
    "return_per_decision_sd": 0.09326375852908117,
    "return_per_decision_wins": 8,
    "return_per_decision_approx_95ci_low": -0.10720797769933098,
    "return_per_decision_approx_95ci_high": -0.025458688967335677,
    "final_old_accuracy_mean": -0.035117860588022236,
    "final_old_accuracy_sd": 0.04275726094569145,
    "final_old_accuracy_wins": 3,
    "final_old_accuracy_approx_95ci_low": -0.053857056421249785,
    "final_old_accuracy_approx_95ci_high": -0.016378664754794686,
    "final_new_accuracy_mean": -0.06732287774385919,
    "final_new_accuracy_sd": 0.21126058804934136,
    "final_new_accuracy_wins": 10,
    "final_new_accuracy_approx_95ci_low": -0.15991191276946093,
    "final_new_accuracy_approx_95ci_high": 0.02526615728174257,
    "residual_calibration_mean": 0.010876837473218114,
    "residual_calibration_sd": 0.06264162191855854,
    "residual_calibration_wins": 9,
    "residual_calibration_approx_95ci_low": -0.01657706379360647,
    "residual_calibration_approx_95ci_high": 0.0383307387400427
  },
  "consequence_geometric_chevron_minus_oracle_geometric_chevron": {
    "return_per_decision_mean": -0.22200000000000003,
    "return_per_decision_sd": 0.09675676382056833,
    "return_per_decision_wins": 0,
    "return_per_decision_approx_95ci_low": -0.26440552143239393,
    "return_per_decision_approx_95ci_high": -0.17959447856760616,
    "final_old_accuracy_mean": -0.08024213388575611,
    "final_old_accuracy_sd": 0.06412411246629686,
    "final_old_accuracy_wins": 0,
    "final_old_accuracy_approx_95ci_low": -0.10834576528216648,
    "final_old_accuracy_approx_95ci_high": -0.05213850248934575,
    "final_new_accuracy_mean": -0.3448678761210609,
    "final_new_accuracy_sd": 0.21423772902502197,
    "final_new_accuracy_wins": 1,
    "final_new_accuracy_approx_95ci_low": -0.43876170070830617,
    "final_new_accuracy_approx_95ci_high": -0.25097405153381563,
    "residual_calibration_mean": -0.06248137670182223,
    "residual_calibration_sd": 0.07087728997616215,
    "residual_calibration_wins": 4,
    "residual_calibration_approx_95ci_low": -0.09354471863756418,
    "residual_calibration_approx_95ci_high": -0.03141803476608028
  },
  "consequence_geometric_chevron_minus_consequence_content_attention": {
    "return_per_decision_mean": -0.015999999999999986,
    "return_per_decision_sd": 0.07435177963238623,
    "return_per_decision_wins": 8,
    "return_per_decision_approx_95ci_low": -0.04858610416719557,
    "return_per_decision_approx_95ci_high": 0.016586104167195595,
    "final_old_accuracy_mean": 0.017348869587711934,
    "final_old_accuracy_sd": 0.048519399948563774,
    "final_old_accuracy_wins": 12,
    "final_old_accuracy_approx_95ci_low": -0.00391569500873579,
    "final_old_accuracy_approx_95ci_high": 0.03861343418415966,
    "final_new_accuracy_mean": -0.0466299951935622,
    "final_new_accuracy_sd": 0.17235632532944092,
    "final_new_accuracy_wins": 7,
    "final_new_accuracy_approx_95ci_low": -0.12216848531214715,
    "final_new_accuracy_approx_95ci_high": 0.028908494925022746,
    "residual_calibration_mean": -0.24138367247390566,
    "residual_calibration_sd": 0.13438805468169931,
    "residual_calibration_wins": 0,
    "residual_calibration_approx_95ci_low": -0.3002818342978244,
    "residual_calibration_approx_95ci_high": -0.18248551064998692
  },
  "consequence_geometric_chevron_minus_raw_sensor_geometric_chevron": {
    "return_per_decision_mean": 0.2145,
    "return_per_decision_sd": 0.11315401190128756,
    "return_per_decision_wins": 20,
    "return_per_decision_approx_95ci_low": 0.16490806774253255,
    "return_per_decision_approx_95ci_high": 0.2640919322574674,
    "final_old_accuracy_mean": 0.21394084640798408,
    "final_old_accuracy_sd": 0.07538708395902056,
    "final_old_accuracy_wins": 20,
    "final_old_accuracy_approx_95ci_low": 0.1809010001138447,
    "final_old_accuracy_approx_95ci_high": 0.24698069270212347,
    "final_new_accuracy_mean": 0.1322432675143197,
    "final_new_accuracy_sd": 0.24412097234358848,
    "final_new_accuracy_wins": 14,
    "final_new_accuracy_approx_95ci_low": 0.025252534091172146,
    "final_new_accuracy_approx_95ci_high": 0.23923400093746724,
    "residual_calibration_mean": 0.12569688910148288,
    "residual_calibration_sd": 0.07444395116344057,
    "residual_calibration_wins": 19,
    "residual_calibration_approx_95ci_low": 0.09307038897971787,
    "residual_calibration_approx_95ci_high": 0.1583233892232479
  }
}
```
