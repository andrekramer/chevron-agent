# Experiment 007: development action-conditioned prediction

- Encoder seeds: 0–1
- Training steps per objective: 500
- Evaluation lifetimes per seed: 10
- Downstream encoder parameters: 812
- Discarded predictor parameters: 624

| Method | Return | Final old | Final new | New probe | q calibration | Promotions | N drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle geometric Chevron | 0.828 +/- 0.037 | 0.964 +/- 0.019 | 0.908 +/- 0.096 | 0.988 +/- 0.054 | 0.194 +/- 0.060 | 3.950 +/- 0.218 | 0.044 +/- 0.003 |
| Raw-sensor geometric Chevron | 0.350 +/- 0.073 | 0.639 +/- 0.091 | 0.363 +/- 0.165 | 0.438 +/- 0.222 | 0.037 +/- 0.091 | 0.200 +/- 0.400 | 0.134 +/- 0.014 |
| Temporal-contrastive geometric Chevron | 0.738 +/- 0.074 | 0.936 +/- 0.038 | 0.753 +/- 0.162 | 0.850 +/- 0.184 | 0.149 +/- 0.071 | 3.550 +/- 0.589 | 0.063 +/- 0.009 |
| Action-predictive geometric Chevron | 0.700 +/- 0.079 | 0.939 +/- 0.043 | 0.632 +/- 0.166 | 0.738 +/- 0.201 | 0.123 +/- 0.071 | 3.000 +/- 0.837 | 0.067 +/- 0.012 |
| Action-predictive content attention | 0.682 +/- 0.075 | 0.919 +/- 0.054 | 0.613 +/- 0.174 | 0.713 +/- 0.227 | 0.321 +/- 0.152 | 2.700 +/- 0.954 | 0.091 +/- 0.020 |

## Transition diagnostics

- Predicted-next cosine: 0.666 +/- 0.004
- Permuted-next cosine: 0.023 +/- 0.003
- Transition cosine gap: 0.643 +/- 0.001
- Raw transition-cosine correlation: 0.440 +/- 0.021
- Encoded transition-cosine correlation: 0.670 +/- 0.003

## Frozen confirmation gate

- FAIL: `old_accuracy_at_least_0.95`
- FAIL: `new_accuracy_at_least_0.75`
- FAIL: `new_probe_at_least_0.75`
- FAIL: `q_calibration_at_least_0.15`
- PASS: `promotions_at_least_3`
- FAIL: `return_better_than_temporal`
- FAIL: `return_noninferior_to_oracle`
- FAIL: `novel_noninferior_to_oracle`
- PASS: `return_noninferior_to_content`
- PASS: `transition_cosine_gap_at_least_0.30`
- FAIL: `transition_correlation_at_least_0.80`

Confirmation triggered: **False**

## Paired diagnostics

```json
{
  "action_predictive_geometric_chevron_minus_temporal_geometric_chevron": {
    "return_per_decision_mean": -0.037833333333333316,
    "return_per_decision_sd": 0.06685302899181246,
    "return_per_decision_wins": 6,
    "return_per_decision_approx_95ci_low": -0.06713296512951486,
    "return_per_decision_approx_95ci_high": -0.008533701537151769,
    "final_old_accuracy_mean": 0.002902107123834785,
    "final_old_accuracy_sd": 0.04785586150732946,
    "final_old_accuracy_wins": 10,
    "final_old_accuracy_approx_95ci_low": -0.018071648928797247,
    "final_old_accuracy_approx_95ci_high": 0.023875863176466814,
    "final_new_accuracy_mean": -0.12076031003432404,
    "final_new_accuracy_sd": 0.1397457663959971,
    "final_new_accuracy_wins": 3,
    "final_new_accuracy_approx_95ci_low": -0.18200659254725807,
    "final_new_accuracy_approx_95ci_high": -0.05951402752139001,
    "residual_calibration_mean": -0.02573073861792292,
    "residual_calibration_sd": 0.0381870369798241,
    "residual_calibration_wins": 6,
    "residual_calibration_approx_95ci_low": -0.04246694548497516,
    "residual_calibration_approx_95ci_high": -0.008994531750870675
  },
  "action_predictive_geometric_chevron_minus_oracle_geometric_chevron": {
    "return_per_decision_mean": -0.12766666666666665,
    "return_per_decision_sd": 0.07753588114864676,
    "return_per_decision_wins": 1,
    "return_per_decision_approx_95ci_low": -0.1616482648516355,
    "return_per_decision_approx_95ci_high": -0.0936850684816978,
    "final_old_accuracy_mean": -0.025174356278185532,
    "final_old_accuracy_sd": 0.04524856857881952,
    "final_old_accuracy_wins": 2,
    "final_old_accuracy_approx_95ci_low": -0.045005415822638624,
    "final_old_accuracy_approx_95ci_high": -0.00534329673373244,
    "final_new_accuracy_mean": -0.27612515609531807,
    "final_new_accuracy_sd": 0.18857085029394122,
    "final_new_accuracy_wins": 1,
    "final_new_accuracy_approx_95ci_low": -0.3587699751024271,
    "final_new_accuracy_approx_95ci_high": -0.19348033708820905,
    "residual_calibration_mean": -0.07029945275314783,
    "residual_calibration_sd": 0.06966883158475339,
    "residual_calibration_wins": 3,
    "residual_calibration_approx_95ci_low": -0.10083316444710046,
    "residual_calibration_approx_95ci_high": -0.03976574105919521
  },
  "action_predictive_geometric_chevron_minus_action_predictive_content_attention": {
    "return_per_decision_mean": 0.017499999999999995,
    "return_per_decision_sd": 0.07731514456396865,
    "return_per_decision_wins": 9,
    "return_per_decision_approx_95ci_low": -0.016384856111310422,
    "return_per_decision_approx_95ci_high": 0.05138485611131041,
    "final_old_accuracy_mean": 0.020091207697574155,
    "final_old_accuracy_sd": 0.03372249189817767,
    "final_old_accuracy_wins": 11,
    "final_old_accuracy_approx_95ci_low": 0.0053116739835919625,
    "final_old_accuracy_approx_95ci_high": 0.03487074141155635,
    "final_new_accuracy_mean": 0.01932944179553214,
    "final_new_accuracy_sd": 0.1910058779277894,
    "final_new_accuracy_wins": 9,
    "final_new_accuracy_approx_95ci_low": -0.06438257512558634,
    "final_new_accuracy_approx_95ci_high": 0.10304145871665063,
    "residual_calibration_mean": -0.197693979864554,
    "residual_calibration_sd": 0.10658651850221752,
    "residual_calibration_wins": 0,
    "residual_calibration_approx_95ci_low": -0.2444075812323295,
    "residual_calibration_approx_95ci_high": -0.1509803784967785
  },
  "action_predictive_geometric_chevron_minus_raw_sensor_geometric_chevron": {
    "return_per_decision_mean": 0.3501666666666666,
    "return_per_decision_sd": 0.10803657058975598,
    "return_per_decision_wins": 20,
    "return_per_decision_approx_95ci_low": 0.30281755195131543,
    "return_per_decision_approx_95ci_high": 0.3975157813820178,
    "final_old_accuracy_mean": 0.30026838094813085,
    "final_old_accuracy_sd": 0.09250893462251698,
    "final_old_accuracy_wins": 20,
    "final_old_accuracy_approx_95ci_low": 0.2597245527450926,
    "final_old_accuracy_approx_95ci_high": 0.3408122091511691,
    "final_new_accuracy_mean": 0.26915689906337,
    "final_new_accuracy_sd": 0.270532233107777,
    "final_new_accuracy_wins": 15,
    "final_new_accuracy_approx_95ci_low": 0.1505909202499435,
    "final_new_accuracy_approx_95ci_high": 0.3877228778767965,
    "residual_calibration_mean": 0.0862602841853407,
    "residual_calibration_sd": 0.11494759705333983,
    "residual_calibration_wins": 15,
    "residual_calibration_approx_95ci_low": 0.03588227857648231,
    "residual_calibration_approx_95ci_high": 0.13663828979419906
  }
}
```

Confirmation was not run because the frozen development gate failed. See
`experiment_007_development_findings.md` for interpretation.
