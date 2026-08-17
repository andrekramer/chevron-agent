# Experiment 010: retrospective-policy confirmation

- Fresh seeds: 101000000–101000099
- Policy signature in observation: **none**

| Condition | Return | Clean accuracy | Retention | Reversed probe | Novel probe | False revisions | Detection occurrences |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct value adaptation | 0.580 +/- 0.041 | 0.865 +/- 0.020 | 0.946 +/- 0.025 | 0.985 +/- 0.059 | 0.860 +/- 0.181 | 0.000 +/- 0.000 | 36.270 +/- 2.153 |
| Protected retrospective Chevron | 0.538 +/- 0.058 | 0.838 +/- 0.032 | 0.948 +/- 0.023 | 0.980 +/- 0.076 | 0.915 +/- 0.147 | 0.060 +/- 0.341 | 14.248 +/- 3.546 |
| Fast-veto Chevron | 0.316 +/- 0.062 | 0.699 +/- 0.031 | 0.791 +/- 0.042 | 0.965 +/- 0.094 | 0.897 +/- 0.158 | 0.300 +/- 0.728 | 13.650 +/- 3.602 |
| Immediate-write Chevron | 0.323 +/- 0.062 | 0.703 +/- 0.033 | 0.763 +/- 0.054 | 0.917 +/- 0.133 | 0.853 +/- 0.184 | 16.870 +/- 5.403 | 2.825 +/- 1.238 |

## Frozen confirmation gate

- PASS: `retention_accuracy_at_least_0.90`
- PASS: `reversed_probe_at_least_0.75`
- PASS: `novel_probe_at_least_0.75`
- PASS: `new_promotions_at_least_3`
- PASS: `unique_revisions_at_least_3`
- PASS: `policy_calibration_at_least_0.10`
- PASS: `false_stable_revisions_at_most_0.25`
- PASS: `no_under_supported_writes`
- PASS: `no_established_overwrites`
- FAIL: `no_duplicate_allocations`
- PASS: `return_noninferior_to_direct`
- PASS: `clean_accuracy_noninferior_to_direct`
- PASS: `retention_noninferior_to_direct`
- PASS: `fewer_false_revisions_than_immediate`

Confirmation passed: **False**

## Paired diagnostics

```json
{
  "retrospective_protected_minus_direct_update": {
    "return_per_decision_mean": -0.04219999999999999,
    "return_per_decision_population_sd": 0.03896196863609436,
    "return_per_decision_approx_95ci_low": -0.04983654585267448,
    "return_per_decision_approx_95ci_high": -0.03456345414732549,
    "return_per_decision_wins": 14,
    "clean_accuracy_mean": -0.027350000000000006,
    "clean_accuracy_population_sd": 0.022903520471752807,
    "clean_accuracy_approx_95ci_low": -0.03183909001246356,
    "clean_accuracy_approx_95ci_high": -0.022860909987536454,
    "clean_accuracy_wins": 11,
    "retention_accuracy_mean": 0.001050000000000002,
    "retention_accuracy_population_sd": 0.022806742424116584,
    "retention_accuracy_approx_95ci_low": -0.0034201215151268474,
    "retention_accuracy_approx_95ci_high": 0.005520121515126852,
    "retention_accuracy_wins": 46,
    "reversed_probe_accuracy_mean": -0.005,
    "reversed_probe_accuracy_population_sd": 0.08645808232895291,
    "reversed_probe_accuracy_approx_95ci_low": -0.021945784136474772,
    "reversed_probe_accuracy_approx_95ci_high": 0.01194578413647477,
    "reversed_probe_accuracy_wins": 4,
    "novel_probe_accuracy_mean": 0.055,
    "novel_probe_accuracy_population_sd": 0.22522211259110417,
    "novel_probe_accuracy_approx_95ci_low": 0.010856465932143587,
    "novel_probe_accuracy_approx_95ci_high": 0.09914353406785642,
    "novel_probe_accuracy_wins": 30,
    "false_stable_revisions_mean": 0.06,
    "false_stable_revisions_population_sd": 0.3411744421846396,
    "false_stable_revisions_approx_95ci_low": -0.006870190668189363,
    "false_stable_revisions_approx_95ci_high": 0.12687019066818936,
    "false_stable_revisions_wins": 3,
    "mean_reversal_detection_occurrences_mean": -22.0225,
    "mean_reversal_detection_occurrences_population_sd": 4.2292279141706235,
    "mean_reversal_detection_occurrences_approx_95ci_low": -22.851428671177445,
    "mean_reversal_detection_occurrences_approx_95ci_high": -21.193571328822557,
    "mean_reversal_detection_occurrences_wins": 0
  },
  "retrospective_protected_minus_retrospective_fast_veto": {
    "return_per_decision_mean": 0.222225,
    "return_per_decision_population_sd": 0.0443098394828959,
    "return_per_decision_approx_95ci_low": 0.2135402714613524,
    "return_per_decision_approx_95ci_high": 0.2309097285386476,
    "return_per_decision_wins": 100,
    "clean_accuracy_mean": 0.1391875,
    "clean_accuracy_population_sd": 0.02602845498199998,
    "clean_accuracy_approx_95ci_low": 0.134085922823528,
    "clean_accuracy_approx_95ci_high": 0.144289077176472,
    "clean_accuracy_wins": 100,
    "retention_accuracy_mean": 0.15645,
    "retention_accuracy_population_sd": 0.036347592767609786,
    "retention_accuracy_approx_95ci_low": 0.1493258718175485,
    "retention_accuracy_approx_95ci_high": 0.1635741281824515,
    "retention_accuracy_wins": 100,
    "reversed_probe_accuracy_mean": 0.015,
    "reversed_probe_accuracy_population_sd": 0.09233092656309694,
    "reversed_probe_accuracy_approx_95ci_low": -0.0030968616063669976,
    "reversed_probe_accuracy_approx_95ci_high": 0.033096861606367,
    "reversed_probe_accuracy_wins": 10,
    "novel_probe_accuracy_mean": 0.0175,
    "novel_probe_accuracy_population_sd": 0.13348689074212494,
    "novel_probe_accuracy_approx_95ci_low": -0.008663430585456485,
    "novel_probe_accuracy_approx_95ci_high": 0.04366343058545649,
    "novel_probe_accuracy_wins": 13,
    "false_stable_revisions_mean": -0.24,
    "false_stable_revisions_population_sd": 0.7227724399837061,
    "false_stable_revisions_approx_95ci_low": -0.3816633982368064,
    "false_stable_revisions_approx_95ci_high": -0.09833660176319359,
    "false_stable_revisions_wins": 1,
    "mean_reversal_detection_occurrences_mean": 0.5975,
    "mean_reversal_detection_occurrences_population_sd": 3.385981209339473,
    "mean_reversal_detection_occurrences_approx_95ci_low": -0.06615231703053659,
    "mean_reversal_detection_occurrences_approx_95ci_high": 1.2611523170305365,
    "mean_reversal_detection_occurrences_wins": 61
  },
  "retrospective_protected_minus_retrospective_immediate_write": {
    "return_per_decision_mean": 0.2147,
    "return_per_decision_population_sd": 0.05253960410966189,
    "return_per_decision_approx_95ci_low": 0.20440223759450626,
    "return_per_decision_approx_95ci_high": 0.22499776240549374,
    "return_per_decision_wins": 100,
    "clean_accuracy_mean": 0.13522499999999998,
    "clean_accuracy_population_sd": 0.031884743608817046,
    "clean_accuracy_approx_95ci_low": 0.12897559025267186,
    "clean_accuracy_approx_95ci_high": 0.1414744097473281,
    "clean_accuracy_wins": 100,
    "retention_accuracy_mean": 0.18404999999999996,
    "retention_accuracy_population_sd": 0.048531922484072273,
    "retention_accuracy_approx_95ci_low": 0.1745377431931218,
    "retention_accuracy_approx_95ci_high": 0.19356225680687814,
    "retention_accuracy_wins": 100,
    "reversed_probe_accuracy_mean": 0.0625,
    "reversed_probe_accuracy_population_sd": 0.1340475661845451,
    "reversed_probe_accuracy_approx_95ci_low": 0.036226677027829154,
    "reversed_probe_accuracy_approx_95ci_high": 0.08877332297217085,
    "reversed_probe_accuracy_wins": 26,
    "novel_probe_accuracy_mean": 0.0625,
    "novel_probe_accuracy_population_sd": 0.21028254801575902,
    "novel_probe_accuracy_approx_95ci_low": 0.021284620588911236,
    "novel_probe_accuracy_approx_95ci_high": 0.10371537941108877,
    "novel_probe_accuracy_wins": 33,
    "false_stable_revisions_mean": -16.81,
    "false_stable_revisions_population_sd": 5.37902407505302,
    "false_stable_revisions_approx_95ci_low": -17.86428871871039,
    "false_stable_revisions_approx_95ci_high": -15.755711281289607,
    "false_stable_revisions_wins": 0,
    "mean_reversal_detection_occurrences_mean": 11.4225,
    "mean_reversal_detection_occurrences_population_sd": 3.4206532636325475,
    "mean_reversal_detection_occurrences_approx_95ci_low": 10.75205196032802,
    "mean_reversal_detection_occurrences_approx_95ci_high": 12.09294803967198,
    "mean_reversal_detection_occurrences_wins": 100
  }
}
```
