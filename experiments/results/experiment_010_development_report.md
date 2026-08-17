# Experiment 010: retrospective-policy development

- Seeds: 100000000–100000019
- Policy signature in observation: **none**

| Condition | Return | Clean accuracy | Retention | Reversed probe | Novel probe | False revisions | Detection occurrences |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct value adaptation | 0.597 +/- 0.049 | 0.870 +/- 0.023 | 0.937 +/- 0.031 | 1.000 +/- 0.000 | 0.863 +/- 0.147 | 0.000 +/- 0.000 | 36.400 +/- 1.786 |
| Protected retrospective Chevron | 0.557 +/- 0.057 | 0.844 +/- 0.032 | 0.944 +/- 0.022 | 1.000 +/- 0.000 | 0.875 +/- 0.168 | 0.000 +/- 0.000 | 13.887 +/- 3.471 |
| Fast-veto Chevron | 0.326 +/- 0.066 | 0.701 +/- 0.034 | 0.792 +/- 0.036 | 0.988 +/- 0.054 | 0.863 +/- 0.147 | 0.350 +/- 0.853 | 13.088 +/- 5.018 |
| Immediate-write Chevron | 0.346 +/- 0.064 | 0.716 +/- 0.030 | 0.772 +/- 0.045 | 0.938 +/- 0.108 | 0.887 +/- 0.147 | 15.500 +/- 5.143 | 2.575 +/- 1.410 |

## Frozen development gate

- PASS: `retention_accuracy_at_least_0.90`
- PASS: `reversed_probe_at_least_0.75`
- PASS: `novel_probe_at_least_0.75`
- PASS: `new_promotions_at_least_3`
- PASS: `unique_revisions_at_least_3`
- PASS: `policy_calibration_at_least_0.10`
- PASS: `false_stable_revisions_at_most_0.25`
- PASS: `no_under_supported_writes`
- PASS: `no_established_overwrites`
- PASS: `no_duplicate_allocations`
- PASS: `return_noninferior_to_direct`
- PASS: `clean_accuracy_noninferior_to_direct`
- PASS: `retention_noninferior_to_direct`
- PASS: `fewer_false_revisions_than_immediate`

Confirmation triggered: **True**

## Paired diagnostics

```json
{
  "retrospective_protected_minus_direct_update": {
    "return_per_decision_mean": -0.04050000000000002,
    "return_per_decision_population_sd": 0.04682146943443786,
    "return_per_decision_approx_95ci_low": -0.06102041373851903,
    "return_per_decision_approx_95ci_high": -0.019979586261481014,
    "return_per_decision_wins": 2,
    "clean_accuracy_mean": -0.026375000000000003,
    "clean_accuracy_population_sd": 0.02757348137250716,
    "clean_accuracy_approx_95ci_low": -0.03845961103014904,
    "clean_accuracy_approx_95ci_high": -0.014290388969850961,
    "clean_accuracy_wins": 2,
    "retention_accuracy_mean": 0.006749999999999995,
    "retention_accuracy_population_sd": 0.02637588861062315,
    "retention_accuracy_approx_95ci_low": -0.00480974286046191,
    "retention_accuracy_approx_95ci_high": 0.0183097428604619,
    "retention_accuracy_wins": 12,
    "reversed_probe_accuracy_mean": 0.0,
    "reversed_probe_accuracy_population_sd": 0.0,
    "reversed_probe_accuracy_approx_95ci_low": 0.0,
    "reversed_probe_accuracy_approx_95ci_high": 0.0,
    "reversed_probe_accuracy_wins": 0,
    "novel_probe_accuracy_mean": 0.0125,
    "novel_probe_accuracy_population_sd": 0.18498310733685927,
    "novel_probe_accuracy_approx_95ci_low": -0.06857242132809405,
    "novel_probe_accuracy_approx_95ci_high": 0.09357242132809404,
    "novel_probe_accuracy_wins": 4,
    "false_stable_revisions_mean": 0.0,
    "false_stable_revisions_population_sd": 0.0,
    "false_stable_revisions_approx_95ci_low": 0.0,
    "false_stable_revisions_approx_95ci_high": 0.0,
    "false_stable_revisions_wins": 0,
    "mean_reversal_detection_occurrences_mean": -22.5125,
    "mean_reversal_detection_occurrences_population_sd": 3.282981685906883,
    "mean_reversal_detection_occurrences_approx_95ci_low": -23.95133016284063,
    "mean_reversal_detection_occurrences_approx_95ci_high": -21.07366983715937,
    "mean_reversal_detection_occurrences_wins": 0
  },
  "retrospective_protected_minus_retrospective_fast_veto": {
    "return_per_decision_mean": 0.231,
    "return_per_decision_population_sd": 0.046512095201140954,
    "return_per_decision_approx_95ci_low": 0.21061517549744418,
    "return_per_decision_approx_95ci_high": 0.25138482450255584,
    "return_per_decision_wins": 20,
    "clean_accuracy_mean": 0.142625,
    "clean_accuracy_population_sd": 0.02463768708706237,
    "clean_accuracy_approx_95ci_low": 0.13182705754553212,
    "clean_accuracy_approx_95ci_high": 0.1534229424544679,
    "clean_accuracy_wins": 20,
    "retention_accuracy_mean": 0.15099999999999997,
    "retention_accuracy_population_sd": 0.0344093010681705,
    "retention_accuracy_approx_95ci_low": 0.13591945889564963,
    "retention_accuracy_approx_95ci_high": 0.1660805411043503,
    "retention_accuracy_wins": 20,
    "reversed_probe_accuracy_mean": 0.0125,
    "reversed_probe_accuracy_population_sd": 0.054486236794258416,
    "reversed_probe_accuracy_approx_95ci_low": -0.011379646144781957,
    "reversed_probe_accuracy_approx_95ci_high": 0.03637964614478196,
    "reversed_probe_accuracy_wins": 1,
    "novel_probe_accuracy_mean": 0.0125,
    "novel_probe_accuracy_population_sd": 0.1243734296383275,
    "novel_probe_accuracy_approx_95ci_low": -0.04200905888015312,
    "novel_probe_accuracy_approx_95ci_high": 0.06700905888015313,
    "novel_probe_accuracy_wins": 3,
    "false_stable_revisions_mean": -0.35,
    "false_stable_revisions_population_sd": 0.852936105461599,
    "false_stable_revisions_approx_95ci_low": -0.7238157300061088,
    "false_stable_revisions_approx_95ci_high": 0.02381573000610876,
    "false_stable_revisions_wins": 0,
    "mean_reversal_detection_occurrences_mean": 0.8,
    "mean_reversal_detection_occurrences_population_sd": 3.507670166934172,
    "mean_reversal_detection_occurrences_approx_95ci_low": -0.7373042314389171,
    "mean_reversal_detection_occurrences_approx_95ci_high": 2.3373042314389174,
    "mean_reversal_detection_occurrences_wins": 11
  },
  "retrospective_protected_minus_retrospective_immediate_write": {
    "return_per_decision_mean": 0.2105,
    "return_per_decision_population_sd": 0.06788133027570983,
    "return_per_decision_approx_95ci_low": 0.18074969529567805,
    "return_per_decision_approx_95ci_high": 0.24025030470432193,
    "return_per_decision_wins": 20,
    "clean_accuracy_mean": 0.12825000000000003,
    "clean_accuracy_population_sd": 0.0367410703436903,
    "clean_accuracy_approx_95ci_low": 0.11214751595249979,
    "clean_accuracy_approx_95ci_high": 0.14435248404750028,
    "clean_accuracy_wins": 20,
    "retention_accuracy_mean": 0.17149999999999999,
    "retention_accuracy_population_sd": 0.04295637321748659,
    "retention_accuracy_approx_95ci_low": 0.15267353936609432,
    "retention_accuracy_approx_95ci_high": 0.19032646063390565,
    "retention_accuracy_wins": 20,
    "reversed_probe_accuracy_mean": 0.0625,
    "reversed_probe_accuracy_population_sd": 0.10825317547305482,
    "reversed_probe_accuracy_approx_95ci_low": 0.01505595400895915,
    "reversed_probe_accuracy_approx_95ci_high": 0.10994404599104085,
    "reversed_probe_accuracy_wins": 5,
    "novel_probe_accuracy_mean": -0.0125,
    "novel_probe_accuracy_population_sd": 0.2301494079940246,
    "novel_probe_accuracy_approx_95ci_low": -0.1133674253661706,
    "novel_probe_accuracy_approx_95ci_high": 0.08836742536617061,
    "novel_probe_accuracy_wins": 4,
    "false_stable_revisions_mean": -15.5,
    "false_stable_revisions_population_sd": 5.142956348249516,
    "false_stable_revisions_approx_95ci_low": -17.754,
    "false_stable_revisions_approx_95ci_high": -13.246,
    "false_stable_revisions_wins": 0,
    "mean_reversal_detection_occurrences_mean": 11.3125,
    "mean_reversal_detection_occurrences_population_sd": 3.159187989025028,
    "mean_reversal_detection_occurrences_approx_95ci_low": 9.927924816956478,
    "mean_reversal_detection_occurrences_approx_95ci_high": 12.697075183043522,
    "mean_reversal_detection_occurrences_wins": 20
  }
}
```
