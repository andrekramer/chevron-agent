# Experiment 012: empty-memory bootstrap development

- Encoder seeds: [1300, 1301]
- RL lifetimes per encoder: 10
- Main condition begins with **zero permanent memory slots**

| Condition | Return | Late core | Shift probe | Core IDs | Retention | Reversed probe | Novel probe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle preloaded protected | 0.563 +/- 0.046 | 0.960 +/- 0.033 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.953 +/- 0.020 | 0.988 +/- 0.054 | 0.950 +/- 0.100 |
| Learned preloaded protected | 0.505 +/- 0.066 | 0.934 +/- 0.036 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.932 +/- 0.029 | 0.963 +/- 0.089 | 0.875 +/- 0.148 |
| Oracle cold protected | 0.399 +/- 0.091 | 0.858 +/- 0.095 | 0.906 +/- 0.104 | 7.650 +/- 0.572 | 0.948 +/- 0.022 | 0.975 +/- 0.075 | 0.863 +/- 0.147 |
| Raw-sensor cold protected | 0.067 +/- 0.133 | 0.598 +/- 0.134 | 0.713 +/- 0.132 | 5.900 +/- 0.943 | 0.802 +/- 0.169 | 0.600 +/- 0.242 | 0.175 +/- 0.160 |
| Learned cold protected | 0.316 +/- 0.104 | 0.815 +/- 0.105 | 0.900 +/- 0.109 | 7.500 +/- 0.742 | 0.921 +/- 0.029 | 0.825 +/- 0.225 | 0.863 +/- 0.124 |
| Learned cold direct adaptation | 0.411 +/- 0.085 | 0.830 +/- 0.102 | 0.887 +/- 0.124 | 7.500 +/- 0.806 | 0.918 +/- 0.036 | 0.900 +/- 0.146 | 0.700 +/- 0.218 |

## Frozen gate

- PASS: `core_promotions_at_least_7.5`
- PASS: `late_core_accuracy_at_least_0.70`
- PASS: `core_probe_at_least_0.75`
- PASS: `retention_accuracy_at_least_0.85`
- PASS: `reversed_probe_at_least_0.70`
- PASS: `novel_probe_at_least_0.70`
- PASS: `postshift_novel_promotions_at_least_3`
- PASS: `unique_revisions_at_least_3`
- PASS: `identity_calibration_at_least_0.10`
- PASS: `policy_calibration_at_least_0.10`
- PASS: `false_stable_revisions_at_most_0.50`
- PASS: `no_duplicate_allocations`
- PASS: `no_established_overwrites`
- PASS: `no_under_supported_writes`
- PASS: `return_better_than_raw_cold`
- PASS: `core_probe_better_than_raw_cold`
- FAIL: `return_noninferior_to_oracle_cold`
- PASS: `clean_accuracy_noninferior_to_oracle_cold`
- PASS: `core_probe_noninferior_to_oracle_cold`
- FAIL: `return_noninferior_to_learned_preloaded`
- PASS: `retention_noninferior_to_learned_preloaded`
- FAIL: `return_noninferior_to_direct`
- PASS: `clean_accuracy_noninferior_to_direct`
- PASS: `retention_noninferior_to_direct`

Overall development result: **FAIL**

## Paired diagnostics

```json
{
  "learned_cold_protected_minus_raw_sensor_cold_protected": {
    "return_per_decision_mean": 0.2485,
    "return_per_decision_population_sd": 0.15851537780291222,
    "return_per_decision_approx_95ci_low": 0.1790275725917109,
    "return_per_decision_approx_95ci_high": 0.3179724274082891,
    "return_per_decision_wins": 19,
    "clean_accuracy_mean": 0.16125,
    "clean_accuracy_population_sd": 0.09363075749987287,
    "clean_accuracy_approx_95ci_low": 0.12021451124331525,
    "clean_accuracy_approx_95ci_high": 0.20228548875668476,
    "clean_accuracy_wins": 19,
    "core_probe_at_shift_mean": 0.1875,
    "core_probe_at_shift_population_sd": 0.1505199322349037,
    "core_probe_at_shift_approx_95ci_low": 0.12153173111260232,
    "core_probe_at_shift_approx_95ci_high": 0.2534682688873977,
    "core_probe_at_shift_wins": 16,
    "retention_accuracy_mean": 0.11850000000000001,
    "retention_accuracy_population_sd": 0.15913909010673652,
    "retention_accuracy_approx_95ci_low": 0.0487542186221991,
    "retention_accuracy_approx_95ci_high": 0.1882457813778009,
    "retention_accuracy_wins": 15,
    "reversed_probe_accuracy_mean": 0.225,
    "reversed_probe_accuracy_population_sd": 0.3152380053229623,
    "reversed_probe_accuracy_approx_95ci_low": 0.08684085263725749,
    "reversed_probe_accuracy_approx_95ci_high": 0.3631591473627425,
    "reversed_probe_accuracy_wins": 14,
    "novel_probe_accuracy_mean": 0.6875,
    "novel_probe_accuracy_population_sd": 0.17455300054711176,
    "novel_probe_accuracy_approx_95ci_low": 0.6109987745196196,
    "novel_probe_accuracy_approx_95ci_high": 0.7640012254803804,
    "novel_probe_accuracy_wins": 20
  },
  "learned_cold_protected_minus_oracle_cold_protected": {
    "return_per_decision_mean": -0.08287499999999999,
    "return_per_decision_population_sd": 0.10423013899539806,
    "return_per_decision_approx_95ci_low": -0.12855587251520048,
    "return_per_decision_approx_95ci_high": -0.0371941274847995,
    "return_per_decision_wins": 2,
    "clean_accuracy_mean": -0.0476875,
    "clean_accuracy_population_sd": 0.06520793064305906,
    "clean_accuracy_approx_95ci_low": -0.07626613565563445,
    "clean_accuracy_approx_95ci_high": -0.01910886434436556,
    "clean_accuracy_wins": 2,
    "core_probe_at_shift_mean": -0.00625,
    "core_probe_at_shift_population_sd": 0.10807260291119114,
    "core_probe_at_shift_approx_95ci_low": -0.05361490657649395,
    "core_probe_at_shift_approx_95ci_high": 0.04111490657649395,
    "core_probe_at_shift_wins": 4,
    "retention_accuracy_mean": -0.02674999999999999,
    "retention_accuracy_population_sd": 0.027261465477849845,
    "retention_accuracy_approx_95ci_low": -0.038697864035048254,
    "retention_accuracy_approx_95ci_high": -0.014802135964951725,
    "retention_accuracy_wins": 2,
    "reversed_probe_accuracy_mean": -0.15,
    "reversed_probe_accuracy_population_sd": 0.21505813167606566,
    "reversed_probe_accuracy_approx_95ci_low": -0.2442533819021896,
    "reversed_probe_accuracy_approx_95ci_high": -0.05574661809781041,
    "reversed_probe_accuracy_wins": 0,
    "novel_probe_accuracy_mean": 0.0,
    "novel_probe_accuracy_population_sd": 0.11180339887498948,
    "novel_probe_accuracy_approx_95ci_low": -0.048999999999999995,
    "novel_probe_accuracy_approx_95ci_high": 0.048999999999999995,
    "novel_probe_accuracy_wins": 2
  },
  "learned_cold_protected_minus_learned_preloaded_protected": {
    "return_per_decision_mean": -0.18875,
    "return_per_decision_population_sd": 0.08153028578387299,
    "return_per_decision_approx_95ci_low": -0.22448222320259403,
    "return_per_decision_approx_95ci_high": -0.15301777679740597,
    "return_per_decision_wins": 0,
    "clean_accuracy_mean": -0.11775,
    "clean_accuracy_population_sd": 0.05052783886136433,
    "clean_accuracy_approx_95ci_low": -0.13989480176023256,
    "clean_accuracy_approx_95ci_high": -0.09560519823976742,
    "clean_accuracy_wins": 0,
    "core_probe_at_shift_mean": -0.1,
    "core_probe_at_shift_population_sd": 0.10897247358851683,
    "core_probe_at_shift_approx_95ci_low": -0.14775929228956391,
    "core_probe_at_shift_approx_95ci_high": -0.05224070771043609,
    "core_probe_at_shift_wins": 0,
    "retention_accuracy_mean": -0.011249999999999987,
    "retention_accuracy_population_sd": 0.020302401335802606,
    "retention_accuracy_approx_95ci_low": -0.020147919700694072,
    "retention_accuracy_approx_95ci_high": -0.002352080299305904,
    "retention_accuracy_wins": 5,
    "reversed_probe_accuracy_mean": -0.1375,
    "reversed_probe_accuracy_population_sd": 0.20116846174288852,
    "reversed_probe_accuracy_approx_95ci_low": -0.22566596565568825,
    "reversed_probe_accuracy_approx_95ci_high": -0.04933403434431176,
    "reversed_probe_accuracy_wins": 0,
    "novel_probe_accuracy_mean": -0.0125,
    "novel_probe_accuracy_population_sd": 0.20116846174288852,
    "novel_probe_accuracy_approx_95ci_low": -0.10066596565568825,
    "novel_probe_accuracy_approx_95ci_high": 0.07566596565568826,
    "novel_probe_accuracy_wins": 4
  },
  "learned_cold_protected_minus_learned_cold_direct": {
    "return_per_decision_mean": -0.09474999999999999,
    "return_per_decision_population_sd": 0.07234854870693674,
    "return_per_decision_approx_95ci_low": -0.12645814950450435,
    "return_per_decision_approx_95ci_high": -0.06304185049549563,
    "return_per_decision_wins": 2,
    "clean_accuracy_mean": -0.05587500000000001,
    "clean_accuracy_population_sd": 0.04037732191465897,
    "clean_accuracy_approx_95ci_low": -0.07357114156391162,
    "clean_accuracy_approx_95ci_high": -0.038178858436088405,
    "clean_accuracy_wins": 1,
    "core_probe_at_shift_mean": 0.0125,
    "core_probe_at_shift_population_sd": 0.0960143218483576,
    "core_probe_at_shift_approx_95ci_low": -0.029580131891428284,
    "core_probe_at_shift_approx_95ci_high": 0.05458013189142828,
    "core_probe_at_shift_wins": 5,
    "retention_accuracy_mean": 0.0034999999999999975,
    "retention_accuracy_population_sd": 0.025840859118845066,
    "retention_accuracy_approx_95ci_low": -0.007825255846999647,
    "retention_accuracy_approx_95ci_high": 0.014825255846999643,
    "retention_accuracy_wins": 11,
    "reversed_probe_accuracy_mean": -0.075,
    "reversed_probe_accuracy_population_sd": 0.225,
    "reversed_probe_accuracy_approx_95ci_low": -0.1736105978077407,
    "reversed_probe_accuracy_approx_95ci_high": 0.02361059780774072,
    "reversed_probe_accuracy_wins": 2,
    "novel_probe_accuracy_mean": 0.1625,
    "novel_probe_accuracy_population_sd": 0.27698149757700424,
    "novel_probe_accuracy_approx_95ci_low": 0.04110750640999257,
    "novel_probe_accuracy_approx_95ci_high": 0.28389249359000746,
    "novel_probe_accuracy_wins": 9
  }
}
```
