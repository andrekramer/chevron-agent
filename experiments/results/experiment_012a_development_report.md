# Experiment 012a: slot maturity development

- Encoder seeds: [1320, 1321]
- Lifetimes per encoder: 10
- Successful uses required for maturity: 4

| Condition | Return | Post-shift return | Shift probe | Mature core | Retention | Novel probe | Core lost | Deferrals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Learned preloaded protected | 0.472 +/- 0.078 | 0.397 +/- 0.106 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.923 +/- 0.034 | 0.762 +/- 0.201 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Learned cold baseline | 0.299 +/- 0.074 | 0.360 +/- 0.101 | 0.906 +/- 0.087 | 0.000 +/- 0.000 | 0.888 +/- 0.064 | 0.775 +/- 0.249 | 0.300 +/- 0.714 | 0.000 +/- 0.000 |
| Learned cold mature | 0.302 +/- 0.076 | 0.364 +/- 0.102 | 0.906 +/- 0.087 | 6.250 +/- 0.994 | 0.889 +/- 0.063 | 0.775 +/- 0.249 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Learned cold immediately protected | 0.299 +/- 0.076 | 0.360 +/- 0.104 | 0.906 +/- 0.087 | 7.650 +/- 0.654 | 0.884 +/- 0.070 | 0.750 +/- 0.237 | 0.000 +/- 0.000 | 2.950 +/- 9.713 |
| Learned cold mature direct | 0.387 +/- 0.057 | 0.472 +/- 0.066 | 0.919 +/- 0.091 | 6.450 +/- 1.161 | 0.919 +/- 0.034 | 0.662 +/- 0.227 | 0.000 +/- 0.000 | 0.150 +/- 0.654 |

## Frozen gate

- PASS: `core_promotions_at_least_7.5`
- FAIL: `mature_core_slots_at_least_7`
- PASS: `core_probe_at_least_0.75`
- PASS: `retention_at_least_0.85`
- PASS: `reversed_probe_at_least_0.70`
- PASS: `novel_probe_at_least_0.70`
- PASS: `postshift_novel_promotions_at_least_3`
- PASS: `unique_revisions_at_least_3`
- PASS: `identity_calibration_at_least_0.10`
- PASS: `policy_calibration_at_least_0.10`
- PASS: `no_mature_slot_evictions`
- PASS: `no_duplicate_allocations`
- PASS: `no_established_overwrites`
- PASS: `no_under_supported_writes`
- PASS: `no_core_slots_lost`
- PASS: `postshift_return_noninferior_to_baseline`
- PASS: `retention_noninferior_to_baseline`
- PASS: `novel_probe_noninferior_to_baseline`
- PASS: `postshift_return_noninferior_to_immediate`
- PASS: `novel_probe_noninferior_to_immediate`
- PASS: `postshift_return_noninferior_to_preloaded`
- FAIL: `retention_noninferior_to_preloaded`
- FAIL: `retention_noninferior_to_direct`
- FAIL: `novel_probe_better_than_direct`
- PASS: `total_return_cost_vs_direct_within_0.15`

Overall development result: **FAIL**

## Paired diagnostics

```json
{
  "learned_cold_mature_minus_learned_cold_baseline": {
    "return_per_decision_mean": 0.0026249999999999997,
    "return_per_decision_population_sd": 0.009793971359974459,
    "return_per_decision_approx_95ci_low": -0.0016673972031954346,
    "return_per_decision_approx_95ci_high": 0.006917397203195434,
    "return_per_decision_wins": 3,
    "postshift_return_mean": 0.0035000000000000005,
    "postshift_return_population_sd": 0.013058628479965948,
    "postshift_return_approx_95ci_low": -0.002223196270927247,
    "postshift_return_approx_95ci_high": 0.009223196270927249,
    "postshift_return_wins": 3,
    "retention_accuracy_mean": 0.0007500000000000007,
    "retention_accuracy_population_sd": 0.019828956099603445,
    "retention_accuracy_approx_95ci_low": -0.007940423177268188,
    "retention_accuracy_approx_95ci_high": 0.00944042317726819,
    "retention_accuracy_wins": 2,
    "core_probe_at_shift_mean": 0.0,
    "core_probe_at_shift_population_sd": 0.0,
    "core_probe_at_shift_approx_95ci_low": 0.0,
    "core_probe_at_shift_approx_95ci_high": 0.0,
    "core_probe_at_shift_wins": 0,
    "reversed_probe_accuracy_mean": 0.0375,
    "reversed_probe_accuracy_population_sd": 0.08926785535678562,
    "reversed_probe_accuracy_approx_95ci_low": -0.0016233625855447131,
    "reversed_probe_accuracy_approx_95ci_high": 0.07662336258554471,
    "reversed_probe_accuracy_wins": 3,
    "novel_probe_accuracy_mean": 0.0,
    "novel_probe_accuracy_population_sd": 0.0,
    "novel_probe_accuracy_approx_95ci_low": 0.0,
    "novel_probe_accuracy_approx_95ci_high": 0.0,
    "novel_probe_accuracy_wins": 0,
    "core_slots_lost_mean": -0.3,
    "core_slots_lost_population_sd": 0.714142842854285,
    "core_slots_lost_approx_95ci_low": -0.6129869006843577,
    "core_slots_lost_approx_95ci_high": 0.012986900684357705,
    "core_slots_lost_wins": 0
  },
  "learned_cold_mature_minus_learned_cold_immediate_protection": {
    "return_per_decision_mean": 0.002999999999999997,
    "return_per_decision_population_sd": 0.01865140745359448,
    "return_per_decision_approx_95ci_low": -0.00517433972868757,
    "return_per_decision_approx_95ci_high": 0.011174339728687565,
    "return_per_decision_wins": 1,
    "postshift_return_mean": 0.003999999999999998,
    "postshift_return_population_sd": 0.024868543271459317,
    "postshift_return_approx_95ci_low": -0.006899119638250096,
    "postshift_return_approx_95ci_high": 0.014899119638250092,
    "postshift_return_wins": 1,
    "retention_accuracy_mean": 0.004999999999999999,
    "retention_accuracy_population_sd": 0.04077376607575024,
    "retention_accuracy_approx_95ci_low": -0.012869890878234267,
    "retention_accuracy_approx_95ci_high": 0.022869890878234267,
    "retention_accuracy_wins": 1,
    "core_probe_at_shift_mean": 0.0,
    "core_probe_at_shift_population_sd": 0.0,
    "core_probe_at_shift_approx_95ci_low": 0.0,
    "core_probe_at_shift_approx_95ci_high": 0.0,
    "core_probe_at_shift_wins": 0,
    "reversed_probe_accuracy_mean": 0.0,
    "reversed_probe_accuracy_population_sd": 0.0,
    "reversed_probe_accuracy_approx_95ci_low": 0.0,
    "reversed_probe_accuracy_approx_95ci_high": 0.0,
    "reversed_probe_accuracy_wins": 0,
    "novel_probe_accuracy_mean": 0.025,
    "novel_probe_accuracy_population_sd": 0.075,
    "novel_probe_accuracy_approx_95ci_low": -0.007870199269246904,
    "novel_probe_accuracy_approx_95ci_high": 0.05787019926924691,
    "novel_probe_accuracy_wins": 2,
    "core_slots_lost_mean": 0.0,
    "core_slots_lost_population_sd": 0.0,
    "core_slots_lost_approx_95ci_low": 0.0,
    "core_slots_lost_approx_95ci_high": 0.0,
    "core_slots_lost_wins": 0
  },
  "learned_cold_mature_minus_learned_preloaded_protected": {
    "return_per_decision_mean": -0.17025,
    "return_per_decision_population_sd": 0.07036023379722384,
    "return_per_decision_approx_95ci_low": -0.20108673207394065,
    "return_per_decision_approx_95ci_high": -0.13941326792605938,
    "return_per_decision_wins": 0,
    "postshift_return_mean": -0.033499999999999995,
    "postshift_return_population_sd": 0.08339647607796281,
    "postshift_return_approx_95ci_low": -0.07005011716047493,
    "postshift_return_approx_95ci_high": 0.003050117160474944,
    "postshift_return_wins": 6,
    "retention_accuracy_mean": -0.03425,
    "retention_accuracy_population_sd": 0.05688310381826924,
    "retention_accuracy_approx_95ci_low": -0.05918011943413026,
    "retention_accuracy_approx_95ci_high": -0.009319880565869743,
    "retention_accuracy_wins": 3,
    "core_probe_at_shift_mean": -0.09375,
    "core_probe_at_shift_population_sd": 0.08727650027355588,
    "core_probe_at_shift_approx_95ci_low": -0.13200061274019018,
    "core_probe_at_shift_approx_95ci_high": -0.05549938725980982,
    "core_probe_at_shift_wins": 0,
    "reversed_probe_accuracy_mean": -0.025,
    "reversed_probe_accuracy_population_sd": 0.175,
    "reversed_probe_accuracy_approx_95ci_low": -0.10169713162824279,
    "reversed_probe_accuracy_approx_95ci_high": 0.05169713162824278,
    "reversed_probe_accuracy_wins": 2,
    "novel_probe_accuracy_mean": 0.0125,
    "novel_probe_accuracy_population_sd": 0.21614520582238228,
    "novel_probe_accuracy_approx_95ci_low": -0.0822298131529879,
    "novel_probe_accuracy_approx_95ci_high": 0.1072298131529879,
    "novel_probe_accuracy_wins": 7,
    "core_slots_lost_mean": 0.0,
    "core_slots_lost_population_sd": 0.0,
    "core_slots_lost_approx_95ci_low": 0.0,
    "core_slots_lost_approx_95ci_high": 0.0,
    "core_slots_lost_wins": 0
  },
  "learned_cold_mature_minus_learned_cold_mature_direct": {
    "return_per_decision_mean": -0.085,
    "return_per_decision_population_sd": 0.04481908075808785,
    "return_per_decision_approx_95ci_low": -0.1046428282077709,
    "return_per_decision_approx_95ci_high": -0.06535717179222911,
    "return_per_decision_wins": 1,
    "postshift_return_mean": -0.1085,
    "postshift_return_population_sd": 0.06198095585652813,
    "postshift_return_approx_95ci_low": -0.13566435159869966,
    "postshift_return_approx_95ci_high": -0.08133564840130032,
    "postshift_return_wins": 1,
    "retention_accuracy_mean": -0.029750000000000016,
    "retention_accuracy_population_sd": 0.05238022050354503,
    "retention_accuracy_approx_95ci_low": -0.05270664380958159,
    "retention_accuracy_approx_95ci_high": -0.006793356190418445,
    "retention_accuracy_wins": 6,
    "core_probe_at_shift_mean": -0.0125,
    "core_probe_at_shift_population_sd": 0.11110243021644486,
    "core_probe_at_shift_approx_95ci_low": -0.06119278694016189,
    "core_probe_at_shift_approx_95ci_high": 0.03619278694016188,
    "core_probe_at_shift_wins": 4,
    "reversed_probe_accuracy_mean": -0.0375,
    "reversed_probe_accuracy_population_sd": 0.16345871038277526,
    "reversed_probe_accuracy_approx_95ci_low": -0.10913893843434588,
    "reversed_probe_accuracy_approx_95ci_high": 0.03413893843434588,
    "reversed_probe_accuracy_wins": 2,
    "novel_probe_accuracy_mean": 0.1125,
    "novel_probe_accuracy_population_sd": 0.2678035660703569,
    "novel_probe_accuracy_approx_95ci_low": -0.004870087756634139,
    "novel_probe_accuracy_approx_95ci_high": 0.22987008775663414,
    "novel_probe_accuracy_wins": 8,
    "core_slots_lost_mean": 0.0,
    "core_slots_lost_population_sd": 0.0,
    "core_slots_lost_approx_95ci_low": 0.0,
    "core_slots_lost_approx_95ci_high": 0.0,
    "core_slots_lost_wins": 0
  }
}
```
