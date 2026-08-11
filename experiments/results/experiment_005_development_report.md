# Experiment 005: development retrospective assent

- Training seeds: 0–1
- Training lifetimes per seed: 60
- Evaluation lifetimes per seed: 10
- Provisional buffer capacity: 4
- Retrospective loss weight: 1.0
- Learned model parameters: 314 each

| Method | Return | Final old | Final new | Old probe | New probe | q calibration | Promotions | N drift | Premature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Content attention + buffer | 0.788 +/- 0.051 | 0.982 +/- 0.013 | 0.821 +/- 0.156 | 1.000 +/- 0.000 | 0.887 +/- 0.167 | 0.457 +/- 0.137 | 3.550 +/- 0.669 | 0.054 +/- 0.007 | 0.000 +/- 0.000 |
| Bilinear null attention + retrospective | 0.458 +/- 0.050 | 0.749 +/- 0.091 | 0.327 +/- 0.175 | 0.769 +/- 0.144 | 0.400 +/- 0.267 | 0.067 +/- 0.049 | 0.000 +/- 0.000 | 0.264 +/- 0.028 | 0.000 +/- 0.000 |
| Chevron + retrospective + buffer | 0.512 +/- 0.083 | 0.806 +/- 0.054 | 0.357 +/- 0.205 | 0.812 +/- 0.093 | 0.438 +/- 0.236 | 0.030 +/- 0.049 | 0.550 +/- 0.589 | 0.135 +/- 0.020 | 0.000 +/- 0.000 |
| Chevron + policy only + buffer | 0.513 +/- 0.088 | 0.805 +/- 0.066 | 0.362 +/- 0.201 | 0.812 +/- 0.101 | 0.412 +/- 0.241 | 0.032 +/- 0.050 | 0.550 +/- 0.669 | 0.136 +/- 0.021 | 0.000 +/- 0.000 |
| Chevron + retrospective + immediate | 0.554 +/- 0.071 | 0.869 +/- 0.074 | 0.354 +/- 0.174 | 0.881 +/- 0.101 | 0.475 +/- 0.208 | 0.044 +/- 0.052 | 0.000 +/- 0.000 | 0.102 +/- 0.020 | 0.010 +/- 0.005 |
| Chevron + retrospective + coupled write | 0.503 +/- 0.076 | 0.804 +/- 0.048 | 0.331 +/- 0.188 | 0.838 +/- 0.105 | 0.350 +/- 0.278 | 0.033 +/- 0.049 | 0.500 +/- 0.592 | 0.150 +/- 0.022 | 0.000 +/- 0.000 |

## Frozen confirmation gate

- FAIL: `old_accuracy_at_least_0.90`
- FAIL: `new_accuracy_at_least_0.75`
- FAIL: `new_probe_at_least_0.75`
- FAIL: `promotions_at_least_3`
- FAIL: `q_calibration_at_least_0.15`
- FAIL: `policy_only_return_ci_low_above_0`
- PASS: `no_premature_writes`
- PASS: `positive_read_write_margin`

Confirmation triggered: **False**

## Paired diagnostics

```json
{
  "chevron_retrospective_buffer_minus_content_attention_buffer": {
    "return_per_decision_mean": -0.27599999999999997,
    "return_per_decision_sd": 0.101091123273403,
    "return_per_decision_wins": 0,
    "return_per_decision_approx_95ci_low": -0.32030513821798345,
    "return_per_decision_approx_95ci_high": -0.2316948617820165,
    "final_old_accuracy_mean": -0.17645364018083248,
    "final_old_accuracy_sd": 0.05162887123665487,
    "final_old_accuracy_wins": 0,
    "final_old_accuracy_approx_95ci_low": -0.19908099065543428,
    "final_old_accuracy_approx_95ci_high": -0.1538262897062307,
    "final_new_accuracy_mean": -0.46406797951812084,
    "final_new_accuracy_sd": 0.2768502900118244,
    "final_new_accuracy_wins": 1,
    "final_new_accuracy_approx_95ci_low": -0.585402968857287,
    "final_new_accuracy_approx_95ci_high": -0.34273299017895464,
    "residual_calibration_mean": -0.42684368138705375,
    "residual_calibration_sd": 0.12889749747200732,
    "residual_calibration_wins": 0,
    "residual_calibration_approx_95ci_low": -0.4833355004165488,
    "residual_calibration_approx_95ci_high": -0.3703518623575587
  },
  "chevron_retrospective_buffer_minus_bilinear_retrospective_buffer": {
    "return_per_decision_mean": 0.05483333333333333,
    "return_per_decision_sd": 0.07855772265967036,
    "return_per_decision_wins": 16,
    "return_per_decision_approx_95ci_low": 0.020403893360512022,
    "return_per_decision_approx_95ci_high": 0.08926277330615465,
    "final_old_accuracy_mean": 0.05705978099989627,
    "final_old_accuracy_sd": 0.09781245620080863,
    "final_old_accuracy_wins": 14,
    "final_old_accuracy_approx_95ci_low": 0.014191581982095407,
    "final_old_accuracy_approx_95ci_high": 0.09992798001769712,
    "final_new_accuracy_mean": 0.030031363662714604,
    "final_new_accuracy_sd": 0.2608790796508663,
    "final_new_accuracy_wins": 11,
    "final_new_accuracy_approx_95ci_low": -0.08430393411464154,
    "final_new_accuracy_approx_95ci_high": 0.14436666144007076,
    "residual_calibration_mean": -0.03724672560104039,
    "residual_calibration_sd": 0.06586819139953445,
    "residual_calibration_wins": 8,
    "residual_calibration_approx_95ci_low": -0.0661147332918083,
    "residual_calibration_approx_95ci_high": -0.008378717910272487
  },
  "chevron_retrospective_buffer_minus_chevron_policy_only_buffer": {
    "return_per_decision_mean": -0.0005000000000000033,
    "return_per_decision_sd": 0.034118386941433515,
    "return_per_decision_wins": 8,
    "return_per_decision_approx_95ci_low": -0.015453042366802554,
    "return_per_decision_approx_95ci_high": 0.014453042366802546,
    "final_old_accuracy_mean": 0.000596828538875227,
    "final_old_accuracy_sd": 0.03674134470308308,
    "final_old_accuracy_wins": 9,
    "final_old_accuracy_approx_95ci_low": -0.01550577575193051,
    "final_old_accuracy_approx_95ci_high": 0.016699432829680963,
    "final_new_accuracy_mean": -0.004691610438928936,
    "final_new_accuracy_sd": 0.09315870404459187,
    "final_new_accuracy_wins": 8,
    "final_new_accuracy_approx_95ci_low": -0.04552021264706937,
    "final_new_accuracy_approx_95ci_high": 0.0361369917692115,
    "residual_calibration_mean": -0.0018997971468765807,
    "residual_calibration_sd": 0.008537504606963649,
    "residual_calibration_wins": 7,
    "residual_calibration_approx_95ci_low": -0.005641523516116695,
    "residual_calibration_approx_95ci_high": 0.0018419292223635342
  },
  "chevron_retrospective_buffer_minus_chevron_retrospective_immediate": {
    "return_per_decision_mean": -0.042,
    "return_per_decision_sd": 0.0804911821838686,
    "return_per_decision_wins": 4,
    "return_per_decision_approx_95ci_low": -0.07727681597068023,
    "return_per_decision_approx_95ci_high": -0.006723184029319776,
    "final_old_accuracy_mean": -0.06348284847647453,
    "final_old_accuracy_sd": 0.061388197838021887,
    "final_old_accuracy_wins": 2,
    "final_old_accuracy_approx_95ci_low": -0.09038741241935096,
    "final_old_accuracy_approx_95ci_high": -0.03657828453359811,
    "final_new_accuracy_mean": 0.003022671525827096,
    "final_new_accuracy_sd": 0.20482989568113846,
    "final_new_accuracy_wins": 10,
    "final_new_accuracy_approx_95ci_low": -0.08674798830534727,
    "final_new_accuracy_approx_95ci_high": 0.09279333135700146,
    "residual_calibration_mean": -0.014030200260860843,
    "residual_calibration_sd": 0.023678824703648106,
    "residual_calibration_wins": 6,
    "residual_calibration_approx_95ci_low": -0.024407902747133906,
    "residual_calibration_approx_95ci_high": -0.003652497774587782
  },
  "chevron_retrospective_buffer_minus_chevron_retrospective_coupled_write": {
    "return_per_decision_mean": 0.009833333333333341,
    "return_per_decision_sd": 0.022568284945474203,
    "return_per_decision_wins": 13,
    "return_per_decision_approx_95ci_low": -5.765364430508624e-05,
    "return_per_decision_approx_95ci_high": 0.01972432031097177,
    "final_old_accuracy_mean": 0.0017641481409079174,
    "final_old_accuracy_sd": 0.044632622136821905,
    "final_old_accuracy_wins": 10,
    "final_old_accuracy_approx_95ci_low": -0.01779696097304324,
    "final_old_accuracy_approx_95ci_high": 0.021325257254859075,
    "final_new_accuracy_mean": 0.026265373694632825,
    "final_new_accuracy_sd": 0.10576662924347428,
    "final_new_accuracy_wins": 9,
    "final_new_accuracy_approx_95ci_low": -0.0200888953622946,
    "final_new_accuracy_approx_95ci_high": 0.07261964275156024,
    "residual_calibration_mean": -0.003555657931207179,
    "residual_calibration_sd": 0.002386296085425773,
    "residual_calibration_wins": 1,
    "residual_calibration_approx_95ci_low": -0.004601498302452099,
    "residual_calibration_approx_95ci_high": -0.002509817559962259
  }
}
```

This remains a delayed contextual-bandit experiment. It does not establish
spatial-game performance or a persistent agent self.

The frozen confirmation was not run because six of the eight development-gate
checks failed. See `experiment_005_development_findings.md` for interpretation.
