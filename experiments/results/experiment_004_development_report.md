# Experiment 004: development reward-derived memory

- Training seeds: 0–1
- Training lifetimes per seed: 60
- Fresh evaluation lifetimes per seed: 10
- Reward delay: 3
- Provisional buffer capacity: 2
- Learned comparator parameters: 314 each
- Learning signal: delayed scalar reward only

| Method | Return/decision | Final old | Final new | Old probe | New probe | q calibration | N drift | Premature writes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Content attention + buffer | 0.734 +/- 0.098 | 0.976 +/- 0.018 | 0.615 +/- 0.255 | 0.981 +/- 0.045 | 0.688 +/- 0.294 | 0.323 +/- 0.176 | 0.063 +/- 0.012 | 0.000 +/- 0.000 |
| Direct MLP + buffer | -0.261 +/- 0.041 | 0.451 +/- 0.051 | 0.042 +/- 0.055 | 0.450 +/- 0.073 | 0.087 +/- 0.143 | -0.005 +/- 0.007 | 0.335 +/- 0.048 | 0.000 +/- 0.000 |
| Chevron + buffer | 0.541 +/- 0.067 | 0.841 +/- 0.058 | 0.371 +/- 0.184 | 0.863 +/- 0.087 | 0.412 +/- 0.265 | 0.035 +/- 0.035 | 0.129 +/- 0.020 | 0.000 +/- 0.000 |
| Chevron immediate write | 0.540 +/- 0.057 | 0.878 +/- 0.071 | 0.322 +/- 0.135 | 0.894 +/- 0.107 | 0.438 +/- 0.222 | 0.040 +/- 0.045 | 0.108 +/- 0.022 | 0.011 +/- 0.007 |
| Chevron coupled write | 0.525 +/- 0.080 | 0.822 +/- 0.067 | 0.377 +/- 0.186 | 0.825 +/- 0.073 | 0.475 +/- 0.192 | 0.038 +/- 0.036 | 0.144 +/- 0.022 | 0.000 +/- 0.000 |

## Paired diagnostics

```json
{
  "chevron_buffer_minus_content_attention_buffer": {
    "return_per_decision_mean": -0.19333333333333333,
    "return_per_decision_sd": 0.11605100713612933,
    "return_per_decision_wins": 2,
    "return_per_decision_approx_95ci_low": -0.2441949297328182,
    "return_per_decision_approx_95ci_high": -0.14247173693384846,
    "final_old_accuracy_mean": -0.13522226852961644,
    "final_old_accuracy_sd": 0.06263770719081997,
    "final_old_accuracy_wins": 0,
    "final_old_accuracy_approx_95ci_low": -0.162674454091363,
    "final_old_accuracy_approx_95ci_high": -0.10777008296786987,
    "final_new_accuracy_mean": -0.2444452590614922,
    "final_new_accuracy_sd": 0.312379885189286,
    "final_new_accuracy_wins": 3,
    "final_new_accuracy_approx_95ci_low": -0.38135178004650955,
    "final_new_accuracy_approx_95ci_high": -0.10753873807647485
  },
  "chevron_buffer_minus_direct_mlp_buffer": {
    "return_per_decision_mean": 0.8015000000000001,
    "return_per_decision_sd": 0.08086140767672789,
    "return_per_decision_wins": 20,
    "return_per_decision_approx_95ci_low": 0.7660609255529887,
    "return_per_decision_approx_95ci_high": 0.8369390744470115,
    "final_old_accuracy_mean": 0.3900930097361815,
    "final_old_accuracy_sd": 0.07538682290663179,
    "final_old_accuracy_wins": 20,
    "final_old_accuracy_approx_95ci_low": 0.357053277853296,
    "final_old_accuracy_approx_95ci_high": 0.423132741619067,
    "final_new_accuracy_mean": 0.3285179995321654,
    "final_new_accuracy_sd": 0.17631054111025746,
    "final_new_accuracy_wins": 20,
    "final_new_accuracy_approx_95ci_low": 0.25124649793799325,
    "final_new_accuracy_approx_95ci_high": 0.4057895011263375
  },
  "chevron_buffer_minus_chevron_immediate": {
    "return_per_decision_mean": 0.0009999999999999926,
    "return_per_decision_sd": 0.0570636251034012,
    "return_per_decision_wins": 9,
    "return_per_decision_approx_95ci_low": -0.024009236375658636,
    "return_per_decision_approx_95ci_high": 0.026009236375658624,
    "final_old_accuracy_mean": -0.036933585248879804,
    "final_old_accuracy_sd": 0.046174209670973985,
    "final_old_accuracy_wins": 1,
    "final_old_accuracy_approx_95ci_low": -0.0571703248886785,
    "final_old_accuracy_approx_95ci_high": -0.016696845609081103,
    "final_new_accuracy_mean": 0.04928542884175021,
    "final_new_accuracy_sd": 0.13274621654893023,
    "final_new_accuracy_wins": 14,
    "final_new_accuracy_approx_95ci_low": -0.008893165694275632,
    "final_new_accuracy_approx_95ci_high": 0.10746402337777605
  },
  "chevron_buffer_minus_chevron_coupled_write": {
    "return_per_decision_mean": 0.015833333333333324,
    "return_per_decision_sd": 0.01958263343280835,
    "return_per_decision_wins": 14,
    "return_per_decision_approx_95ci_low": 0.0072508658246262965,
    "return_per_decision_approx_95ci_high": 0.024415800842040352,
    "final_old_accuracy_mean": 0.018889955443221994,
    "final_old_accuracy_sd": 0.038815036397237226,
    "final_old_accuracy_wins": 13,
    "final_old_accuracy_approx_95ci_low": 0.0018785156962852038,
    "final_old_accuracy_approx_95ci_high": 0.03590139519015878,
    "final_new_accuracy_mean": -0.00551242096086585,
    "final_new_accuracy_sd": 0.05269684531974055,
    "final_new_accuracy_wins": 8,
    "final_new_accuracy_approx_95ci_low": -0.028607831714473224,
    "final_new_accuracy_approx_95ci_high": 0.017582989792741525
  }
}
```

This is a delayed contextual-bandit RL experiment. It does not yet establish
spatial game performance, PPO trainability, or a persistent core self.

The frozen confirmation was not run because this development configuration did
not meet the protocol's decision rule. See
`experiment_004_development_findings.md` for interpretation and the declared
capacity-four diagnostic.
