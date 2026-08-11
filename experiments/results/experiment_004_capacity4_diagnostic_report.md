# Experiment 004: capacity-four development diagnostic

- Training seeds: 0–0
- Training lifetimes per seed: 60
- Fresh evaluation lifetimes per seed: 10
- Reward delay: 3
- Provisional buffer capacity: 4
- Learned comparator parameters: 314 each
- Learning signal: delayed scalar reward only

| Method | Return/decision | Final old | Final new | Old probe | New probe | q calibration | N drift | Premature writes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Content attention + buffer | 0.818 +/- 0.049 | 0.977 +/- 0.014 | 0.848 +/- 0.094 | 0.988 +/- 0.037 | 0.925 +/- 0.115 | 0.477 +/- 0.134 | 0.053 +/- 0.007 | 0.000 +/- 0.000 |
| Direct MLP + buffer | -0.227 +/- 0.036 | 0.483 +/- 0.044 | 0.046 +/- 0.071 | 0.487 +/- 0.037 | 0.025 +/- 0.075 | -0.002 +/- 0.009 | 0.346 +/- 0.029 | 0.000 +/- 0.000 |
| Chevron + buffer | 0.552 +/- 0.081 | 0.832 +/- 0.078 | 0.401 +/- 0.198 | 0.875 +/- 0.079 | 0.400 +/- 0.278 | 0.038 +/- 0.045 | 0.123 +/- 0.023 | 0.000 +/- 0.000 |
| Chevron immediate write | 0.565 +/- 0.062 | 0.893 +/- 0.079 | 0.367 +/- 0.135 | 0.938 +/- 0.084 | 0.425 +/- 0.195 | 0.049 +/- 0.052 | 0.090 +/- 0.022 | 0.014 +/- 0.007 |
| Chevron coupled write | 0.540 +/- 0.086 | 0.818 +/- 0.081 | 0.408 +/- 0.185 | 0.875 +/- 0.079 | 0.425 +/- 0.275 | 0.042 +/- 0.046 | 0.138 +/- 0.025 | 0.000 +/- 0.000 |

## Paired diagnostics

```json
{
  "chevron_buffer_minus_content_attention_buffer": {
    "return_per_decision_mean": -0.2653333333333333,
    "return_per_decision_sd": 0.08773135199834088,
    "return_per_decision_wins": 0,
    "return_per_decision_approx_95ci_low": -0.3197097886593948,
    "return_per_decision_approx_95ci_high": -0.2109568780072718,
    "final_old_accuracy_mean": -0.1451955415876411,
    "final_old_accuracy_sd": 0.07879511749902508,
    "final_old_accuracy_wins": 0,
    "final_old_accuracy_approx_95ci_low": -0.194033261387953,
    "final_old_accuracy_approx_95ci_high": -0.09635782178732924,
    "final_new_accuracy_mean": -0.44670437304725236,
    "final_new_accuracy_sd": 0.1743615747233217,
    "final_new_accuracy_wins": 0,
    "final_new_accuracy_approx_95ci_low": -0.554774796704963,
    "final_new_accuracy_approx_95ci_high": -0.3386339493895418
  },
  "chevron_buffer_minus_direct_mlp_buffer": {
    "return_per_decision_mean": 0.779,
    "return_per_decision_sd": 0.07025614511148419,
    "return_per_decision_wins": 10,
    "return_per_decision_approx_95ci_low": 0.7354547901175835,
    "return_per_decision_approx_95ci_high": 0.8225452098824165,
    "final_old_accuracy_mean": 0.3495881917456909,
    "final_old_accuracy_sd": 0.09799967263750503,
    "final_old_accuracy_wins": 10,
    "final_old_accuracy_approx_95ci_low": 0.28884736535055316,
    "final_old_accuracy_approx_95ci_high": 0.4103290181408286,
    "final_new_accuracy_mean": 0.35567889125164664,
    "final_new_accuracy_sd": 0.18537224770132463,
    "final_new_accuracy_wins": 10,
    "final_new_accuracy_approx_95ci_low": 0.24078398177831142,
    "final_new_accuracy_approx_95ci_high": 0.4705738007249819
  },
  "chevron_buffer_minus_chevron_immediate": {
    "return_per_decision_mean": -0.013000000000000017,
    "return_per_decision_sd": 0.08817000973848166,
    "return_per_decision_wins": 5,
    "return_per_decision_approx_95ci_low": -0.0676483382101945,
    "return_per_decision_approx_95ci_high": 0.041648338210194466,
    "final_old_accuracy_mean": -0.06097201856857918,
    "final_old_accuracy_sd": 0.04903124959188499,
    "final_old_accuracy_wins": 0,
    "final_old_accuracy_approx_95ci_low": -0.09136190191455251,
    "final_old_accuracy_approx_95ci_high": -0.03058213522260584,
    "final_new_accuracy_mean": 0.03388280283017124,
    "final_new_accuracy_sd": 0.1957800696418365,
    "final_new_accuracy_wins": 8,
    "final_new_accuracy_approx_95ci_low": -0.0874629415146082,
    "final_new_accuracy_approx_95ci_high": 0.1552285471749507
  },
  "chevron_buffer_minus_chevron_coupled_write": {
    "return_per_decision_mean": 0.011999999999999988,
    "return_per_decision_sd": 0.018605720612386214,
    "return_per_decision_wins": 7,
    "return_per_decision_approx_95ci_low": 0.000468054889798869,
    "return_per_decision_approx_95ci_high": 0.023531945110201107,
    "final_old_accuracy_mean": 0.014214167812355405,
    "final_old_accuracy_sd": 0.03066128122443103,
    "final_old_accuracy_wins": 7,
    "final_old_accuracy_approx_95ci_low": -0.004789891178683645,
    "final_old_accuracy_approx_95ci_high": 0.03321822680339445,
    "final_new_accuracy_mean": -0.006720018631783359,
    "final_new_accuracy_sd": 0.046952909689063065,
    "final_new_accuracy_wins": 2,
    "final_new_accuracy_approx_95ci_low": -0.03582173356015036,
    "final_new_accuracy_approx_95ci_high": 0.022381696296583643
  }
}
```

This is a delayed contextual-bandit RL experiment. It does not yet establish
spatial game performance, PPO trainability, or a persistent core self.

This one-training-seed run is a post-development capacity diagnostic, not a
confirmation result.
