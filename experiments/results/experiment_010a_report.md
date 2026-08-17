# Experiment 010a: pre-consolidation identity revalidation

- Fresh seeds: 102000000–102000099
- Changed mechanism: promotion-time identity revalidation only

| Condition | Return | Clean accuracy | Retention | Reversed probe | Novel probe | Duplicates | Reconciliations |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original protected Chevron | 0.531 +/- 0.065 | 0.835 +/- 0.035 | 0.948 +/- 0.022 | 0.980 +/- 0.068 | 0.915 +/- 0.151 | 0.010 +/- 0.099 | 0.000 +/- 0.000 |
| Revalidated protected Chevron | 0.531 +/- 0.065 | 0.835 +/- 0.035 | 0.948 +/- 0.022 | 0.980 +/- 0.068 | 0.915 +/- 0.151 | 0.000 +/- 0.000 | 0.010 +/- 0.099 |

## Frozen audit gate

- PASS: `zero_duplicate_allocations`
- PASS: `zero_established_overwrites`
- PASS: `zero_under_supported_writes`
- PASS: `retention_at_least_0.90`
- PASS: `reversed_probe_at_least_0.75`
- PASS: `novel_probe_at_least_0.75`
- PASS: `new_promotions_at_least_3`
- PASS: `unique_revisions_at_least_3`
- PASS: `return_noninferior`
- PASS: `clean_accuracy_noninferior`
- PASS: `retention_noninferior`
- PASS: `reversed_probe_noninferior`
- PASS: `novel_probe_noninferior`
- PASS: `reconciliations_cover_original_duplicates`

Correction passed: **True**

## Paired diagnostics

```json
{
  "protected_revalidated_minus_original": {
    "return_per_decision_mean": -2.4999999999999466e-05,
    "return_per_decision_population_sd": 0.00024874685927664966,
    "return_per_decision_approx_95ci_low": -7.37543844182228e-05,
    "return_per_decision_approx_95ci_high": 2.375438441822387e-05,
    "return_per_decision_wins": 0,
    "clean_accuracy_mean": 1.2499999999999733e-05,
    "clean_accuracy_population_sd": 0.00012437342963832483,
    "clean_accuracy_approx_95ci_low": -1.1877192209111935e-05,
    "clean_accuracy_approx_95ci_high": 3.68771922091114e-05,
    "clean_accuracy_wins": 1,
    "retention_accuracy_mean": -5.000000000000004e-05,
    "retention_accuracy_population_sd": 0.0004974937185533104,
    "retention_accuracy_approx_95ci_low": -0.00014750876883644888,
    "retention_accuracy_approx_95ci_high": 4.75087688364488e-05,
    "retention_accuracy_wins": 0,
    "reversed_probe_accuracy_mean": 0.0,
    "reversed_probe_accuracy_population_sd": 0.0,
    "reversed_probe_accuracy_approx_95ci_low": 0.0,
    "reversed_probe_accuracy_approx_95ci_high": 0.0,
    "reversed_probe_accuracy_wins": 0,
    "novel_probe_accuracy_mean": 0.0,
    "novel_probe_accuracy_population_sd": 0.0,
    "novel_probe_accuracy_approx_95ci_low": 0.0,
    "novel_probe_accuracy_approx_95ci_high": 0.0,
    "novel_probe_accuracy_wins": 0,
    "new_promotions_mean": -0.01,
    "new_promotions_population_sd": 0.099498743710662,
    "new_promotions_approx_95ci_low": -0.02950175376728975,
    "new_promotions_approx_95ci_high": 0.009501753767289753,
    "new_promotions_wins": 0,
    "duplicate_allocations_mean": -0.01,
    "duplicate_allocations_population_sd": 0.099498743710662,
    "duplicate_allocations_approx_95ci_low": -0.02950175376728975,
    "duplicate_allocations_approx_95ci_high": 0.009501753767289753,
    "duplicate_allocations_wins": 0,
    "identity_reconciliations_mean": 0.01,
    "identity_reconciliations_population_sd": 0.099498743710662,
    "identity_reconciliations_approx_95ci_low": -0.009501753767289753,
    "identity_reconciliations_approx_95ci_high": 0.02950175376728975,
    "identity_reconciliations_wins": 1
  }
}
```
