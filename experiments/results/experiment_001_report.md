# Experiment 001: causal factorisation

## Question

Can retrieval remain fixed while independent A/N compatibility changes
read assent, residual mass, allocation, and write permission?

## Protocol

- Seeds: 50
- Queries per method: 4800
- Each address retrieves a two-slot family and cannot distinguish its members.
- Diagnostic evidence matches the first member, the second member, or neither.
- No parameters are trained; this is a causal mathematical diagnostic, not RL.

## Results

| Method | Overall accuracy | Match | No match | No-match q | No-match N drift |
|---|---:|---:|---:|---:|---:|
| Standard attention | 33.33% | 50.00% | 0.00% | 0.0000 | 0.147335 |
| Retrieval twice | 33.33% | 50.00% | 0.00% | 0.0263 | 0.141392 |
| Chevron A/N assent | 100.00% | 100.00% | 100.00% | 0.9999 | 0.000000 |

## Causal checks

- Maximum change in retrieval alpha when only diagnostic evidence changed: 0.00000000
- Mean Chevron assent switch margin: 0.993254
- Maximum change in retrieval-twice assent under the same intervention: 0.00000000

## Evidence-noise stress test

Additional seeds: 20

| Evidence noise | Chevron overall | Chevron match | Chevron no match |
|---:|---:|---:|---:|
| 0.10 | 100.00% | 100.00% | 100.00% |
| 0.15 | 99.38% | 99.06% | 100.00% |
| 0.20 | 93.12% | 89.69% | 100.00% |
| 0.25 | 76.25% | 64.38% | 100.00% |
| 0.30 | 61.88% | 42.81% | 100.00% |
| 0.40 | 47.29% | 20.94% | 100.00% |

## Allocation and write selectivity

| Method | Allocate on no match | False allocation on match | Target write | Non-target write |
|---|---:|---:|---:|---:|
| Standard attention | 0.00% | 0.00% | 0.488043 | 0.511957 |
| Retrieval twice | 0.00% | 0.00% | 0.479246 | 0.479246 |
| Chevron A/N assent | 100.00% | 0.00% | 0.466809 | 0.000020 |

## Claim boundary

This engineered diagnostic tests the declared equations and causal separation. It does not show that an RL optimizer will learn the factorisation from reward, or that Chevron improves game performance.
