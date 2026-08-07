# Experiment 002: learned assent

## Question

Can an independently supervised A/N compatibility gate learn assent
under noise while an equally trained address-only gate cannot?

## Protocol

- Independent training seeds: 10
- Confirmation seed range: 100–109
- Optimisation steps per seed: 800
- Positive-match loss weight: 2.0
- Fresh evaluation queries per seed: 24576
- A evidence and N memory use unknown independently rotated coordinates.
- Retrieval identifies a two-slot family but not its member.
- The retrieval-twice control receives the same labels but only address features.
- Learned gate parameters: Chevron 290; retrieval twice 290.
- This is supervised learning, not reinforcement learning.

## Held-out results

Mean +/- population SD across independently trained seeds.

| Method | Overall accuracy | Match | No match | No-match q |
|---|---:|---:|---:|---:|
| Standard attention | 33.47 +/- 0.21% | 50.24 +/- 0.19% | 0.00 +/- 0.00% | 0.0000 +/- 0.0000 |
| Learned retrieval twice | 33.47 +/- 0.21% | 50.24 +/- 0.19% | 0.00 +/- 0.00% | 0.5053 +/- 0.0068 |
| Learned Chevron assent | 91.51 +/- 0.15% | 92.69 +/- 0.14% | 89.14 +/- 0.24% | 0.8394 +/- 0.0005 |

## Chevron noise curve

| Evidence noise | Overall | Match | No match |
|---:|---:|---:|---:|
| 0.00 | 96.42 +/- 0.25% | 100.00 +/- 0.00% | 89.13 +/- 0.58% |
| 0.10 | 96.25 +/- 0.24% | 100.00 +/- 0.00% | 88.82 +/- 0.67% |
| 0.15 | 96.29 +/- 0.31% | 99.94 +/- 0.04% | 89.04 +/- 1.01% |
| 0.20 | 94.51 +/- 0.32% | 97.30 +/- 0.23% | 89.01 +/- 0.89% |
| 0.25 | 88.03 +/- 0.29% | 87.41 +/- 0.32% | 89.27 +/- 0.76% |
| 0.30 | 77.54 +/- 0.60% | 71.52 +/- 0.62% | 89.58 +/- 1.00% |

## Allocation and write gating

| Method | Allocate no match | False allocate match | Target write | Non-target write | No-match write |
|---|---:|---:|---:|---:|---:|
| Standard attention | 0.00 +/- 0.00% | 0.00 +/- 0.00% | 0.4995 +/- 0.0000 | 0.5005 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Learned retrieval twice | 0.00 +/- 0.00% | 0.00 +/- 0.00% | 0.1906 +/- 0.0033 | 0.1906 +/- 0.0033 | 0.3812 +/- 0.0066 |
| Learned Chevron assent | 89.14 +/- 0.24% | 7.09 +/- 0.13% | 0.2565 +/- 0.0005 | 0.0165 +/- 0.0002 | 0.1079 +/- 0.0004 |

## Gate diagnostics

- Chevron positive compatibility recall: 81.11 +/- 0.21%
- Chevron negative compatibility specificity: 99.84 +/- 0.01%
- Retrieval-twice positive recall: 37.34 +/- 25.23%
- Retrieval-twice negative specificity: 93.22 +/- 4.57%
- Learned Chevron threshold: 0.1268 +/- 0.0005
- Learned Chevron slope: 10.090 +/- 0.031

## Finding

Learned Chevron assent exceeded the parameter-matched retrieval-twice control by 58.04 percentage points overall. It learned both sides of the decision: 92.69% acceptance of familiar cases and 89.14% rejection of no-match cases. The small seed-to-seed spread shows that this was not dependent on a fortunate initialisation.

When retrieval ambiguity is held constant, a separately supervised gate with diagnostic A/N compatibility evidence can learn to admit familiar content and preserve unresolved mass for incompatible content. Merely learning a second address-based retrieval computation does not recover that distinction.

## Claim boundary

This experiment tests whether separately supervised A/N assent can be learned on fresh memories under noise. It does not show that RL reward alone learns assent or that the resulting agent solves games.

The diagnostic is synthetic, compatibility labels are supplied directly, and each trained gate is evaluated under the coordinate transforms it saw during training. Buffer dynamics, consolidation, non-stationarity, and reward-derived learning remain untested.

## Next decision

Proceed to a minimal sequential environment in which retrospective outcomes train assent and rejected evidence enters a bounded provisional buffer. That experiment should test whether the factorisation improves behaviour, rather than only supervised compatibility classification.
