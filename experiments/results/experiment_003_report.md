# Experiment 003: equal information and delayed buffering

## Protocol

- Confirmation seeds: 200–209
- Pretraining steps per seed: 2000
- Sequential observations per seed: 600
- Distribution shift at step: 200
- Retrospective outcome delay: 3 steps
- Chevron parameters: 314
- Direct MLP parameters: 314
- Both learned models receive the same A evidence, N content, retrieval prior, and labels.
- Chevron uses its residual threshold; the direct classifier uses its native slots-plus-null argmax.

## Part A: equal-information held-out diagnostic

| Method | Overall | Familiar | No match |
|---|---:|---:|---:|
| Standard attention | 33.39 +/- 0.33% | 50.14 +/- 0.42% | 0.00 +/- 0.00% |
| Direct slot-or-null MLP | 78.20 +/- 1.52% | 80.16 +/- 0.93% | 74.30 +/- 3.62% |
| Chevron assent | 91.88 +/- 0.20% | 91.69 +/- 0.27% | 92.25 +/- 0.49% |

## Part B: delayed sequential consolidation

| Method | Overall | Before shift | After shift | Initial categories | Revealed novel |
|---|---:|---:|---:|---:|---:|
| Standard attention | 38.47 +/- 1.61% | 48.65 +/- 2.70% | 33.38 +/- 1.43% | 49.75 +/- 2.03% | 0.00 +/- 0.00% |
| Direct MLP + buffer 2 | 92.57 +/- 2.02% | 94.95 +/- 1.52% | 91.38 +/- 2.90% | 94.70 +/- 1.45% | 87.18 +/- 4.30% |
| Chevron + buffer 1 | 93.75 +/- 1.42% | 96.10 +/- 0.77% | 92.58 +/- 2.24% | 94.66 +/- 1.08% | 91.72 +/- 2.99% |
| Chevron + buffer 2 | 93.97 +/- 1.34% | 96.10 +/- 0.77% | 92.90 +/- 2.10% | 94.62 +/- 1.08% | 92.80 +/- 2.94% |
| Chevron candidates in N | 90.98 +/- 2.08% | 95.80 +/- 1.12% | 88.58 +/- 3.31% | 91.88 +/- 1.63% | 88.97 +/- 3.79% |

## Consolidation and protection

| Method | Buffer evictions | Premature N writes | Established overwrites | Initial retained / 8 | Novel learned / 4 | Initial probe | Novel probe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard attention | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 8.00 +/- 0.00 | 0.00 +/- 0.00 | 50.00 +/- 0.00% | 0.00 +/- 0.00% |
| Direct MLP + buffer 2 | 0.20 +/- 0.60 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 8.00 +/- 0.00 | 4.00 +/- 0.00 | 100.00 +/- 0.00% | 100.00 +/- 0.00% |
| Chevron + buffer 1 | 5.00 +/- 3.52 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 8.00 +/- 0.00 | 4.00 +/- 0.00 | 100.00 +/- 0.00% | 100.00 +/- 0.00% |
| Chevron + buffer 2 | 0.30 +/- 0.64 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 8.00 +/- 0.00 | 4.00 +/- 0.00 | 100.00 +/- 0.00% | 100.00 +/- 0.00% |
| Chevron candidates in N | 0.00 +/- 0.00 | 45.30 +/- 8.12 | 15.10 +/- 4.91 | 7.50 +/- 0.50 | 3.90 +/- 0.30 | 93.75 +/- 6.25% | 97.50 +/- 7.50% |

## Findings

With equal inputs and exactly equal parameter counts, Chevron reached 91.88% on the held-out diagnostic versus 78.20% for the direct MLP. This supports a useful comparison-and-residual inductive bias at this parameter budget; it does not show that an MLP cannot learn the task.

In the sequential task, the direct MLP with the same separate buffer still reached 92.57% and learned all categories. The delayed-buffer result therefore does not depend on the projected-cosine gate alone.

Chevron with a separate capacity-2 buffer beat candidate interposition by 2.98 +/- 0.97 paired percentage points and won on 10/10 seeds. The separate buffer retained all eight initial categories and learned all four new ones on every seed, without any pre-outcome N write. Interposition caused 45.3 premature writes and 15.1 established overwrites per lifetime.

A one-entry buffer also retained and learned every category, but averaged 5.0 evictions and was slightly less accurate. Capacity two is safer for the tested three-step delay, but depth one remains viable under this load.

## Claim boundary

This experiment can test equal-information compatibility learning and delayed protected consolidation. It cannot establish an RL, game-solving, or general-agency advantage.
The stream is synthetic, outcomes directly supervise compatibility, and the replacement policy intentionally exposes the cost of putting unresolved candidates into established memory.

## RL decision

Proceed to a small reinforcement-learning environment. The pre-RL criteria are met: both learned same-information systems acquired all new categories while retaining the original set, and separate buffering consistently protected N. The first RL test should preserve the capacity-2 buffer and derive assent or write eligibility from retrospective reward rather than supplied compatibility labels.
