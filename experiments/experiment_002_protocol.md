# Experiment 002: frozen confirmation protocol

## Question

Can a learned Chevron assent gate distinguish retrieval from permission to read when an equally trained address-only gate cannot?

## Development and freeze

Seeds 0 and 1 were used only for development. An unweighted binary-cross-entropy loss was too conservative: it rejected unfamiliar cases reliably but also rejected too many noisy familiar cases. A positive-match loss weight of 2.0 gave the better stability/plasticity balance in the development run:

- familiar-case accuracy: 92.84%
- no-match accuracy: 88.98%
- overall accuracy: 91.56%

The positive-match weight of 2.0 and all other settings are frozen before the confirmation run. They will not be retuned on confirmation results.

## Confirmation

- Training seeds: 100–109
- Optimisation: 800 steps per seed, batch size 256
- Evaluation: 4,096 fresh examples at each of six noise levels per seed
- Total evaluation examples: 245,760
- Primary learned model: asymmetric Chevron assent from diagnostic A evidence and retained N content
- Parameter-matched control: a second learned retrieval computation using only A-channel address evidence
- Reference: ordinary attention without abstention

All methods share the same retrieval distribution. The two learned gates receive the same per-slot supervision, but only the Chevron gate receives evidence that can resolve which family member is compatible. Evaluation uses new memories and queries while retaining the coordinate transforms learned during training. Generalisation to unseen transforms is outside this experiment.

## Decision rule

The result supports proceeding if the learned Chevron gate reliably improves both familiar-case acceptance and no-match rejection over the controls across fresh seeds. This remains a supervised mechanism diagnostic, not evidence of an autonomous agent or reinforcement-learning benefit.
