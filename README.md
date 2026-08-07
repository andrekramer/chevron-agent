# Chevron Agent

A conventional reinforcement-learning agent with explicit fast adaptive (A)
and slower retained (N) channels.

See [chevron-agent-design.md](chevron-agent-design.md) for the initial architecture and experimental plan.

The current mathematical specification is [maths2.md](maths2.md).

## Experiment 001: causal factorisation

The first controlled experiment holds retrieval fixed while changing only the
compatibility between diagnostic A evidence and retained N content. It compares
standard attention, a second retrieval gate, and independent Chevron assent.

Run the invariant suite:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

Run the seeded diagnostic and evidence-noise stress test:

```bash
python -m experiments.experiment_001_factorisation
```

See the generated
[Experiment 001 report](experiments/results/experiment_001_report.md) and
[raw results](experiments/results/experiment_001_results.json).

## Experiment 002: learned assent

The second experiment replaces the hand-set compatibility rule with a learned
gate. Diagnostic A evidence and retained N content are independently encoded,
while a parameter-matched retrieval-twice control receives only address
features. The frozen confirmation uses ten fresh training seeds.

Run the confirmation experiment:

```bash
python -m experiments.experiment_002_learned_assent
```

See the [frozen protocol](experiments/experiment_002_protocol.md), generated
[Experiment 002 report](experiments/results/experiment_002_report.md), and
[raw results](experiments/results/experiment_002_results.json).
