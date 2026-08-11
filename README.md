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

## Experiment 003: delayed buffer bridge

The final supervised pre-RL experiment gives Chevron and a conventional direct
MLP the same A/N evidence and the same 314-parameter budget. It then introduces
nearby new categories whose outcomes arrive three steps late, comparing
capacity-1 and capacity-2 provisional buffers with candidates interposed into
N immediately.

Run the frozen ten-seed confirmation:

```bash
python -m experiments.experiment_003_delayed_buffer
```

See the [frozen protocol](experiments/experiment_003_protocol.md), generated
[Experiment 003 report](experiments/results/experiment_003_report.md), and
[raw results](experiments/results/experiment_003_results.json).

## Experiment 004: reward-derived memory

The first RL bridge replaces direct compatibility labels with actions and a
scalar reward arriving three decisions later. Eight retained context-action
memories are followed by four nearby novel contexts. It compares conventional
content attention, a parameter-matched direct MLP, full Chevron buffering,
immediate writing, and coupled read/write gates.

Run the development configuration:

```bash
python -m experiments.experiment_004_reward_memory --label development
```

The development run did not trigger confirmation: Chevron's learned residual
did not separate unresolved from resolved observations strongly enough, and a
capacity-four diagnostic showed that buffer size was not the main limitation.
The separate write gate produced a small exploratory protection benefit, while
the provisional buffer correctly prevented writes before delayed reward.

See the [protocol](experiments/experiment_004_protocol.md),
[development findings](experiments/results/experiment_004_development_findings.md),
[development report](experiments/results/experiment_004_development_report.md),
and [raw development results](experiments/results/experiment_004_development_results.json).

## Experiment 005: retrospective assent

Experiment 005 keeps the delayed-context task fixed and asks whether reward can
train assent more directly. When an outcome arrives, a retrospective loss
scores whether the admitted memory support predicted the selected action's
success or failure. A projected bilinear slot-or-null attention control replaces
the underfitting MLP while matching Chevron's 314-parameter budget.

Run the development configuration:

```bash
python -m experiments.experiment_005_retrospective_assent --label development
```

The frozen development gate failed, so confirmation was not run. Retrospective
Chevron was statistically indistinguishable from policy-only Chevron and still
failed to calibrate q or promote most novel contexts. The fixed content
controller solved the same task, localising the problem to learned comparison
from sparse reward rather than memory capacity or task feasibility.

See the [protocol](experiments/experiment_005_protocol.md),
[development findings](experiments/results/experiment_005_development_findings.md),
[development report](experiments/results/experiment_005_development_report.md),
and [raw results](experiments/results/experiment_005_development_results.json).

## Experiment 005a: geometric gate isolation

Experiment 005a removes the unsuccessful reward-trained A/N projections while
retaining per-slot Chevron assent, residual mass, provisional storage, and
separate write permission. Its parameter-free half-cosine mismatch uses the
already-frozen content threshold and is evaluated without additional training.

Run the frozen development-lifetime diagnostic:

```bash
python -m experiments.experiment_005a_geometric_gate
```

Run the predeclared 100-lifetime fresh-seed confirmation:

```bash
python -m experiments.experiment_005a_geometric_gate --fresh-confirmation
```

All confirmation criteria passed. Buffered geometric Chevron reached 0.972
old-context accuracy and 0.896 novel-context accuracy, promoted 3.92 of four
novel contexts, and made no premature writes. Immediate writing reached only
0.352 novel accuracy and made premature writes on 8.56% of decisions. The
result localises the learned experiments' failure to comparison geometry rather
than the gate or buffer dynamics.

See the [protocol and outcome](experiments/experiment_005a_protocol.md),
[findings](experiments/results/experiment_005a_findings.md),
[diagnostic report](experiments/results/experiment_005a_geometric_report.md),
and [fresh-seed confirmation report](experiments/results/experiment_005a_confirmation_report.md).
