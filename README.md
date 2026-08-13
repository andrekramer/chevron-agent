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

## Experiment 006: temporal-contrastive geometry

Experiment 006 hides the useful latent geometry behind a fixed nonlinear
sensor. A small 812-parameter encoder receives paired adjacent views of
persisting states and trains with symmetric InfoNCE. It receives no category,
action, reward, compatibility, or memory labels. The encoder is frozen before
the confirmed geometric Chevron mechanism is applied.

```bash
python -m experiments.experiment_006_predictive_geometry --label development
```

The encoder raised latent-cosine correlation from 0.453 to 0.784 and downstream
return from 0.346 to 0.732, but narrowly missed the frozen retention,
acquisition, calibration, and oracle-distance criteria. Confirmation was not
run.

Experiment 006a tested whether the remaining gap was only a mismatch-scale
problem by deriving the threshold and slope from unlabelled temporal-positive
and hard-negative distributions:

```bash
python -m experiments.experiment_006a_calibrated_gate --label development
```

Calibration did not improve the inherited gate, so its confirmation was also
withheld. See the [Experiment 006 protocol](experiments/experiment_006_protocol.md),
[Experiment 006 findings](experiments/results/experiment_006_development_findings.md),
[Experiment 006a protocol](experiments/experiment_006a_protocol.md), and
[Experiment 006a findings](experiments/results/experiment_006a_development_findings.md).

## Experiment 007: action-conditioned prediction

Experiment 007 trains the same compact encoder to predict the next embedding
under one of four fixed latent actions. Pretraining receives observations,
random actions, and next observations, but no categories, downstream policy,
reward, compatibility, or memory labels.

```bash
python -m experiments.experiment_007_action_prediction --label development
```

The forward model learned its task, producing a 0.643 cosine gap between true
and permuted next embeddings. Nevertheless, it recovered less useful comparison
geometry than temporal contrastive learning and reduced Chevron return from
0.738 to 0.700 and novel accuracy from 0.753 to 0.632. Confirmation was not
triggered.

See the [protocol and outcome](experiments/experiment_007_protocol.md),
[development findings](experiments/results/experiment_007_development_findings.md),
[report](experiments/results/experiment_007_development_report.md), and
[raw results](experiments/results/experiment_007_development_results.json).

## Experiment 008: consequence geometry

Experiment 008 directly aligns embedding cosine with dense counterfactual
action-consequence similarity. The task's correct actions are generated by the
same fixed affordance world, while the encoder receives no category, memory, or
compatibility labels. The Chevron gate and provisional buffer remain frozen.

```bash
python -m experiments.experiment_008_consequence_geometry --label development
```

The objective raised held-out consequence-cosine correlation from 0.107 to
0.488, but consequence Chevron reached only 0.611 return versus 0.699 for
temporal Chevron and 0.678 for action-predictive Chevron. The confirmation gate
failed. A post-development oracle-target audit then showed that the exact
consequence signature was itself insufficient: it reached 0.494 return, and
13.8% of distinct contexts sharing an address family remained above the assent
similarity boundary.

The result distinguishes behavioural consequence similarity from memory
identity. The next hypothesis should keep identity/persistence and consequence
compatibility as separate relations rather than compressing both into one
cosine geometry.

See the [protocol and outcome](experiments/experiment_008_protocol.md),
[development findings](experiments/results/experiment_008_development_findings.md),
[development report](experiments/results/experiment_008_development_report.md),
[raw development results](experiments/results/experiment_008_development_results.json),
and [ideal-target audit](experiments/results/experiment_008_target_audit_report.md).

## Experiment 009: dual-relation assent

Experiment 009 separates memory identity from policy compatibility. After the
task shift, novel contexts require new identities while familiar contexts with
reversed actions require policy revision within an existing identity.

```bash
python -m experiments.experiment_009_dual_relation --label development
```

Dual routing beat a collapsed gate by +0.047 return and improved reversal
accuracy by +0.211 while avoiding duplicate identities. It did not beat the
simpler identity-only agent, which adapted action values directly from delayed
reward and reached 0.746 return versus 0.587. A shared four-entry queue also
caused 78.85 dual candidate evictions and depressed novel acquisition.

The confirmation gate failed. The result supports keeping identity and policy
mismatch conceptually distinct, but not the tested conservative revision path
as a performance improvement. The next diagnostic should separate their
provisional queues before introducing learned representations.

See the [protocol and outcome](experiments/experiment_009_protocol.md),
[development findings](experiments/results/experiment_009_development_findings.md),
[development report](experiments/results/experiment_009_development_report.md),
and [raw development results](experiments/results/experiment_009_development_results.json).

## Experiment 009a: split queues or more capacity?

Experiment 009a separates routing from capacity by comparing shared four-entry,
split 2+2, shared eight-entry, and split 4+4 provisional layouts.

```bash
python -m experiments.experiment_009a_split_queues --label development
```

Separate fixed queues did not help. Shared 8 and Split 4+4 had statistically
indistinguishable return, while the split layout caused 3.30 more evictions.
Capacity was decisive: Shared 8 improved return over Shared 4 by +0.138 and
novel accuracy by +0.266, with both paired intervals above zero, while reducing
evictions from 78.85 to 22.75.

The current design implication is to keep candidate types and consolidation
destinations separate while sharing flexible provisional capacity. The routing
confirmation gate failed; the capacity result remains a strong development
finding that should receive a narrowly scoped fresh-seed confirmation.

See the [protocol and outcome](experiments/experiment_009a_protocol.md),
[development findings](experiments/results/experiment_009a_development_findings.md),
[development report](experiments/results/experiment_009a_development_report.md),
and [raw development results](experiments/results/experiment_009a_development_results.json).

## Experiment 009b: shared-capacity confirmation

Experiment 009b confirms the strong Experiment 009a capacity diagnostic on 100
untouched lifetimes, comparing only dual Shared 4, dual Shared 8, and the
identity-only control.

```bash
python -m experiments.experiment_009b_capacity_confirmation
```

Every predeclared criterion passed. Shared 8 improved return over Shared 4 by
+0.156, reversed accuracy by +0.164, and novel accuracy by +0.278; all paired
intervals excluded zero. Evictions fell by 76.7%, with no premature writes,
established overwrites, or duplicate identities.

Shared-8 dual Chevron was non-inferior overall to identity-only adaptation:
return 0.750 versus 0.760, paired difference interval -0.023 to +0.003. It was
slightly slower on deterministic reversals but significantly better on novel
acquisition.

See the [frozen protocol and outcome](experiments/experiment_009b_protocol.md),
[confirmation findings](experiments/results/experiment_009b_confirmation_findings.md),
[confirmation report](experiments/results/experiment_009b_confirmation_report.md),
and [raw confirmation results](experiments/results/experiment_009b_confirmation_results.json).
