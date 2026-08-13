# Experiment 008: frozen consequence-geometry protocol

## Question

Can an explicitly relational consequence objective learn a comparison geometry
that supports the confirmed Chevron gate better than temporal similarity and
one-step action prediction?

Experiments 006 and 007 showed that invariance and predictive accuracy alone do
not force cosine distance to mean behavioural compatibility. Experiment 008
therefore supervises relations between states by the consequences available
under actions, while leaving the Chevron gate and memory mechanics frozen.

## Scope

This is a dense-affordance upper-bound diagnostic, not yet a claim that sparse
reinforcement learning discovers the geometry. Each pretraining state is
evaluated under all four actions. This counterfactual access is intentionally
strong: failure would rule out proceeding directly to the harder sampled-
outcome version, while success would justify that next reduction.

The learner receives no category IDs, address families, memory contents,
compatibility labels, provisional-buffer state, delayed-context trajectories,
or audit labels. Dense action consequences do reveal which actions are better;
that is the declared supervision being tested rather than hidden leakage.

## Fixed affordance world

A fixed seed-808 world maps a 12-dimensional unit latent state to four bounded
immediate rewards. The four action-conditioned transitions are the frozen
orthogonal dynamics from Experiment 007. For state `x`, action `a`, discount
`gamma = 0.8`, and fixed dynamics `T_a`, define:

```text
R_a(x) = tanh(2.5 * (b_a dot x + c_a))
Q_a(x) = R_a(x) + gamma * max_b R_b(T_a(x))
C(x) = normalize(Q(x) - mean(Q(x)))
```

`C(x)` is the consequence signature. The delayed-context correct action is
`argmax_a Q_a(x)`. Lifetime construction retains four address families with
two established contexts and one later novel context per family. The three
contexts in each family are required to have different optimal actions, so
broad address retrieval cannot solve compatibility by itself.

The same fixed nonlinear sensor from Experiments 006 and 007 hides latent
geometry. Downstream outcomes remain delayed by three decisions.

## Consequence-metric encoder

The encoder remains the same 812-parameter two-layer network. For a batch of
latent states, two independently noisy sensor views are produced. The encoder
is trained so the cross-view cosine matrix matches the cosine matrix of the
consequence signatures:

```text
z_1 = normalize(Encoder(o_1))
z_2 = normalize(Encoder(o_2))
S_z = z_1 z_2^T
S_C = C(x) C(x)^T
L_consequence = mean((S_z - S_C)^2)
```

This objective constrains the comparison metric itself. Unlike Experiment 007,
it cannot succeed merely by letting an auxiliary predictor adapt to arbitrary
encoder coordinates. Training uses the existing 500-step, batch-256, AdamW
budget and the same observation-noise distribution as the previous encoders.

## Downstream conditions

All learned encoders are frozen before the delayed-memory evaluation.

1. Oracle latent geometric Chevron.
2. Raw-sensor geometric Chevron.
3. Temporal-contrastive geometric Chevron.
4. Action-predictive geometric Chevron.
5. Consequence-metric geometric Chevron.
6. Consequence-metric content attention using the same representation.

The geometric read threshold, slope, stricter write threshold, four-entry
provisional buffer, promotion support, delayed eligibility, and memory-update
rules are inherited unchanged from Experiment 005a.

## Development and confirmation

Development uses encoder seeds 0 and 1 and ten paired fresh lifetimes per seed.
Each learned objective receives 500 updates. The evaluation lifetimes are new
and shared by all six conditions.

Confirmation is triggered only if consequence-metric Chevron:

- reaches at least 0.95 final old accuracy;
- reaches at least 0.75 final novel and clean novel-probe accuracy;
- reaches at least 0.15 residual calibration and three promotions;
- has a paired return interval with lower bound above zero versus both temporal
  and action-predictive Chevron;
- has paired return and novel-accuracy lower bounds above -0.05 versus the
  oracle;
- has a return lower bound above -0.05 versus content attention using the same
  representation;
- reaches consequence-cosine correlation of at least 0.85 on held-out states;
  and
- improves consequence-cosine correlation by at least 0.20 over the raw
  nonlinear sensor.

If triggered, confirmation trains untouched encoder seeds 900 through 909 and
evaluates twenty new lifetimes per seed. No parameter, threshold, or task rule
then changes.

## Interpretation

Passing would show that consequence relations can provide the geometry needed
by Chevron assent under dense counterfactual supervision. It would justify a
follow-up that learns the same relation from sampled action, reward, and next-
observation tuples before entering a spatial environment.

Failure would show that even directly metric-aligned affordance supervision is
insufficient under the fixed gate and task. The project should then inspect the
meaning of compatibility or the task construction rather than tuning encoder
width, training duration, or gate thresholds.

## Development outcome

The frozen gate failed. Consequence-cosine correlation improved from 0.107 in
the raw nonlinear sensor to 0.488 in the learned representation, but missed the
0.85 criterion. Consequence-metric Chevron reached 0.611 return, 0.893 final old
accuracy, 0.540 final novel accuracy, 0.662 novel-probe accuracy, 0.154 q
calibration, and 2.45 promotions. It was worse than both temporal-contrastive
Chevron (0.699 return) and action-predictive Chevron (0.678 return).

Confirmation was not run. A post-development ideal-target audit then supplied
the exact consequence signature directly to the frozen gate. It also failed,
reaching 0.494 return and 0.554 novel accuracy. Across the same lifetimes, 13.8%
of distinct within-family context pairs had consequence similarity above the
0.62 assent boundary. The result localises the problem beyond encoder capacity:
consequence similarity alone is not a sufficient memory-identity relation.
