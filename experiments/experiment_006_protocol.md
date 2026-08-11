# Experiment 006: frozen temporal-contrastive geometry protocol

## Question

Can a small encoder learn the comparison geometry required by the confirmed
Chevron gate from dense temporal consistency rather than category labels,
compatibility labels, correct actions, or sparse reward?

## Hidden geometry

The delayed-context task from Experiment 005a remains unchanged in its latent
12-dimensional space. Before the agent sees an observation, a fixed nonlinear
sensor maps it through two random mixing layers and a tanh nonlinearity back to
12 dimensions. This deliberately distorts raw cosine similarity while
preserving the underlying information.

The same sensor is used for all development and confirmation seeds. Its weights
are fixed from seed 606 and are never trained.

## Temporal training signal

Pretraining samples an unlabelled latent state and produces two independently
noised observations of that same persisting state. After the nonlinear sensor,
the pair represents two temporally adjacent views. Other pairs in the batch are
negatives.

A shared encoder is trained with symmetric InfoNCE:

```text
e_1 = normalize(Encoder(sensor(view_1)))
e_2 = normalize(Encoder(sensor(view_2)))

L = 0.5 * CE(e_1 e_2^T / temperature, diagonal)
  + 0.5 * CE(e_2 e_1^T / temperature, diagonal)
```

The encoder has two linear layers, hidden width 32, GELU activation, and a
12-dimensional normalised output. Training uses 500 steps, batch size 256,
view noise 0.15, temperature 0.10, AdamW learning rate 0.003, and weight decay
0.00001.

No latent identity is supplied to the encoder. The pairing says only that two
views are temporally contiguous observations of the same persisting state.
Action, reward, memory state, and all audit labels are absent.

## Frozen downstream mechanism

After pretraining, the encoder is frozen. Clean initial N templates, online A
evidence, and provisional candidates all pass through the same sensor and
encoder. Chevron then uses the confirmed Experiment 005a gate unchanged:

```text
M_j = 0.5 * (1 - cosine(A, N_j))
r_read_j = sigmoid(40 * (0.19 - M_j))
r_write_j = sigmoid(40 * (0.14 - M_j))
```

The four-entry buffer, two-positive-outcome promotion rule, memory rates, and
delayed-context lifetime are unchanged.

## Conditions

1. Oracle geometric Chevron: latent geometry, establishing the attainable
   downstream result.
2. Raw-sensor geometric Chevron: no encoder.
3. Random-encoder geometric Chevron: the same untrained architecture.
4. Temporal-contrastive geometric Chevron: the proposed system.
5. Temporal-contrastive content attention: conventional downstream comparison
   using the same learned representation.

## Development and confirmation

Development uses encoder seeds 0 and 1. Each encoder is evaluated on ten fresh
lifetimes, for 20 paired downstream evaluations per condition.

Confirmation is triggered only if temporal-contrastive geometric Chevron:

- reaches at least 0.95 final old accuracy;
- reaches at least 0.75 final novel and clean novel-probe accuracy;
- reaches at least 0.15 residual calibration and three promotions;
- has paired return intervals with lower bounds above zero versus both raw
  sensor and random encoder;
- has paired return and final-novel lower bounds above -0.05 versus the oracle;
- has a lower-bound return above -0.05 versus content attention using the same
  encoder; and
- produces a temporal-pair cosine gap of at least 0.30 and latent/encoded
  pairwise-cosine correlation of at least 0.60.

If triggered, confirmation trains encoder seeds 700 through 709 and evaluates
each on twenty untouched lifetimes. All sensor, encoder, optimiser, gate, and
memory parameters remain frozen.

## Interpretation

Passing would show that Chevron does not require a handed-in semantic geometry:
a dense, label-free temporal signal can learn one sufficient for delayed
category acquisition. That would justify moving to a small spatial environment
where temporal continuity arises naturally.

Failure would leave the geometric gate result intact but show that this simple
contrastive representation learner is not an adequate bridge to agency.

## Development outcome

The two-seed development run improved latent-cosine correlation from 0.453 in
raw sensor space to 0.784 after temporal contrastive training. Temporal-pair
cosine separation reached 0.752. Downstream return improved from 0.346 for raw
sensor geometry and -0.045 for a random encoder to 0.732.

However, temporal geometric Chevron reached only 0.937 final old accuracy,
0.727 final novel accuracy, 0.135 residual calibration, and 3.65 promotions.
It missed the frozen thresholds of 0.95, 0.75, and 0.15 respectively, and its
paired performance remained more than 0.05 below the oracle. Confirmation was
therefore not triggered.

The result is encouraging representation learning but a negative go/no-go
result. Temporal instance consistency recovers substantial geometry without
labels, yet does not fully recover the consequence-relevant distinctions needed
by the confirmed gate.
