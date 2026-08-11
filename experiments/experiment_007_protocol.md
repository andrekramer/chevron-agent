# Experiment 007: frozen action-conditioned prediction protocol

## Question

Can action-conditioned transition prediction learn a comparison geometry that
supports the confirmed Chevron gate better than temporal instance consistency?

Experiment 006 learned persistence and noise invariance but not enough
consequence-relevant structure. Experiment 007 adds dynamics while retaining a
compact synthetic task and the same downstream memory problem.

## Predictive world

A fixed latent dynamics system contains four actions. Each action applies a
different near-identity orthogonal transformation to a 12-dimensional latent
state, followed by transition noise 0.05. The transformations are generated
once from seed 607 and never trained.

The existing nonlinear sensor from Experiment 006 maps both current and next
latent states into distorted 12-dimensional observations. Training examples
contain only:

```text
current observation, selected action, next observation
```

Latent state, category, downstream correct action, reward, compatibility,
memory contents, and audit labels are unavailable to the learner. Training
actions are sampled uniformly and are unrelated to the delayed-context task's
correct actions.

## Encoder and forward model

The encoder is the same 812-parameter two-layer network used in Experiment 006.
A training-only predictor contains one 12 by 12 linear transformation and bias
per action. For a sampled action:

```text
h_t = normalize(Encoder(o_t))
h_next = normalize(Encoder(o_next))
prediction = normalize(Predictor[h_t, action])
```

The loss is InfoNCE over predicted and observed next embeddings:

```text
L_transition = CE(prediction h_next^T / 0.10, diagonal)
```

The forward predictor is discarded after pretraining. The frozen encoder alone
supplies A and N geometry to the downstream agent.

Training uses 500 steps, batch size 256, AdamW learning rate 0.003, weight decay
0.00001, current-observation noise 0.15, and transition noise 0.05.

## Downstream mechanism

The delayed-context task, nonlinear sensor, four-entry provisional buffer,
promotion rule, and geometric Chevron gate remain unchanged. The downstream
correct-action assignments are generated independently of the predictive
dynamics, preventing the encoder from learning the policy during pretraining.

## Conditions

1. Oracle geometric Chevron in latent space.
2. Raw-sensor geometric Chevron.
3. Temporal-contrastive geometric Chevron from Experiment 006's objective.
4. Action-predictive geometric Chevron.
5. Action-predictive content attention using the same learned representation.

## Development and confirmation

Development uses encoder seeds 0 and 1 and ten fresh paired lifetimes per seed.
Both learned encoders receive 500 updates and the same sensor observations,
optimiser budget, and downstream lifetimes.

Confirmation is triggered only if action-predictive geometric Chevron:

- reaches at least 0.95 final old accuracy;
- reaches at least 0.75 final novel and clean novel-probe accuracy;
- reaches at least 0.15 residual calibration and three promotions;
- has a paired return interval with lower bound above zero versus temporal
  contrastive Chevron;
- has paired return and novel-accuracy lower bounds above -0.05 versus the
  oracle;
- has a return lower bound above -0.05 versus content attention using the same
  predictive representation;
- reaches encoded/latent cosine correlation of at least 0.80; and
- gives the true next embedding a mean cosine at least 0.30 above a permuted
  next embedding.

If triggered, confirmation trains untouched encoder seeds 800 through 809 and
evaluates each on twenty fresh lifetimes. No development parameter then changes.

## Interpretation

Passing would justify a compact spatial trap/shortcut environment: the agent
would have demonstrated a complete small-scale path from action-conditioned
experience to learned geometry, assented retrieval, provisional consolidation,
and protected retained memory.

Failure would show that merely predicting invertible transitions still does not
identify the comparison geometry needed for behavioural compatibility. The
next step would require explicit prediction of affordances or longer-horizon
consequences rather than another encoder or gate calibration sweep.

## Development outcome

The action-conditioned model learned its prediction task: true predicted-next
cosine exceeded permuted-next cosine by 0.643. However, encoded transition
cosine correlated only 0.670 with latent transition cosine, below the 0.784
latent-cosine correlation previously achieved by temporal contrastive learning.

Downstream action-predictive Chevron reached 0.700 return, 0.939 final old
accuracy, 0.632 final novel accuracy, 0.738 clean novel-probe accuracy, 0.123
residual calibration, and 3.0 promotions. Temporal-contrastive Chevron on the
same lifetimes reached 0.738 return and 0.753 novel accuracy. The
action-predictive condition failed every primary performance and causal
improvement criterion except promotion count and non-inferiority to content
attention. Confirmation was not run.

One-step transition prediction can succeed in an arbitrary learned coordinate
system. It does not require cosine distance to express behavioural
compatibility. A future representation objective would need to constrain
relations between states according to their reward and transition
distributions—a bisimulation-like or affordance-predictive objective—rather
than merely reconstructing the next embedding.
