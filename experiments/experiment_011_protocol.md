# Experiment 011: frozen persistence-derived identity protocol

## Question

Can the identity comparison used by Chevron be learned from temporal
persistence, while the already confirmed retrospective-policy mechanism is
left unchanged?

Experiment 006 learned a general cosine geometry from pairs of noisy views. It
improved the hidden geometry substantially, but did not meet its downstream
gate. Experiment 011 asks a narrower question. The identity gate does not need
to reconstruct every latent similarity. It needs to admit repeated views of
the same continuing identity and reject a confusable change of identity.

Experiment 010a supplies the fixed downstream mechanism: delayed reward,
protected retrospective policy revision, a shared typed provisional bank, and
promotion-time identity revalidation. No policy rule is relearned or tuned in
this experiment.

## Task

The RL lifetime, action space, delayed reward process, reward noise, identity
gate, provisional bank, and consolidation rules are exactly those of
Experiment 010a:

- four address families;
- eight established and four later novel identities;
- four actions;
- 800 decisions with the task shift at step 200 and retention phase at 600;
- reward delayed by three decisions and reversed independently on 10% of
  events;
- two unexpected incumbent failures before policy search;
- two positive alternative outcomes before policy revision;
- two positive outcomes before new-identity promotion; and
- promotion-time revalidation before permanent identity allocation.

The identity vector is no longer observed directly. A frozen nonlinear sensor
maps every latent identity observation into a normalised 12-dimensional sensor
vector. The agent sees only this sensor vector. Latent identities and context
labels remain audit-only information.

## Learned identity representation

The trainable encoder is deliberately small: a 12-to-48-to-12 residual MLP
whose output is normalised. Its final residual layer begins at zero, so the
untrained encoder begins as the raw sensor geometry rather than an arbitrary
random remapping.

Pretraining uses new synthetic identities on every optimisation step. It never
uses the RL task's identities, actions, rewards, category labels, or future
correct policies. Each training example is identified only by membership in a
short temporal persistence window.

The full persistence condition samples 64 windows with four noisy observations
per window. Windows are arranged in confusable pairs whose latent cosine is
0.55, close to the downstream novel-identity construction. Observations from
one persistence window are positives; observations from every other window,
including the confusable paired window, are negatives. The multi-positive
contrastive loss for observation `i` is:

```text
L_i = -log(
    sum(exp(sim(i, p) / temperature) for p in same persistence window)
    /
    sum(exp(sim(i, a) / temperature) for all a other than i)
)
```

This is temporal grouping, not identity supervision: the learner is told which
observations persisted together, but is never given a semantic identity label.
Because every optimisation batch contains newly sampled identities, it cannot
solve pretraining by memorising a category table.

All learned-representation conditions use 750 AdamW steps, 256 sensor
observations per step, temperature 0.10, view noise 0.15, learning rate 0.003,
and weight decay 0.00001. Gradient norm is clipped at 1.0.

## Fixed downstream identity gate

The downstream gate is inherited without calibration or threshold search:

```text
alpha_j = uniform retrieval over slots in the observed address family
r_id_j = sigmoid(40 * (cosine_j - 0.62))
w_id_j = alpha_j * r_id_j
q_id = 1 - sum_j w_id_j
```

Candidate routing, provisional capacity, promotion support, identity
revalidation, and all policy thresholds are also frozen at the Experiment 010a
values. A representation either works with the established gate or it does
not.

## Conditions

1. **Oracle protected Chevron**: latent identity geometry and the complete
   protected/revalidated Experiment 010a mechanism. This is the downstream
   ceiling, not a deployable condition.
2. **Raw-sensor protected Chevron**: frozen nonlinear sensor with no learned
   correction.
3. **Pairwise-temporal protected Chevron**: the residual encoder is trained on
   two noisy views of 128 independent identities per step using symmetric
   pairwise InfoNCE. This is a compute-matched version of the Experiment 006
   objective.
4. **Multi-view-temporal protected Chevron**: 64 independent persistence
   windows with four views each and the multi-positive loss, but no deliberate
   confusable neighbour.
5. **Hard-persistence protected Chevron**: the full proposed identity learner,
   with four-view persistence windows and paired confusable changes.
6. **Hard-persistence direct adaptation**: the same learned identity
   representation, sensor stream, buffer, and revalidation, but the established
   policy uses direct value adaptation. This checks whether learning identity
   forces abandonment of the protected policy path.

All six conditions receive paired RL lifetimes. Conditions three through six
receive encoders trained from the same training seed, with identical model
size, optimiser budget, and observation count.

## Representation diagnostics

On unseen identities, the frozen 0.62 boundary is evaluated directly:

- same-identity admission: cosine between a clean identity template and a
  noisy observation of that identity is at least 0.62;
- confusable-change rejection: cosine between a clean identity template and a
  noisy observation of a different identity constructed at latent cosine 0.55
  is below 0.62; and
- balanced identity-decision accuracy: the mean of the two rates.

Latent-to-encoded cosine correlation and temporal positive/negative gaps are
reported as secondary diagnostics. They are not success criteria because the
experiment tests an identity decision, not reconstruction of the complete
latent metric.

## Development seeds

Development trains three encoder objectives for each of training seeds 0 and
1. Every trained encoder is evaluated on ten paired RL lifetimes, for twenty
lifetimes per condition. RL seeds are `110,000,000-110,000,009` for training
seed 0 and `110,001,000-110,001,009` for training seed 1.

No architecture, threshold, training schedule, comparison, or criterion may be
changed after inspecting development results.

## Frozen development gate

Fresh-seed confirmation is triggered only if hard-persistence protected
Chevron:

- achieves at least 0.90 phase-three stable retention;
- reaches at least 0.75 final reversed and novel policy-probe accuracy;
- promotes at least three of four novel identities and three of four policy
  revisions on average;
- has identity residual calibration of at least 0.10 and policy residual
  calibration of at least 0.10;
- makes zero duplicate identity allocations, established-memory overwrites,
  and under-supported permanent writes;
- has unseen same-identity admission of at least 0.90, confusable-change
  rejection of at least 0.80, and balanced identity-decision accuracy of at
  least 0.85;
- improves realised return over raw-sensor protected Chevron with a paired 95%
  lower bound above zero;
- improves realised return over pairwise-temporal protected Chevron with a
  paired 95% lower bound above zero;
- is non-inferior to multi-view-temporal protected Chevron in realised return
  with a paired 95% lower bound above -0.02;
- is non-inferior to oracle protected Chevron in realised return and clean
  accuracy with paired 95% lower bounds above -0.08, and in stable retention
  with a lower bound above -0.05; and
- is non-inferior to hard-persistence direct adaptation in realised return and
  clean accuracy with paired 95% lower bounds above -0.08, and in stable
  retention with a lower bound above -0.03.

If every criterion passes, confirmation will train ten untouched encoder seeds
`1100-1109`. Each will be evaluated on twenty paired RL lifetimes, using seeds
`111,000,000-111,000,019` through
`111,009,000-111,009,019`. The complete protocol will remain frozen.

## Interpretation

Passing would show that Chevron's identity assent can be learned from a local
temporal persistence signal while retaining delayed-outcome policy revision and
memory protection. It would justify the next step into a small visual RL task.
It would not show that the representation is generally optimal, biologically
faithful, or superior to standard attention.

Failure would identify one of three boundaries: the representation does not
support the fixed identity decision, the hard-persistence curriculum adds
nothing beyond simpler temporal pairing, or learned identity destabilises the
downstream memory mechanism. The failed criterion, rather than a post-hoc
threshold sweep, will determine the next experiment.

## Development outcome

The frozen development gate failed, so confirmation was not run.

All three temporal learners repaired the sensor's identity boundary. The raw
sensor rejected only 0.161 of confusable identity changes, while pairwise,
multi-view, and hard-persistence learning rejected 0.824, 0.829, and 0.824.
Their balanced identity-decision accuracies were 0.905, 0.907, and 0.904.

Hard-persistence protected Chevron improved return over the raw sensor by
+0.139, approximate paired 95% interval +0.106 to +0.172, and raised novel
probe accuracy from 0.088 to 0.738. It retained 0.922 stable accuracy and made
no duplicate allocation, established overwrite, or under-supported write.

However, the full proposal trailed the simpler pairwise encoder by 0.021 return
and 0.125 novel-probe accuracy. It missed the 0.75 novel-probe floor, the
required improvement over pairwise temporal learning, multi-view
non-inferiority, and oracle return non-inferiority. The extra persistence-window
and hard-negative machinery is therefore not supported.

The strongest development condition was pairwise temporal learning: 0.491
return, 0.928 retention, and 0.863 novel-probe accuracy. Because it was a
control rather than the predeclared confirmation target, it requires a new
narrow fresh-seed confirmation rather than inheriting confirmation from this
failed experiment.
