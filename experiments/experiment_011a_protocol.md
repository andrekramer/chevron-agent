# Experiment 011a: frozen pairwise-identity confirmation

## Purpose

Experiment 011 tested three ways to learn identity geometry from temporal
persistence. Its proposed hard-persistence condition failed the frozen
development gate and is not being rescued or renamed. The simplest control,
pairwise temporal contrastive learning, was the strongest development
condition: it reached 0.491 return, 0.928 retention, and 0.863 novel-probe
accuracy while making no duplicate identity, established overwrite, or
under-supported write.

Experiment 011a is a new, narrow fresh-seed confirmation of that simpler
learner. It freezes the pairwise encoder exactly as it ran in Experiment 011.

## Frozen representation learner

The observation is produced by the same deterministic nonlinear sensor with
seed 606. The encoder is the same normalised residual MLP:

```text
encoded(x) = normalise(x + MLP_12_to_48_to_12(x))
```

The final residual layer begins at zero. Training uses:

- 750 AdamW steps;
- 256 sensor observations per step, arranged as 128 temporal pairs;
- two independently noisy views of a newly sampled latent identity;
- symmetric pairwise InfoNCE with temperature 0.10;
- latent view noise 0.15;
- learning rate 0.003 and weight decay 0.00001; and
- gradient-norm clipping at 1.0.

New latent identities are sampled on every optimisation step. The learner sees
only the pairing of two observations from one persistence event. It does not
receive RL identities, categories, actions, rewards, policies, or correctness
signals.

There is no multi-view loss, hard-negative curriculum, threshold calibration,
or hyperparameter search.

## Frozen downstream agent and task

The complete Experiment 010a mechanism remains unchanged:

- four address families, eight established and four later novel identities;
- four actions and an 800-decision lifetime;
- shift at step 200 and retention phase at step 600;
- reward delayed by three decisions and reversed on 10% of events;
- identity assent threshold 0.62 and sigmoid slope 40;
- identity residual trigger 0.80 and admitted-mass threshold 0.25;
- one shared capacity-eight typed provisional bank;
- two positive outcomes before new-identity promotion;
- two unexpected incumbent failures before policy search;
- two positive alternative outcomes before policy revision; and
- promotion-time identity revalidation before permanent allocation.

The supplied latent identity is used only by the oracle control and the audit.

## Conditions

1. **Oracle protected Chevron**: latent identity geometry with protected
   retrospective policy revision and promotion-time revalidation.
2. **Raw-sensor protected Chevron**: the nonlinear sensor output without a
   learned encoder.
3. **Pairwise-temporal protected Chevron**: the frozen pairwise encoder with
   the protected retrospective policy mechanism.
4. **Pairwise-temporal direct adaptation**: the same learned representation,
   sensor stream, provisional bank, and identity revalidation, but established
   policies use direct value adaptation.

All conditions receive paired RL lifetimes and condition-local action-randomness
streams derived from the lifetime seed.

## Untouched seeds

Ten encoders are trained with seeds `1200-1209`. Each encoder is evaluated on
twenty paired lifetimes. RL seeds are:

```text
112,000,000-112,000,019
112,001,000-112,001,019
...
112,009,000-112,009,019
```

This produces 200 paired lifetimes per condition. None of these encoder or RL
seeds were used in Experiment 011 development. No result from them may change
the mechanism, thresholds, training schedule, comparisons, or criteria below.

## Frozen confirmation criteria

Pairwise-temporal protected Chevron is confirmed only if it:

- achieves at least 0.90 phase-three stable retention;
- reaches at least 0.75 final reversed and novel policy-probe accuracy;
- promotes at least three of four novel identities and recovers at least three
  of four reversed policy identities on average;
- has identity residual calibration of at least 0.10 and policy residual
  calibration of at least 0.10;
- averages no more than 0.25 false stable-context policy revisions;
- makes zero duplicate identity allocations, established-memory overwrites,
  and under-supported permanent writes;
- achieves unseen same-identity admission of at least 0.90, confusable-change
  rejection of at least 0.80, and balanced identity-decision accuracy of at
  least 0.85 at the unchanged 0.62 boundary;
- improves realised return and novel-probe accuracy over raw-sensor protected
  Chevron with paired 95% lower bounds above zero;
- is non-inferior to oracle protected Chevron in realised return and clean
  accuracy with paired 95% lower bounds above -0.08, and in stable retention
  with a lower bound above -0.05; and
- is non-inferior to pairwise-temporal direct adaptation in realised return and
  clean accuracy with paired 95% lower bounds above -0.08, and in stable
  retention with a lower bound above -0.03.

Every criterion must pass. There is no partial confirmation.

## Interpretation

Passing would confirm, on this synthetic delayed-context workload, that a small
identity encoder trained only from temporal pairs can supply a useful fixed
Chevron assent gate while preserving the already established memory-protection
invariants. It would support moving the same separation into a small visual RL
environment.

Failure would constrain the claim to Experiment 011 development evidence. It
would show that the apparent pairwise advantage was seed-sensitive, too far
from oracle identity, or insufficiently competitive with direct adaptation.
The confirmation result will be reported under this experiment number either
way.

## Confirmation outcome

All twenty-two frozen criteria passed across ten untouched encoder seeds and
200 untouched paired RL lifetimes.

The pairwise encoder achieved 0.988 same-identity admission, 0.817
confusable-change rejection, 0.902 balanced identity-decision accuracy, and
0.794 latent-cosine correlation.

Pairwise-temporal protected Chevron reached 0.481 return, 0.926 retention,
0.931 reversed-probe accuracy, and 0.771 novel-probe accuracy. Against the raw
sensor it improved return by +0.121, approximate paired 95% interval +0.110 to
+0.132, and novel probes by +0.625, interval +0.585 to +0.665.

It promoted 3.405 of four novel identities, recovered 3.745 of four reversed
policy identities, and made no duplicate identity allocation, established
overwrite, or under-supported write. Its return remained within the frozen
non-inferiority margins against both oracle identity and direct value
adaptation.

Direct adaptation retained a return advantage of 0.058, while protected
Chevron produced a 0.104 higher novel probe, interval +0.068 to +0.139. This is
the expected trade: slower protected updating in exchange for stronger
consolidated novel memory.
