# Experiment 004: frozen protocol for reward-derived memory

## Question

Can a small acting agent use delayed scalar reward to preserve eight
established context-action memories while acquiring four nearby contexts, and
do Chevron's provisional buffer and stricter write permission improve that
stability-plasticity trade-off?

## Scope

This is the first reinforcement-learning bridge, implemented as a delayed
contextual bandit. The agent chooses actions and receives reward; it receives
no compatibility label, target slot, latent category ID, or correct action.

It is not yet a spatial game, PPO result, or complete Chevron Agent. The task
isolates reward-derived eligibility before navigation and long-horizon control
are added.

## Lifetime

- Four address families.
- Two established contexts per family, for eight initial N slots.
- One nearby novel context per family, introduced after step 200.
- 600 decisions per lifetime.
- Novel-context probability after the shift: 0.35.
- Four actions with deterministic reward: +1 correct, -1 incorrect.
- Reward delay: three decisions.
- Permanent capacity: twelve slots.
- Provisional capacity: two entries.

Each family's two established contexts and its novel context require three
distinct actions, sampled from a per-family permutation. The retrieved old
pair therefore cannot already express the novel action. Initial N contains clean
templates and correct action values for the eight established contexts. This
represents retained prior knowledge; only the four later contexts require
within-lifetime acquisition.

The current diagnostic evidence and retained N templates share a normalised
comparison space in this first RL bridge. Experiment 002 has already shown
that independently projected A/N comparison spaces can be learned under direct
supervision. Learning an A-to-N promotion transform from reward is deferred.

## Learning signal

The policy samples an action from the admitted memory read. REINFORCE updates
the learned comparator after delayed eligibility. Positive reward confirms
the chosen action; negative reward rules out only that action and does not
reveal the correct alternative. No category, correct-action, or compatibility
loss is used.

For online memory, only a positively rewarded eligible action can confirm a
candidate action value. Permanent memory may update only when that delayed
outcome becomes eligible, except in the declared immediate-write ablation.

Model parameters are trained across independent lifetimes. During confirmation
evaluation they are frozen; only declared A/N and provisional state may change
within a lifetime.

## Conditions

1. Content attention + buffer: conventional cosine retrieval over diagnostic
   content with a similarity-based null/allocation test.
2. Direct MLP + buffer: learned slot-or-null comparator with the same A/N
   inputs and parameter count as Chevron.
3. Chevron + buffer: address retrieval, learned A/N assent, residual q,
   capacity-two provisional storage, and stricter write assent.
4. Chevron immediate write: unresolved candidates enter N before reward.
5. Chevron coupled write: write permission uses the read gate rather than a
   positive threshold margin.

The direct MLP and Chevron comparator each have 314 trainable parameters. All
conditions receive the same observations, initial memories, capacity, reward
stream, and four-action inference rule.

## Metrics

Primary:

- lifetime return per decision;
- final-200 established-context accuracy;
- final-200 novel-context accuracy;
- final clean probes of all eight established and four novel contexts.

Mechanism:

- q on unresolved versus resolved novel contexts;
- promotion count and promotion precision;
- premature permanent writes;
- established-slot overwrites;
- established N-template drift;
- false candidates on established contexts;
- buffer evictions;
- read/write gate separation;
- final category coverage.

The diagnostic retention-plasticity score is reported but does not replace its
components:

    S = R_old + P_new - 0.5 O_est - 0.5 W_prem

## Development and confirmation

Development seeds 0 and 1 may be used to establish that policy-gradient
learning is numerically stable and that neither learned comparator is plainly
underfit. Learning duration, noise, thresholds, and update rates will then be
frozen and recorded here before confirmation.

Confirmation will use ten untouched training seeds beginning at 300. Each
trained seed will be evaluated on twenty fresh lifetimes. Paired differences
and every individual seed will be retained.

## Decision rule

Proceed toward a spatial game if:

1. at least one learned same-information system retains the established
   memories and acquires all four novel mappings from delayed reward;
2. the full Chevron condition produces calibrated unresolved mass;
3. separate buffering or stricter write permission improves protection over
   its corresponding Chevron ablation across fresh seeds; and
4. comparison with the strongest conventional condition is reported even if
   Chevron does not win.

Failure to outperform the conventional comparators does not invalidate the
mechanism. It means this task does not require the Chevron factorisation.

## Development outcome

Development was frozen after seeds 0 and 1, with 60 training lifetimes and ten
fresh evaluation lifetimes per seed. The full Chevron condition did not meet
the decision rule, so the untouched confirmation seeds were not run.

The strongest condition was the hand-specified content-attention controller.
It reached 0.734 return per decision, 0.976 final old-context accuracy, and
0.615 final novel-context accuracy. Chevron with a capacity-two buffer reached
0.541, 0.841, and 0.371 respectively. Its residual calibration was only 0.035:
unresolved observations received only slightly more residual mass than resolved
ones. It promoted 0.85 of four novel contexts on average.

A declared post-development capacity diagnostic reran training seed 0 with a
four-entry buffer. The content controller improved from 0.569 to 0.848 final
novel accuracy and from 2.3 to 3.8 promotions. Chevron changed only from 0.392
to 0.401 final novel accuracy and remained at 0.9 promotions; residual
calibration remained weak (0.038). The original capacity-two comparison was
therefore somewhat buffer-limited, but capacity does not explain the learned
Chevron result.

Separate write assent did show a small development advantage over coupled
read/write gating: +0.0158 return per decision, with an approximate paired 95%
interval of +0.0073 to +0.0244. This is an exploratory result, not a confirmed
claim. The buffer did not improve overall return over immediate writing.

The development evidence supports the safety mechanics—the buffer prevents
premature writes and the write margin is measurable—but not the stronger claim
that policy-gradient reward alone learns a useful vigilance signal in this
configuration. The next controlled experiment should improve the retrospective
credit signal for assent before moving to a spatial game.
