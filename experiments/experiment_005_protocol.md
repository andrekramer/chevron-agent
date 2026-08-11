# Experiment 005: frozen retrospective-assent protocol

## Question

Can delayed scalar reward train a useful Chevron assent signal when reward is
applied retrospectively to the admitted memory support that predicted the
chosen action's outcome?

Experiment 004 supplied reward only through REINFORCE. It learned a competent
policy but did not calibrate residual mass or reliably promote novel contexts.
This experiment changes the credit signal, not the environment or the memory
protection rules.

## Fixed task

The delayed-context lifetime is unchanged from Experiment 004 except for the
already-declared capacity diagnostic:

- eight established contexts in four address families;
- four nearby novel contexts introduced after decision 200;
- four possible actions, with three distinct correct actions per family;
- 600 decisions and a reward delay of three;
- deterministic reward of +1 for a correct action and -1 otherwise;
- twelve permanent slots and a four-entry provisional buffer;
- promotion after two coherent positive outcomes.

The agent receives observation evidence, address family, its selected action,
and delayed scalar reward. Latent context identity, correct action,
compatibility, and target slot remain audit-only data.

## Retrospective consistency objective

At action time, let the non-negative support supplied by admitted memory for
the selected action be:

```text
s_t = sum_j w_tj * clamp(N_action[j, a_t], 0, 1)
```

When delayed reward arrives, define the observed success target:

```text
h_t = 1 if reward_t > 0 else 0
```

The additional loss is:

```text
L_consistency = BCE(clamp(s_t, epsilon, 1-epsilon), h_t)
```

The total training loss is the existing REINFORCE and entropy loss plus
`L_consistency` with frozen weight 1.0.

This objective does not reveal which unchosen action was correct. It reinforces
a read that supported a successful action and suppresses a read that supported
an unsuccessful action. When no slot supported an action, it supplies no
spurious slot-specific positive gradient; provisional promotion remains the
mechanism for creating a new action memory.

## Strong conventional comparator

The weak direct MLP from Experiment 004 is replaced by projected bilinear null
attention. It independently projects current evidence and retained content,
uses their normalised dot product as a slot score, and includes a learned null
logit in the same softmax.

This control receives the same information and retrospective objective but
conflates retrieval and admission. With 12-dimensional inputs and a
13-dimensional comparison space, it has 314 trainable parameters—the same as
the Chevron gate.

## Conditions

1. Content attention + buffer: the strongest fixed controller from Experiment
   004, now with buffer capacity four.
2. Bilinear null attention + retrospective loss + buffer.
3. Chevron + retrospective loss + buffer and separate write assent.
4. Chevron + policy loss only + buffer: direct test of the new objective.
5. Chevron + retrospective loss + immediate write: buffer ablation.
6. Chevron + retrospective loss + coupled read/write gate: write-protection
   ablation.

All learned conditions use the same observations, task instances, action RNG
streams, parameter budget, optimiser, and training duration.

## Development and confirmation

Development uses training seeds 0 and 1, 60 lifetimes per seed, and ten fresh
evaluation lifetimes per seed. The objective weight, thresholds, buffer size,
and optimiser are frozen before inspecting those aggregate results.

Confirmation is permitted only if development satisfies all of the following:

1. retrospective Chevron final old-context accuracy is at least 0.90;
2. final novel-context accuracy and clean novel probe accuracy are each at
   least 0.75, with at least three of four novel contexts promoted on average;
3. residual calibration is at least 0.15;
4. the approximate paired 95% interval for return versus policy-only Chevron
   has a lower bound above zero;
5. buffered Chevron makes no premature permanent writes and has a positive
   read/write margin.

If triggered, confirmation will train seeds 400 through 409 and evaluate each
on twenty fresh lifetimes. No development parameter may then change.

## Interpretation

Success would justify the first small spatial trap/shortcut environment. It
would not establish a general agent, a persistent self, or superiority to
standard attention.

Failure would indicate that explicit buffering and write protection remain
useful engineering constraints, but this reward-derived Chevron assent rule
does not yet justify further architectural scaling.

## Development outcome

The frozen development run used seeds 0 and 1, with 60 training lifetimes and
ten fresh evaluation lifetimes per seed. Only the two protection checks passed;
all performance, acquisition, calibration, and causal-improvement checks
failed. Confirmation seeds 400–409 were therefore not run.

Retrospective Chevron reached 0.512 return per decision, 0.806 final old
accuracy, 0.357 final novel accuracy, 0.438 clean novel-probe accuracy, 0.030
residual calibration, and 0.55 promotions. Policy-only Chevron reached 0.513,
0.805, 0.362, 0.412, 0.032, and 0.55 respectively. The paired return difference
was -0.0005, with an approximate 95% interval from -0.0155 to +0.0145. The new
objective therefore produced no detectable improvement.

The projected bilinear control was stronger than Experiment 004's direct MLP
but still weak on novelty: 0.458 return, 0.327 final novel accuracy, 0.067
residual calibration, and no promotions. Retrospective Chevron exceeded its
return by 0.0548, with an approximate paired interval from +0.0204 to +0.0893,
but neither learned system approached the fixed content controller.

The fixed controller reached 0.788 return, 0.982 final old accuracy, 0.821
final novel accuracy, 0.457 residual calibration, and 3.55 promotions. This
shows that the task and memory mechanics are solvable at the declared compute
budget. The remaining failure is learning an appropriately calibrated
comparison from sparse action reward.

The buffered conditions again made no premature writes, and separate write
assent retained a positive margin. Those are implementation properties, not
evidence that the learned Chevron gate improves agency. The project should not
advance this configuration to a spatial game.
