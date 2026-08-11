# Experiment 004 development findings

## Outcome

Experiment 004 is a useful negative development result. It establishes the
first reward-only Chevron Agent loop, but this configuration did not earn a
confirmation run or a move to a spatial game.

The task contained eight established contexts and four nearby contexts that
appeared later. Each family required three distinct actions, so an agent could
not solve a novel context by merely blending the two old action memories.
Outcomes arrived three decisions after action selection. The learning code saw
only observations, chosen actions, and scalar rewards—not context identities,
correct actions, compatibility labels, or target memory slots.

## Main development result

Across two training seeds and 20 fresh evaluation lifetimes per condition:

| Condition | Return/decision | Final old accuracy | Final new accuracy | q calibration | Promotions |
|---|---:|---:|---:|---:|---:|
| Content attention + buffer | 0.734 | 0.976 | 0.615 | 0.323 | 2.60 |
| Direct MLP + buffer | -0.261 | 0.451 | 0.042 | -0.005 | 0.05 |
| Chevron + buffer | 0.541 | 0.841 | 0.371 | 0.035 | 0.85 |
| Chevron immediate write | 0.540 | 0.878 | 0.322 | 0.040 | 0.00 |
| Chevron coupled write | 0.525 | 0.822 | 0.377 | 0.038 | 0.85 |

The fixed content-attention controller was the strongest system. Full Chevron
substantially beat the direct MLP, but that control plainly underfit, so this is
not evidence for general Chevron superiority. More importantly, Chevron lost
to the strong fixed controller by 0.193 return per decision, with an
approximate paired 95% interval from -0.244 to -0.142.

Chevron's residual mass was only weakly calibrated. Mean q differed by 0.035
between unresolved and resolved cases, compared with 0.323 for the content
controller. Accordingly, the Chevron buffer promoted fewer than one of the
four novel contexts on average.

## Capacity diagnostic

The original protocol deliberately used a two-entry provisional buffer. After
the failure, training seed 0 was rerun with four entries to test whether the
buffer was simply too small. This was an exploratory diagnostic over ten fresh
lifetimes, not a confirmation run.

For the content controller, increasing capacity from two to four raised final
novel accuracy from 0.569 to 0.848, raised promotions from 2.3 to 3.8, and cut
evictions from 34.0 to 10.1. For Chevron, final novel accuracy moved only from
0.392 to 0.401, promotions remained 0.9, and q calibration remained weak
(0.038). The small buffer constrained the control, but it was not the main
cause of Chevron's failure to acquire novel memories.

Raw diagnostic evidence is retained in
`experiment_004_capacity4_diagnostic_results.json` with its generated report.

## What worked

- The provisional path enforced the protection invariant: there were no
  permanent writes before delayed evidence in the buffered conditions.
- The immediate-write ablation did make premature writes (1.1% in the main
  development run).
- Separate write assent created a positive read/write margin and reduced N
  drift relative to coupled gating (0.129 versus 0.144).
- Separate write assent improved return over coupled gating by 0.0158 in
  development; its approximate paired 95% interval was +0.0073 to +0.0244.
- Promoted Chevron candidates were usually valid when promotion occurred.

These observations support the implementation of read/write separation and
delayed eligibility. They remain exploratory because the untouched
confirmation seeds were not used.

## What did not work

- Delayed policy-gradient reward did not teach the Chevron assent gate a
  sharply calibrated unresolved signal.
- The full buffer did not improve total return over immediate writing.
- The learned Chevron agent did not acquire all four novel mappings while
  retaining the eight established mappings.
- The parameter-matched direct MLP was too weak to serve as the strongest
  conventional baseline.

## Conclusion and next experiment

The result narrows the problem. Storage capacity and safe write mechanics are
not sufficient; the agent needs a better retrospective learning signal for the
meaning of assent. The next small-compute experiment should keep the same task
and memory rules but train assent from reward prediction error or a short
eligibility-return target, while still withholding latent category and
compatibility labels. It should also add a stronger bilinear or small-attention
comparator with the same information and compute budget.

Only if that experiment produces calibrated q, learns nearly all four novel
contexts, and retains the old contexts on untouched seeds should the project
move to a spatial trap/shortcut environment. That is a refinement of the path
toward Chevron Agent, not a change of direction.
