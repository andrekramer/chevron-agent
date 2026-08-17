# Experiment 011a confirmation findings

## Result

Pairwise temporal identity learning passed all twenty-two frozen confirmation
criteria across ten untouched encoder seeds and 200 untouched paired RL
lifetimes.

The learned encoder saw only pairs of noisy observations that persisted
together. It received no RL category, action, reward, policy, or correctness
label. At the unchanged cosine boundary of 0.62 it achieved:

- 0.988 same-identity admission;
- 0.817 rejection of confusable identity changes;
- 0.902 balanced identity-decision accuracy; and
- 0.794 correlation with the hidden latent cosine geometry.

## Downstream result

| Condition | Return | Retention | Reversed probe | Novel probe | False stable revisions |
|---|---:|---:|---:|---:|---:|
| Oracle protected | 0.546 | 0.948 | 0.986 | 0.922 | 0.090 |
| Raw-sensor protected | 0.360 | 0.905 | 0.949 | 0.146 | 3.060 |
| Pairwise-temporal protected | 0.481 | 0.926 | 0.931 | 0.771 | 0.195 |
| Pairwise-temporal direct | 0.539 | 0.928 | 0.973 | 0.667 | 0.000 |

Against the distorted raw sensor, learned-identity Chevron improved realised
return by +0.121, with an approximate paired 95% interval from +0.110 to
+0.132. Novel-probe accuracy improved by +0.625, interval +0.585 to +0.665.
The learned system also improved retention by +0.020.

Pairwise-temporal protected Chevron promoted 3.405 of four novel identities and
recovered 3.745 of four reversed policy identities on average. Identity
residual calibration was 0.223 and policy residual calibration was 0.573. It
made zero duplicate identity allocations, zero established-memory overwrites,
and zero under-supported permanent writes across the complete confirmation.

## The cost of protection

The protected retrospective policy path returned 0.481 versus 0.539 for direct
value adaptation, a paired difference of -0.058. Its confidence interval,
-0.065 to -0.051, remained inside the frozen -0.08 non-inferiority margin.
Stable retention differed by only -0.002, interval -0.006 to +0.002.

Protection bought better consolidated novel memory. Its final novel probe was
0.771 versus 0.667 for direct adaptation, a paired improvement of +0.104 with
interval +0.068 to +0.139. It was slower on reversed policies, as expected from
requiring repeated evidence before revision.

The learned system also remained within every frozen oracle margin. Its return
gap to supplied latent identity was -0.065, interval -0.074 to -0.055, and its
retention gap was -0.022, interval -0.025 to -0.019.

## What is now confirmed

On this compact delayed-context task, Chevron no longer requires a supplied
identity comparison. A small residual encoder trained from temporal pairs can
recover enough identity structure for the fixed assent gate to distinguish
repeat experience from confusable novelty. The learned representation works
together with retrospective policy evidence, typed provisional storage, and
promotion-time revalidation without violating the tested memory-protection
invariants.

Experiment 011 remains a failed development test of the more elaborate
hard-persistence curriculum. Experiment 011a confirms the separately frozen,
simpler pairwise learner; it does not rename or retroactively pass Experiment
011.

## Limits

The task remains synthetic. Broad address families are supplied, the nonlinear
sensor is fixed, temporal pairs are generated cleanly during pretraining, and
the action space is small. Standard attention, recurrent networks, and generic
buffer controls have not yet been compared on a visual or spatial task. The
result does not establish a general agent, a stable personality, or a
coherently evolving self.

## Next step

The next round should move the confirmed mechanism into a small partially
observable visual or grid-world task. Temporal adjacency can train identity,
while delayed outcomes continue to drive policy suspicion and revision. The
task should make the internal result behaviourally visible through retained
preferences, trap avoidance, adaptation to a changed route, and cautious
treatment of a genuinely novel place.

That round should compare at least learned Chevron, direct adaptation with the
same encoder and buffer, and a standard recurrent or attention memory with a
matched parameter budget.
