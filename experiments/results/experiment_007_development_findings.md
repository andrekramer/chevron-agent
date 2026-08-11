# Experiment 007 development findings

## Outcome

One-step action-conditioned prediction did not learn a better Chevron
comparison geometry than temporal instance consistency. Confirmation was not
run.

## Prediction result

The action-conditioned InfoNCE loss fell from approximately 5.30 to 2.95. Mean
cosine between the predicted and true next embedding was 0.666, compared with
0.023 for a permuted next embedding. The 0.643 gap comfortably passed the
predeclared prediction criterion.

Despite this, encoded transition-cosine correlation with latent transition
cosine was only 0.670. Temporal contrastive learning had previously recovered a
broader latent-cosine correlation of 0.784. The forward model learned to predict
within its own coordinate system without recovering the metric needed by
Chevron assent.

## Downstream comparison

| Condition | Return | Final old | Final new | New probe | q calibration | Promotions |
|---|---:|---:|---:|---:|---:|---:|
| Oracle geometric Chevron | 0.828 | 0.964 | 0.908 | 0.988 | 0.194 | 3.95 |
| Raw-sensor geometric Chevron | 0.350 | 0.639 | 0.363 | 0.438 | 0.037 | 0.20 |
| Temporal-contrastive Chevron | 0.738 | 0.936 | 0.753 | 0.850 | 0.149 | 3.55 |
| Action-predictive Chevron | 0.700 | 0.939 | 0.632 | 0.738 | 0.123 | 3.00 |
| Action-predictive content attention | 0.683 | 0.919 | 0.613 | 0.713 | 0.321 | 2.70 |

Action-predictive Chevron remained much better than raw sensor geometry and
slightly exceeded content attention using the same representation. But it was
worse than the temporal encoder on return, novel acquisition, novel probes, q
calibration, and promotions. It failed the predeclared causal-improvement,
absolute-performance, correlation, and oracle-distance checks.

## Interpretation

Predicting the next observation is not the same as learning behavioural
equivalence. An encoder and forward model can jointly choose any coordinate
system in which transitions are predictable. Cosine distance in that system
need not answer Chevron's question: should the retained policy or category
represented by this N slot be trusted for the present evidence?

The next plausible objective is relational. A bisimulation-like representation
would make two states close when their reward consequences and action-conditioned
transition distributions are close, and far when either differs. An
affordance-predictive alternative would compare the vector of possible outcomes
under actions rather than reconstructing a raw next state.

Either is a genuinely new hypothesis. Increasing forward-model steps, width,
or gate tuning would not address the identified ambiguity and is not
recommended.
