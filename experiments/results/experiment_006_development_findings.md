# Experiment 006 development findings

## Outcome

Temporal contrastive learning recovered a substantial amount of the hidden
comparison geometry without labels, but it did not recover enough to trigger
confirmation or a move to a spatial environment.

## Representation result

The fixed nonlinear sensor reduced the correlation between observed cosine
similarity and latent cosine similarity to 0.453. After 500 symmetric InfoNCE
updates, the two encoder seeds reached a mean correlation of 0.784.

Adjacent views had mean cosine 0.760, compared with 0.008 for permuted views, a
gap of 0.752. The contrastive loss fell from approximately 4.97 to 2.05. These
results demonstrate genuine label-free representation learning rather than a
random projection effect.

## Downstream result

| Condition | Return | Final old | Final new | New probe | q calibration | Promotions |
|---|---:|---:|---:|---:|---:|---:|
| Oracle geometric Chevron | 0.819 | 0.972 | 0.890 | 0.963 | 0.189 | 3.85 |
| Raw-sensor geometric Chevron | 0.346 | 0.685 | 0.327 | 0.388 | -0.004 | 0.50 |
| Random-encoder geometric Chevron | -0.045 | 0.423 | 0.164 | 0.150 | -0.001 | 0.00 |
| Temporal geometric Chevron | 0.732 | 0.937 | 0.727 | 0.887 | 0.135 | 3.65 |
| Temporal content attention | 0.732 | 0.942 | 0.702 | 0.775 | 0.385 | 3.00 |

The temporal encoder produced a large improvement over both raw and random
representations. It also gave Chevron higher novel accuracy and novel-probe
accuracy than content attention using the same embedding, while their mean
returns were equal.

Nevertheless, temporal Chevron missed the predeclared old accuracy, novel
accuracy, residual calibration, and oracle-noninferiority criteria. The misses
were modest but simultaneous, so untouched confirmation seeds were not used.

## Interpretation

Temporal pairing teaches persistence and noise invariance. It does not directly
teach which distinctions predict different transitions, affordances, or action
consequences. Two observations can be stable instances while still being embedded
poorly for deciding whether one retained policy applies to the other.

The result supports the overall path but not the current objective. The next
representation learner should predict what happens next, preferably conditioned
on action, so the comparison space reflects behavioural consequences rather
than instance identity alone.

Experiment 007 subsequently tested one-step action-conditioned prediction. It
learned accurate next-embedding predictions but produced worse comparison
geometry and downstream performance than temporal contrastive learning. This
shows that prediction alone is insufficient: the representation objective must
explicitly make distance correspond to compatible reward and transition
distributions.
