# Experiment 003: frozen confirmation protocol

## Questions

1. Does Chevron retain an advantage when a conventional model receives exactly the same A evidence, N content, retrieval prior, and outcome labels?
2. When compatibility is revealed late, does keeping candidates outside N protect established categories?
3. Is a one-entry buffer sufficient, or does a two-entry buffer materially improve acquisition?

## Development and freeze

Seeds 0 and 1 were used only for development. Two issues were corrected before confirmation:

- The direct classifier now uses its native argmax over memory slots plus its null class. Applying Chevron's absolute residual threshold to a multiclass probability was not calibrated fairly.
- Pretraining was increased from 400 to 2,000 steps. At 400 steps the direct MLP was visibly underfit; at 2,000 it reached 92–93% sequential accuracy on the development streams.

No confirmation seed has been inspected. The architecture, thresholds, stream, outcome delay, buffer capacities, training duration, and metrics below are frozen before running seeds 200–209.

## Part A: equal-information diagnostic

- Models: projected-cosine Chevron assent and a conventional direct slot-or-null MLP
- Parameters: exactly 314 in each learned model
- Inputs: identical diagnostic A evidence, retained N contents, and retrieval prior
- Training: 2,000 supervised steps per seed, batch size 256
- Evaluation: 4,096 fresh memories and queries at each of five noise levels
- Decisions: Chevron uses q > 0.80; the direct classifier uses slots-plus-null argmax
- Reference: standard attention without abstention

The direct MLP deliberately has no separately observable retrieval, assent, and residual computations. If it matches Chevron, the static result supports the information and abstention requirement but not a unique Chevron comparator. If Chevron retains an advantage, it is evidence for the projected comparison inductive bias at a fixed parameter budget, not proof of universal superiority.

## Part B: delayed sequential consolidation

- Stream length: 600 observations
- Shift: one new category per address family appears after step 200
- New-category similarity: cosine 0.55 to an established family member
- Retrospective outcome delay: three steps
- Established N capacity: 12 slots, beginning with eight categories
- Online supervision: both learned models update only when the delayed outcome arrives

Compared systems:

- standard attention;
- direct MLP with a separate capacity-2 buffer;
- Chevron with a separate capacity-1 buffer;
- Chevron with a separate capacity-2 buffer;
- Chevron with unresolved candidates written into N immediately.

Buffered candidates cannot affect N before their outcomes. The interposed control uses the same learned Chevron gate but lets candidates participate in N immediately; when full, it replaces the least recently used slot.

## Primary metrics and decision

- overall, pre-shift, and post-shift decision accuracy;
- established-category retention;
- revealed-novel-category accuracy;
- candidate evictions and successful promotions;
- premature writes and established-slot overwrites;
- final clean probes of the eight original and four new categories.

The result supports moving to RL if at least one learned same-information model can acquire all four new categories while retaining all eight initial categories, and if separate buffering reduces premature N writes or retention loss relative to interposition across fresh seeds. This remains a supervised sequential diagnostic rather than an RL or agency result.
