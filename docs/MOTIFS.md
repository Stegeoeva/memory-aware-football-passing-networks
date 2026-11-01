# Real two- and three-pass motifs

The motif analysis uses actual event-UUID links, not powers of an aggregate adjacency matrix. Two-pass motifs are ABA/ABC; three-pass motifs are ABAB, ABAC, ABCA, ABCB and ABCD. Canonical letters encode first appearance of distinct player IDs. There are **33,358 two-pass and 27,175 three-pass motifs**.

Each adjacent pair must exist in the audited real transition table. Failed passes, different handlers, confirmed opponent control, restarts or continuity boundaries are not skipped. Rolling motifs may overlap. A→B→C contains two passes; it must not be described as a three-pass sequence.

## Outcome landmark and selection

Later outcomes begin **after the final receipt**, excluding the passes required to define the motif from its subsequent success. Motif comparisons condition on completion of all required passes and do not measure the cost of unsuccessful formation attempts. Future indices, last-receipt IDs and source actions remain auditable.

R3 uses first-shot source xG, confirmed-loss zero or retained-endpoint training-only xT within three later ball actions. shot10, retention/shot and new box entry use the corresponding bounded-continuous-play horizon from final receipt. A literature-inspired shot15 sensitivity starts at the final pass start and flags own goals separately. The later [pure source-xG extension](MOTIF_XG.md) uses a different common provider-possession boundary; its percentages are not interchangeable with earlier shot10 values.

## Two context adjustments

**Initial receipt:** compare later continuation with typical outcomes for the initial receipt context. **Completion/geometry:** additionally account for final receipt location, pressure, role and the chain's duration, progression and spatial spread. Completion adjustment can remove part of the pathway through which a motif helps; it is not a total causal effect.

Separate q models are fit for two-/three-pass conditional cohorts, excluding pattern and player IDs. Five outer/three inner whole-match folds match the continuation design. The 20 training-only xT fits are reused with verified splits. Held-out rows do not train their encoders, q or xT.

Adjusted comparisons are differences in mean held-out residuals. All raw, initial-adjusted and completion-adjusted estimates remain in the full exports. Binary differences use percentage points and R3 uses proxy units per 100.

## Timing and uncertainty

Four policies are exported: all linked paths, ≤5 seconds after the previous recorded event, ≤5 seconds between pass starts, and no intermediate carry/dribble. The previous event can be pressure or receipt; these two timing definitions answer different questions. Regulation-only sensitivity is separate. Subsets retain full-cohort OOF models and are not refit per filter.

Use 2,000 shared match/whole-possession bootstrap draws. Overlapping motifs receive shared weights; paired contrasts are recomputed jointly. Intervals are pointwise and conditional on fitted models, not simultaneous tests. Groups below 50 known outcomes/five matches retain estimates without intervals. No individual-test family or FDR claim was made.

The source-xG extension demonstrates why shot frequency alone is incomplete: ABCD has higher point estimates of xG/motif than ABAB, but their interval includes zero, and linked-shot quality is close. Context and spatial completion matter; a motif shape alone is not evidence of tactical superiority.

## Reproduce and inspect

~~~shell
python scripts/run_motifs.py
python scripts/check_motifs.py
~~~

`--stage report` regenerates only presentation from numerical exports. Report (`../results/motifs/report.html`, generated locally) · effects (`../results/motifs/tables/pattern_effects.csv`, generated locally) · contrasts (`../results/motifs/tables/pattern_contrasts.csv`, generated locally) · dictionary (`../results/motifs/data_dictionary.csv`, generated locally) · validation (`../results/motifs/delivery_check.json`, generated locally).

[Håland et al. (2020)](https://doi.org/10.1515/jqas-2019-0097) studies three-pass football motifs with a different provider, competition, GAM and features. This project is a documented adaptation, not a reproduction. [Rosvall et al.](https://arxiv.org/abs/1305.4807) provides the memory-state framework; community detection is future work.
