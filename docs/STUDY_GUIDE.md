# Study guide

## Research question

When B receives, does knowing previous sender A improve prediction of next recipient C? Do real A–B–C continuations differ in attacking outcomes after allowing for receipt context? The sample covers all 51 EURO 2024 matches. Real paths are reconstructed from event links, rather than inferred from aggregate adjacency matrices.

## Complementary levels

1. **Circulation:** directed weighted networks describe completed-pass volume, prominence and connectivity. Their paths need not have occurred in that order.
2. **Memory:** first order `P(C | B)` and second order `P(C | A,B)` use identical held-out paths. States represent players or zones; memory transitions are `(A,B) → (B,C)`.
3. **Continuation:** outcomes follow an initial receipt or completed motif. Later failures and unknowns remain explicit. Shot frequency, pure source-xG quality and the exploratory mixed R3 proxy are distinct measures.

## Findings

The 6 × 3 zone model gains **0.06079 bits/pass [95% CI 0.05466, 0.06673]** after common context. Identity effects are smaller and baseline-sensitive. Count-based identity memory gains 0.00253 bits/pass with an interval including zero. Retrospective nested and chronological rolling evaluations have different histories and tuning.

Source-xG outcomes cover 33,355 two-pass and 27,173 three-pass motifs. ABCD has higher point estimates than ABAB, but their xG/motif difference includes zero. Average linked-shot quality is close. Neither superiority nor equivalence is established.

R3 supports 127 player estimates, but no positive supported player excess has a pointwise interval wholly above zero. A definitive ranking would overstate the evidence.

## Reading the outputs

| Question | Output |
|---|---|
| What was reconstructed? | Sequence report (`../results/euro_sequences/report.html`, generated locally), typed schema (`../results/euro_sequences/schema.json`, generated locally) |
| Does history help prediction? | Memory report (`../results/memory/report.html`, generated locally), [methods](MEMORY_EVALUATION.md) |
| What follows a player's pass? | Player estimates (`../results/continuation/tables/player_effects.csv`, generated locally), including support and intervals |
| Which real motifs have better continuation? | Motif report (`../results/motifs/report.html`, generated locally), [methods](MOTIFS.md) |
| More shots or better shots? | Source-xG report (`../results/motif_xg/report.html`, generated locally), [methods](MOTIF_XG.md) |
| How do ordinary networks look? | Gallery (`../results/report.html`, generated locally), stable lineups (`../results/fixed_lineups/report.html`, generated locally) |

All raw, adjusted, unsupported and sensitivity results remain in CSV. Highlighted rows are not a selection of significant findings. IDs are join keys; names are labels.

## Interpretation

Continuation depends on the receiver, teammates, opponents, location, pressure and match situation. A backward pass can retain possession or prepare an attack; B's error does not establish that A's pass was poor. Unobserved alternatives cannot be evaluated.

Completed motifs exclude formation failures. Completion/geometry adjustment removes part of the spatial pathway through which a motif may help. Shared outcomes create overlapping credits. Residual differences are descriptive, not causal grades. See the component methods and [reproduction guide](REPRODUCIBILITY.md).
