# Shot frequency, source xG and motif quality

Report with six figures (`../results/motif_xg/report.html`, generated locally) · primary comparison (`../results/motif_xg/tables/primary_comparison.csv`, generated locally) · all contrasts (`../results/motif_xg/tables/pattern_contrasts.csv`, generated locally) · checks (`../results/motif_xg/delivery_check.json`, generated locally).

## Common horizon and observed outcomes

Start **after the final receipt of a completed motif**. The primary horizon is the remainder of the same contiguous provider possession ID/team in the same match/period, stopping earlier at confirmed opponent control, a goal or observed Half End. Event indices preserve source order.

Own restarts would remain if the provider ID/team stayed unchanged and would be flagged; none occurred in this export. Stoppage annotations are separately flagged. Other possessions/opponent actions are not added. A sensitivity uses ten seconds from receipt or the same natural boundary, whichever is earlier. This differs from the earlier bounded-continuous-play window; do not compare the rates as identical-horizon estimates.

Observed Half End closes a remaining-possession window. An unresolved file tail, invalid/incomplete receipt or missing source xG for any selected shot remains missing. **Only an observed complete no-shot window receives zero.** All three primary metrics use the same complete-outcome cohort. Own goals remain distinct without invented shots/xG.

Boundary UUID/index and receipt UUID/index are preserved. For the 10-second horizon, boundary UUID/index describes the underlying possession boundary even if the deadline is earlier; window_end_clock_seconds records the actual earlier time limit.

## Metrics and overlap

| Metric | Definition |
|---|---|
| Shot rate | Complete motifs with at least one later source Shot / all complete motifs |
| xG/motif | Mean sum of all later source-shot xG, including known no-shot zeros |
| xG/linked shot | Sum of linked source xG / number of motif–Shot links |
| Shots/motif | Number of motif–Shot links / complete motifs |
| Unique-shot xG | Mean source xG after deduplicating physical Shot UUIDs within pattern/order/horizon/policy/scope |

Source xG is supplied by StatsBomb. No new xG model is trained and no xT is mixed into this target. All later shots, including rebounds, are retained. Summed xG can exceed one and is not the probability of at least one goal.

**xG/motif = shots/motif × xG/linked shot.** Shot rate differs from shots/motif when multiple shots occur. The quality metric is a ratio of sums, not the mean of each motif's xG/shot ratio.

One physical shot can follow several motifs. The primary ratio counts those links; the unique-shot sensitivity measures each UUID once per group. The same shot can still support different patterns. Credits must not be added across motifs/patterns as independent opportunities.

## Uncertainty and context

Two thousand shared match/whole-possession bootstrap draws preserve overlapping paths. Ratios and paired differences are recomputed in every draw. Frequency/value intervals require ≥50 observed motifs/five matches. Quality additionally requires ≥20 unique shots across five shot-containing matches, with ≥80% valid draws. Smaller groups remain without CI; quality with no shots is undefined, not zero.

New q models fit source-xG sum and shot occurrence separately by motif length/horizon. Five outer/three inner whole-match folds fit encoders and tune only on training matches. Ridge predictions for xG are floored at zero without an upper cap; logistic regression models shot occurrence. There are **80 outer models and 920 total fits**, all converged. Initial-receipt and completion/geometry adjustments remain distinct.

Full-possession xG models **do not consistently beat the training-mean baseline**. Three-pass initial/completion MSE is 0.003995/0.003852 versus 0.003816 baseline. Adjusted residual contrasts are exploratory sensitivities, not proof that context has been removed. Raw source metrics and cluster intervals remain primary. Completion adjustment also removes part of the spatial pathway.

Intervals are pointwise and conditional on fitted q where relevant, without full retraining uncertainty. No family of individual tests or FDR claim was made. The estimates describe associations, not causal effects.

## Primary results: remainder of same possession

| Pattern | Observed motifs | Later shot | xG/motif | xG/linked shot |
|---|---:|---:|---:|---:|
| ABA | 5,918 | 14.40% | 0.01345 | 0.08611 |
| ABC | 27,437 | 16.30% | 0.01574 | 0.08882 |
| ABAB | 831 | 14.92% | 0.01362 | 0.08776 |
| ABAC | 4,237 | 14.54% | 0.01343 | 0.08496 |
| ABCA | 2,678 | 16.62% | 0.01519 | 0.08459 |
| ABCB | 4,115 | 14.78% | 0.01425 | 0.08803 |
| ABCD | 15,312 | 17.18% | 0.01690 | 0.09039 |

Coverage is 33,355/33,358 two-pass and 27,173/27,175 three-pass motifs. Selected shots have source xG; missing cases arise from final-receipt quality.

ABCD versus ABAB yields approximately 1.69 versus 1.36 xG per 100 motifs. The difference is **+0.00328/motif [95% CI −0.00391, +0.00931]**. Linked-shot quality is 0.0904 versus 0.0878, with difference CI [−0.0390, +0.0335]. Unique-shot quality is 0.0926 versus 0.0817; its interval also includes zero. Thus the point estimate mainly accompanies more shots, but neither superiority nor equivalence is established.

All completion/geometry-adjusted three-pass xG/motif contrasts versus ABCD include zero. Completed motifs exclude formation failures, and event data do not reveal unseen alternatives or causal sender responsibility.

## Reproduce and files

~~~shell
python scripts/run_motif_xg.py
python scripts/check_motif_xg.py
~~~

Use the existing memory lock and valid preceding exports. `--stage analysis` fits analysis; `--stage report` regenerates figures/summary/dictionary only. No new dependency or network call is added.

The tables include 83,854 sequence windows, source shot links, motif shot links, 121,066 OOF motif/horizon rows, primary comparison, all metrics/contrasts, model MSE, parameters, fit audit and category coverage. The dictionary documents 190 columns; manifests record code/input/output hashes. Independent checks reconstruct windows, shots, ratios, contrasts and primary paired intervals. Tests cover missingness, boundaries, observed zeros, multi-shot sums above one and uncapped xG predictions.
