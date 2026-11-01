# Memory prediction and downstream continuation

## Prediction sample and models

Use 33,358 actual successful A→B→C transitions satisfying the [sequence rules](EURO_SEQUENCES.md). Next-recipient prediction is conditional on observing a completed next pass, not a model of all attempts or loss risk. Candidates are shared across models and come from the declared team/representation state universe; player names are labels, not keys.

Compare M0, first-order M1, second-order M2, and their common-context M1/M2 variants. Memory states are `(A,B) → (B,C)`. The representations are 6 × 3 zones, 4 × 3 zones and within-team player IDs. Count models use smoothing/backoff for rare or unseen histories. Context models use train-fitted encoders and regularization; unknown categories follow the documented encoder handling rather than borrowing held-out information.

Context X is available at initial receipt: position/zone, pressure, B role, teams/opponent, time/period and reconstructed score. Future-release and final-score fields are excluded. See feature coverage (`../results/continuation/tables/q_feature_coverage.csv`, generated locally), configuration and dictionaries for exact fields.

## Validation and uncertainty

Five outer folds group whole matches, keeping both teams together. Three inner folds select smoothing/regularization without outer labels. Three fixed split repeats are robustness checks; they are not independent samples. This yields 45 outer memory experiments across representations. Full candidate probabilities and IDs are saved for primary folds in NPZ, readable with `allow_pickle=False`.

Primary predictive uncertainty uses 2,000 paired whole-match bootstrap draws on held-out losses. It conditions on fitted OOF models, without the full uncertainty of retraining or repeated-team dependence. Brier score, top-one accuracy, calibration, team/match results and coverage accompany log loss. Rolling evaluation uses only earlier calendar dates and fixed smoothing; its available history differs from retrospective nested CV.

| Representation | M2−M1 improvement, bits/pass [95% CI] | M2+X−M1+X improvement |
|---|---|---|
| 6 × 3 zones | 0.06193 [0.05473, 0.06858] | 0.06079 [0.05466, 0.06673] |
| 4 × 3 zones | 0.06674 [0.06151, 0.07160] | 0.05213 [0.04669, 0.05727] |
| Player IDs | 0.00253 [−0.00200, 0.00656] | 0.00915 [0.00249, 0.01601] |

Spatial history helps beyond the shared receipt context. Identity effects are smaller and baseline-sensitive: identity context models have worse absolute loss than count baselines, and rolling second-order counts deteriorate by 0.08488 bits with less history/fixed smoothing. These are different evaluation conditions, not interchangeable significance tests.

## Continuation estimand

Retain all 41,927 completed initial open-play passes, including subsequent B failures. R1/R3/R5 stop within the first 1/3/5 later ball actions. At the first shot use its source xG; after confirmed loss use zero; for a retained endpoint use a training-only spatial xT surface. Unresolved horizons remain censored. R3 has 39,528 known values, 94.28% coverage.

The 12 × 8 xT model is trained on outer-training matches only, and refitted within inner tuning: **five outer + 15 inner = 20 surfaces**. q predicts the continuation outcome from receipt context, excluding A identity. Encoder fitting, hyperparameter selection and xT all exclude the evaluated match. Pure source-xG motif targets are separate and do not mix xT into their outcomes.

**Adjusted continuation excess per 100 = 100 × mean[R3 − OOF q(X)].** Binary outcome excesses use percentage points. This is a descriptive proxy, not calibrated expected goals or a causal playmaker grade. A backward pass is not automatically a negative action; later failure by B does not isolate A's responsibility.

## Player, pair and triple reporting

Primary R3 support requires players ≥100 known outcomes/three matches, pairs ≥30/two matches and triples ≥20/two matches, plus adequate bootstrap coverage. This supports 127 players, 242 pairs and 30 triples. Unsupported rows remain in CSV. Support is outcome-specific, not based merely on initial pass count.

Matches and whole possessions are resampled hierarchically for continuation uncertainty. Role-specific rank ranges/top-three frequencies disclose ranking instability. Raw values, adjusted excess, 95% intervals, role, coverage and support should be read together. No supported positive primary player estimate has an interval wholly above zero. No family of individual tests or FDR claims was run.

Triple estimates condition on completed B→C and use a separate conditional q baseline. They are not interchangeable with unconditional initial-pass estimates. Delete-match/delete-team sensitivities retain fitted OOF models rather than retraining them. Repeated team appearances and full training uncertainty remain limitations.

## Outputs and reproduction

~~~shell
python scripts/run_memory_study.py
python scripts/check_memory_study.py
~~~

Use `--stage memory`, `--stage continuation`, `--stage report`, and `--representation` to run components. Source restoration and valid sequence exports are prerequisites. Hash-verified ignored checkpoints resume identical runs. [Full reproduction guide](REPRODUCIBILITY.md).

| Directory | Contents |
|---|---|
| results/memory/{representation}/ | OOF/rolling predictions, tuning, audits, splits, primary NPZ probabilities |
| results/memory/tables/ | Prediction metrics, paired intervals, calibration, team/match effects and sensitivities |
| results/continuation/tables/ | OOF outcomes/q/residuals, endpoint material, all entity effects, coverage and audits |
| results/continuation/xt_models.json | All 20 xT surfaces/matrices and exact training matches |
| results/memory/data_dictionary.csv | Every new column; sample-inferred output dtypes, not a substitute for the typed source schema |

CSV.gz loads directly in pandas. Nullable booleans may need explicit `boolean` dtype. NPZ files use no pickle. Locally generated run manifests record source/config/code/output hashes. The public package separately records its aggregate-figure selection in [public release provenance](../provenance/public_release.json).

## Literature and scope

- [Rosvall et al. (2014)](https://arxiv.org/abs/1305.4807): memory states; Infomap communities remain future work.
- [Håland et al. (2020)](https://doi.org/10.1515/jqas-2019-0097): three-pass football motifs; our two-pass base transitions are not a replication.
- [VAEP](https://dtai-static.cs.kuleuven.be/sports/vaep/) and [xT](https://karun.in/blog/expected-threat.html): future-action valuation; R3 is a simpler explicit proxy.
- [Bates, Hastie and Tibshirani](https://arxiv.org/abs/2104.00673): cross-validation uncertainty; folds/repeats do not create independent uncertainty samples.

Findings concern this tournament and conditional sample. Generalizable sender-identity effects require more competitions and richer spatial information.
