# Tournament-wide event sequences

This stage constructs common data for all 51 EURO 2024 matches. It fits no prediction or valuation models. Coverage report (`../results/euro_sequences/report.html`, generated locally) · column dictionary (`../results/euro_sequences/data_dictionary.csv`, generated locally) · typed schema (`../results/euro_sequences/schema.json`, generated locally).

## Sources and counts

Sources are pinned to StatsBomb Open Data commit `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`. All 51 event-file SHA256 hashes and match metadata are verified against committed manifests. Mismatches stop execution. The historical directory name `xt_training` reflects why those files were first downloaded; this construction stage does not fit xT.

| Quantity | Count |
|---|---:|
| Matches / teams | 51 / 24 |
| Original events including shootouts | 187,924 |
| Exported events, periods 1–4 | 187,864 |
| Excluded shootout events | 60 |
| All pass attempts / unsuccessful attempts | 53,888 / 8,514 |
| Completed initial open-play A→B passes | 41,927 |
| Initial passes: regulation / extra time | 40,612 / 1,315 |
| Observed next B attempts / unsuccessful | 38,357 / 4,979 |
| Eligible successful A→B→C memory targets | 33,358 |
| Linked receipt events / receipt fallbacks | 41,892 / 35 |
| Receipt fallbacks beyond a boundary | 5 |
| Future-window rows | 293,489 |

Five estimated receipts extend past an injury/period boundary. They remain flagged, censored and excluded from memory targets; times are not silently shifted. Twenty additional completed next passes have an initial-team/provider-possession-team mismatch and remain in full tables with exclusion flags.

## Tables and keys

| Table | Observation unit |
|---|---|
| matches | Match, source and audit counts; final score is for validation |
| events | Original event, periods 1–4, with UUID/index, location, pressure, role and reconstructed score |
| players | Observed player/team/match from events/tactics; not the entire unused bench |
| pass_attempts | Every attempt, including failures/restarts and filter reasons |
| receiver_sequences | One completed initial A→B; receipt context and separate future outcomes |
| two_pass_sequences | B's actual next attempt; actual C exists only after completion |
| sequence_actions | First up to five future ball actions, with order and source UUID |
| future_windows | Initial pass × action/time/remaining-bounded-possession horizon |
| possession_runs | Contiguous provider possession ID/team, with opponent-control checks |
| coverage_by_match_team | All 102 team–matches and observation coverage |
| zone_definitions | Exact boundaries for 6 × 3 and 4 × 3 grids |
| role_definitions | Provider positions and declared broad roles |
| legacy_comparison | All 2,401 original case-study passes versus the previous export |

Use IDs rather than names for joins. Sequence IDs identify initial passes. Nullable integer/boolean dtypes and every field's provenance/missingness are specified in the schema; CSV.gz loads directly with pandas. Fixed gzip timestamps make data export reproducible.

## Continuity and context

Source event indices determine order. Initial passes are completed open-play attempts with an identified different recipient, excluding direct restarts. All attempts remain in the attempts table for completion/loss analysis. Resolution follows the same receiving player, allowing their carries/successful dribbles and declared same-player advantage events; another handler is not skipped. A failed next pass retains its intended recipient separately rather than inventing an actual C.

Paths stop at possession/period boundaries, confirmed opponent control and relevant interruptions. Missing linked receipts fall back to pass duration/endpoint, with unobserved pressure. Already observed Starting XI/tactics and event positions determine roles; future pass-release information is not used at initial receipt. Role fallbacks are recorded and are not certain tactical-role measurements.

Receipt features separate initial location/zones, receipt pressure, B role, teams, time/period and reconstructed pre-event score from future-release fields and outcomes. Final metadata scores are audit-only. Paired own-goal annotations are counted once; all reconstructed final scores match metadata. Shootouts are excluded; in-match penalty shots remain.

## Future horizons and censoring

Outcomes cover 1/3/5 later ball actions, 5/10/20 seconds from receipt, and the remaining bounded-continuous-play suffix. Each keeps source UUIDs, observation coverage, source-shot xG and explicit censoring. An early ambiguous boundary is not automatically a known no-shot zero. A confirmed loss can determine an outcome before the nominal horizon ends.

One later chance may follow many earlier passes: future-xG credits overlap and cannot be summed as independent chances. This stage adds no xT trained on evaluation matches. Its bounded-continuous-play windows are distinct from the later [source-xG extension's provider-possession horizon](MOTIF_XG.md).

All 2,401 legacy case-study passes remain. Two extra receiver-pass links are recovered by allowing the declared same-receiver Foul Won advantage rule; differences are logged.

## Reproduce and validate

~~~shell
python scripts/download_data.py
python scripts/download_xt_training.py
python scripts/build_euro_sequences.py
python scripts/check_euro_sequences.py
~~~

After source restoration, the builder is local. The source/run manifests record URLs, bytes, hashes, environment and rules; validation checks IDs, links, coverage, scores, typed roundtrips and legacy inclusion. This is event data, not frame-by-frame tracking.
