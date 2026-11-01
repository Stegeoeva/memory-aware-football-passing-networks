# Stable-lineup presentation policy

The three case studies reconstruct **exact active membership** and **nominal rosters** separately. Exact membership preserves all 33 recorded intervals, including temporary departures/returns. Nominal membership changes at substitutions and dismissals, retaining 29 intervals: 25 regulation and four extra time. There are 28 eleven-player nominal rosters and one ten-player roster after Carvajal's dismissal.

Temporary Player Off/On events do not split nominal main figures. The player remains in the nominal roster and their absence duration is disclosed. Eleven plotted nodes therefore do not assert eleven simultaneous players at every instant. Exact-active intervals and temporary absences remain separately auditable.

## Boundaries

- Starting XI supplies initial membership; substitutions replace players; red/second-yellow dismissals remove them.
- Multiple substitutions within one stoppage are atomic if no football action intervenes. Their first change event is the common boundary, avoiding artificial intermediate rosters.
- Opponent substitutions do not split the other team's interval. Tactical Shift without membership change updates references rather than splitting a roster.
- An unchanged roster can span half-time; breaks are excluded from duration. Extra time is a separate scope. A roster that returns later remains a separate chronological interval.
- Half-open event-index intervals `start_index, end_index)` assign passes. Period-relative clocks and observed half ends determine duration including stoppage time.

## Display rules

Main figures require **at least five minutes and 20 completed open-play passes**. This gives 24 individual maps plus eight overviews. Five short/sparse intervals remain in all CSV/GraphML exports with exclusion reasons. Different real rosters are never merged to meet a threshold. These thresholds are presentation choices, not statistical tests.

Every edge from **one completed pass** appears. Width is **0.30 pt × pass count**, shared across aggregate and lineup figures. Nodes encode sent+received involvement. Different durations make counts unsuitable for direct rate comparisons.

All nominal players appear. Filled nodes have located completed-pass involvement. Hollow nodes use other events within the same interval or schematic tactical-role references if none exist. `position_source` records the source. Reference positions are not tracking. Small visual offsets separate overlapping labels; grey ticks mark original positions. CSV/GraphML coordinates stay unchanged.

All 2,401 passes are assigned once in each representation: 2,160 regulation and 241 extra time. Summed interval pair counts reproduce the aggregate networks; exact-active checks also prevent absent endpoints. Display filtering does not change the analytical sample.

[Report (`../results/fixed_lineups/report.html`, generated locally) · validation (`../results/fixed_lineups/validation.json`, generated locally) · nominal intervals (`../results/fixed_lineups/tables/segments.csv`, generated locally) · exact audit (`../results/fixed_lineups/tables/active_segments.csv`, generated locally) · temporary absences (`../results/fixed_lineups/tables/temporary_absences.csv`, generated locally).
