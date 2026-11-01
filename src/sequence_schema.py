"""Column definitions and feature/outcome separation for the data-only export."""
import pandas as pd

RECEIPT_FEATURES = [
    "team_id", "opponent_id", "a_id", "b_id", "period", "scope",
    "a_clock_seconds", "a_duration_seconds", "a_x", "a_y", "a_end_x", "a_end_y",
    "a_position_id", "a_role", "a_under_pressure", "a_pass_length", "a_pass_height",
    "a_start_zone_6x3", "a_start_zone_4x3", "a_end_zone_6x3", "a_end_zone_4x3",
    "receipt_clock_seconds", "receipt_elapsed_seconds", "receipt_x", "receipt_y",
    "receipt_zone_6x3", "receipt_zone_4x3", "b_position_id", "b_role",
    "b_under_pressure_at_receipt", "b_pressure_state", "home_score_at_receipt", "away_score_at_receipt",
    "score_difference_at_receipt",
]

FIELD_HELP = {
    "match_id": "Provider match ID; grouping unit for future validation splits.",
    "event_id": "Original event UUID, globally unique in this export.",
    "index": "Source order within match; authoritative for reconstruction, not rounded time.",
    "sequence_id": "Stable composite match_id:initial-pass-event-UUID.",
    "period": "1/2 regulation, 3/4 extra time; shootout period 5 excluded.",
    "scope": "regulation or extra_time; added time remains in the corresponding period.",
    "clock_seconds": "Seconds from period start, parsed from source timestamp.",
    "elapsed_seconds": "Cumulative period time including added time, excluding half-time breaks.",
    "duration_seconds": "Original action duration; absent/null becomes zero.",
    "team_id": "Provider event actor / initial passing team ID, as specified by table.",
    "team": "Provider team name; IDs define identity.",
    "opponent_id": "Other participating team's provider ID.",
    "opponent": "Other participating team's source name.",
    "possession_id": "Original provider possession ID within match.",
    "possession_team_id": "Provider possession team, which may differ from a defensive actor's team.",
    "possession_key": "match:period:possession composite key.",
    "possession_run_key": "possession_key:first-event-index for a contiguous provider run.",
    "player_id": "Provider player ID; nullable for administrative/team-only events.",
    "player": "Provider full name for display, not an identity key.",
    "position_id": "Position from event or latest already observed tactical/position record.",
    "position": "Provider name of the corresponding position.",
    "role": "Coarse, declared grouping of provider positions, not inferred tracking roles.",
    "role_source": "event_position, Starting XI, Tactical Shift, substitution_inherited_position or unobserved.",
    "role_source_event_id": "Supporting source UUID; cannot occur after context observation time.",
    "event_type": "Original event type name.",
    "outcome": "Source outcome; absent becomes Complete only for Pass/Carry/Ball Receipt*.",
    "completed": "Successful Pass/Carry/Ball Receipt* or Dribble Complete; false for other types.",
    "x": "Event origin x, team-relative StatsBomb 120x80 units; attack toward x=120.",
    "y": "Event origin y, team-relative StatsBomb coordinates.",
    "end_x": "Recorded action endpoint x; nullable, never fabricated for a take-on.",
    "end_y": "Recorded action endpoint y; nullable if the action has no endpoint.",
    "under_pressure": "Source flag defaulting false if absent; not flagged does not prove no pressure.",
    "pressure_flag_present": "Whether original event includes under_pressure, separate from its value.",
    "recipient_id": "Source pass recipient; on failure this can be intended rather than actually reached.",
    "recipient": "Source pass recipient name; actual receipt not implied on failures.",
    "pass_type": "Source pass.type.name; missing denotes ordinary type.",
    "pass_length": "Source pass.length in coordinate units, not necessarily metres.",
    "pass_height": "Source pass.height.name.",
    "pass_angle": "Source pass.angle in radians.",
    "body_part": "Source body_part.name where supplied.",
    "xg": "Original shot.statsbomb_xg; no model fitted here; null on non-shots.",
    "shot_type": "Source shot.type.name, including in-match penalties in periods 1-4.",
    "play_pattern": "Source play pattern; restart filtering uses direct pass type.",
    "on_ball_action": "Pass/Carry/Dribble/Shot/Dispossessed/Miscontrol/Clearance; receipt/pressure do not count.",
    "controlled_touch": "Documented control heuristic; successful receipts included, mere pressure/block excluded.",
    "stoppage_reason": "Explicit restart/foul without advantage/out/goal/interruption, or null.",
    "out": "Source event out flag, default false.",
    "off_camera": "Source off_camera flag, default false; does not guarantee full observation.",
    "related_event_ids": "Semicolon-separated original related-event UUIDs; preferred receipt linkage.",
    "home_score_before": "Home score strictly before event; paired own goals count once.",
    "away_score_before": "Away score strictly before event, excluding shootout goals.",
    "home_score_after": "Home score after the canonical increment at this event.",
    "away_score_after": "Away score after the canonical increment at this event.",
    "scoring_team_id": "Canonical increment at Shot Goal or Own Goal Against; duplicate Own Goal For excluded.",
    "goal_beneficiary_team_id": "Beneficiary of goal including paired own-goal views; not additive goal counts.",
    "open_play": "Pass type outside Corner/Free Kick/Goal Kick/Kick Off/Throw-in.",
    "initial_pass_eligible": "Completed open-play pass with distinct named passer/recipient, without lineup display filtering.",
    "initial_exclusion_reason": "Semicolon-separated restart/unsuccessful/missing identity/self-pass reasons.",
    "next_control_event_id": "First subsequent recorded control in the same period, possibly after restart.",
    "next_control_team_id": "Team at that next control, null at period end.",
    "seconds_to_next_control": "Action start to next recorded control; no causal blame implied.",
    "next_control_is_restart": "Next controller's event is a direct restart.",
    "next_control_status": "same_team_control / opponent_control / censored_period_end.",
    "a_id": "Initial passer A's provider ID.", "a": "A's provider full name.",
    "b_id": "Initial recipient and prospective next passer B's provider ID.", "b": "B's full name.",
    "c_id": "Actually reached C only on a completed B pass; null on all failures.",
    "c": "Actually reached C's full name, null for unsuccessful next passes.",
    "a_event_id": "UUID of initial completed open-play A->B.",
    "b_event_id": "UUID of observed B next-pass attempt, including failures.",
    "a_index": "Source event index of initial A pass.",
    "receipt_event_id": "Linked Ball Receipt* UUID or null.",
    "receipt_source": "related_event / local_event_order / pass_duration_fallback / ambiguous_receipts.",
    "receipt_candidates": "Number of matching candidates at the selected linkage stage.",
    "receipt_observed": "An actual receipt event was linked.",
    "receipt_completed": "Linked receipt completion, null for unobserved receipt.",
    "receipt_clock_seconds": "Receipt timestamp or pass start plus source duration.",
    "receipt_elapsed_seconds": "Receipt clock on cumulative period timeline excluding breaks.",
    "receipt_x": "Observed receipt location x, falling back to initial pass endpoint.",
    "receipt_y": "Observed receipt location y, falling back to initial pass endpoint.",
    "receipt_coordinate_source": "receipt_location or pass_endpoint; event positions, not tracking.",
    "b_under_pressure_at_receipt": "Only linked receipt pressure; null if absent receipt. Never borrowed from release.",
    "b_pressure_flag_present_at_receipt": "Presence of under_pressure in actual receipt, null if receipt absent.",
    "b_pressure_state": "flagged / not_flagged / unobserved_receipt.",
    "home_score_at_receipt": "Score before actual receipt or explicitly marked pass-start proxy.",
    "away_score_at_receipt": "Score before actual receipt or pass-start proxy; never final score.",
    "score_difference_at_receipt": "Initial passing team's goals minus opponent goals at observation time.",
    "score_context_source": "receipt_pre_event or pass_start_proxy.",
    "stop_event_id": "First boundary UUID; null for end of available period without boundary event.",
    "stop_index": "Original index of first boundary event.",
    "stop_reason": "Earliest opponent control/provider possession change/explicit interruption/period end.",
    "stop_clock_seconds": "Boundary start, or start+duration for included own terminal action.",
    "stop_action_included": "Own terminal action included; direct restart itself always excluded.",
    "stop_possession_id": "Provider possession ID at boundary.",
    "receipt_after_boundary": "Receipt time exceeds sequence boundary; anomaly is not silently clipped.",
    "provider_possession_team_matches": "Provider possession team equals initial actor team.",
    "receiver_status": "observed / other_teammate_handler / no_resolution_before_boundary.",
    "receiver_carries_before_release": "Actual B carries before release/resolution.",
    "receiver_dribbles_before_release": "Successful B take-ons before release/resolution.",
    "receiver_resolution_event_id": "First B resolution after optional carries/successful take-ons; null if censored.",
    "receiver_resolution_type": "B resolution type, e.g. pass, shot or miscontrol.",
    "receiver_resolution_outcome": "Original B resolution outcome.",
    "receiver_resolution_clock_seconds": "B resolution period clock.",
    "other_handler_event_id": "Intervening teammate event preventing a manufactured A-B-C link.",
    "other_handler_id": "ID of that intervening teammate.",
    "b_next_pass_observed": "B resolution is a pass attempt before any boundary.",
    "b_next_pass_completed": "Observed next attempt completed; denominator is observed attempts.",
    "future_actions_before_boundary": "All own on-ball actions in bounded future suffix.",
    "first_future_action_event_id": "First future own-action UUID, not necessarily B.",
    "receiver_next_control_status": "Next control after B resolution, possibly at restart; null if unresolved.",
    "receiver_next_control_event_id": "UUID supporting that next-control classification.",
    "receiver_loss_confirmed": "Recorded B failure followed by opponent control; unknown/censored remains null.",
    "b_intended_recipient_id": "Source named/intended recipient of B attempt, kept even on failure.",
    "b_intended_recipient": "Its source name; distinct from actual C.",
    "b_completed": "B next-pass completion from source absence of pass.outcome.",
    "b_open_play": "Next B pass is not a direct restart.",
    "b_outcome": "Source next-pass outcome, Complete if absent.",
    "seconds_between_pass_starts": "B start clock minus A start clock.",
    "seconds_from_receipt_to_release": "B start minus actual/estimated receipt time.",
    "b_dx": "Next-pass end_x minus origin x in coordinate units.",
    "b_direction": "forward if dx>5, backward if dx<-5, else lateral; not a quality judgment.",
    "b_under_pressure_at_release": "Release-time pressure: FUTURE relative to initial receipt.",
    "b_position_id_at_release": "Release-time position: FUTURE relative to initial receipt.",
    "c_role_at_release": "C role known at B release: FUTURE relative to initial receipt.",
    "c_role_source_event_id": "Source UUID supporting release-time C role.",
    "memory_target_eligible": "Actual successful A-B-C with valid timing/provider continuity; no model fitted.",
    "memory_target_exclusion_reason": "Semicolon-separated explicit reasons the next pass is not a primary memory target; empty if eligible.",
    "opponent_noncontrol_annotations": "Count of intervening opposing on-ball annotations without confirmed control, e.g. Miscontrol; retained for strict-path sensitivity.",
    "opponent_noncontrol_event_ids": "UUIDs of those ambiguous opposing-touch annotations; not receipt-time predictor features.",
    "spatial_target_eligible": "Memory target plus all three required 6x3 zones observed.",
    "pattern": "ABA return or ABC third player; null on B failure; only a two-pass label.",
    "step": "One-based own future on-ball action number, capped at five.",
    "seconds_from_receipt": "Future action start minus actual/estimated receipt.",
    "horizon_kind": "actions / seconds / possession; time horizons start at receipt.",
    "horizon_value": "1/3/5 actions or 5/10/20 seconds; null for bounded possession suffix.",
    "observed_actions": "Actual own on-ball actions recorded in window.",
    "horizon_reached": "Action/time cutoff reached; false for the unbounded possession suffix.",
    "observation_complete": "Cutoff reached or observed opponent control/goal terminates the outcome.",
    "censored": "Not observation_complete; short period/stoppage is not a known negative.",
    "termination_in_window": "Boundary before action/time cutoff, or full bounded suffix requested.",
    "termination_reason": "Boundary reason inside this window, otherwise null.",
    "confirmed_loss_in_window": "Observed opponent-control boundary inside this horizon.",
    "goal_terminal_in_window": "A goal boundary, either for or against initial passing team.",
    "goal_for_team_in_window": "Goal boundary benefited initial passing team.",
    "goal_against_team_in_window": "Goal boundary benefited opponent.",
    "shot_observed": "At least one actual own shot in the window.",
    "shot_outcome_known": "Observed positive shot or complete observation; otherwise cannot assume zero.",
    "shots": "Number of actual own shots in window/run.",
    "source_xg_sum": "Sum of original shot xG; overlapping windows/passes do not represent independent chances.",
    "shot_event_ids": "Ordered UUIDs of contributing source shots.",
    "box_entry_observed": "Completed move from outside box to x>=102 and 18<=y<=62.",
    "box_entry_outcome_known": "Positive box entry or complete observation.",
    "box_entries": "Completed box-entry moves; failed attempted entries excluded.",
    "final_third_entries": "Completed moves from x<80 to x>=80.",
    "max_forward_gain": "Maximum nonnegative completed-move endpoint x minus initial pass endpoint x.",
    "last_action_event_id": "Last observed own action UUID in window.",
    "last_action_type": "Its source event type.",
    "last_action_completed": "Its completion flag; a failed final action is not automatically retained possession.",
    "last_action_end_x": "Its recorded endpoint x, nullable.",
    "last_action_end_y": "Its recorded endpoint y, nullable.",
    "action_event_ids": "Ordered own-action UUIDs; join to events for actual trajectories.",
    "intervening_opponent_controls": "Opponent controls inside provider possession run; own chains stop before these.",
    "legacy_b_event_id": "Next-pass UUID in existing three-match extension.",
    "current_b_event_id": "Next-pass UUID under current documented rules.",
    "next_pass_agrees": "Both UUIDs equal, including both missing.",
    "difference": "unchanged / new_observation / removed_or_changed.",
    "current_stop_reason": "Current bounded-suffix termination reason.",
    "current_receiver_status": "Current B-resolution status.",
}


def column_info(table, column, row_units):
    description = FIELD_HELP.get(column)
    if description is None:
        for prefix, actor in [("a_", "initial A"), ("b_", "B at receipt"), ("recipient_", "recipient known at this event")]:
            base = column.removeprefix(prefix)
            if column.startswith(prefix) and base in FIELD_HELP:
                description = f"For {actor}: " + FIELD_HELP[base]
                break
    if "zone_" in column:
        grid = "6x3" if "6x3" in column else "4x3"
        description = f"Zero-based {grid} row-major zone for {column.replace('_', ' ')}; null for invalid/missing coordinates."
    if description is None:
        description = f"{column.replace('_', ' ')}; metadata/count for: {row_units[table]}"
    if table == "receiver_sequences":
        availability = "as_of_receipt" if column in RECEIPT_FEATURES else (
            "future_outcome" if column.startswith(("stop_", "receiver_", "other_handler", "future_", "first_future_", "b_next_"))
            else "audit_or_identifier")
    elif table in {"two_pass_sequences", "future_windows", "sequence_actions"}:
        availability = "future_outcome_or_join_key"
    elif table in {"matches", "possession_runs", "coverage_by_match_team", "legacy_comparison"}:
        availability = "audit_only_not_receipt_features"
    elif table in {"zone_definitions", "role_definitions", "players"}:
        availability = "lookup_metadata"
    else:
        availability = "event_record_mixed_context_and_outcome"
    if "role" in column or "position" in column:
        provenance = "event.position / already observed Starting XI or Tactical Shift / substitution; source UUID retained"
    elif "pressure" in column:
        provenance = "source under_pressure; reception reads only linked Ball Receipt*"
    elif column in {"xg", "source_xg_sum"}:
        provenance = "original shot.statsbomb_xg; sums join explicitly referenced shot events"
    elif column in {"event_id", "index", "player_id", "team_id", "possession_id", "related_event_ids"}:
        provenance = "original event JSON; provider identifiers preserved"
    else:
        provenance = "verified original events/match metadata, deterministic derivation described in methodology"
    return {"description": description, "availability": availability, "provenance": provenance}


def data_dictionary(tables, row_units):
    return pd.DataFrame([{"table": name, "column": col, "dtype": str(frame[col].dtype),
                          "rows": len(frame), "missing": int(frame[col].isna().sum()),
                          **column_info(name, col, row_units)}
                         for name, frame in tables.items() for col in frame])
