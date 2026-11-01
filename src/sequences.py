"""Build auditable EURO event sequences without fitting or evaluating models.

All context is observed at the initial pass/reception; later fields are labelled
outcomes. Event indices order records. Period clocks measure time. Physical
touches and explicit stoppages stop chains even if possession IDs do not change.
"""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import ROOT, MATCH_IDS, RESTARTS, SOURCE_COMMIT, SOURCE_URL
from .lineups import seconds, period_bounds

SCHEMA_VERSION = "1.0.0"
ACTION_HORIZONS = (1, 3, 5)
TIME_HORIZONS = (5, 10, 20)
ON_BALL = {"Pass", "Carry", "Dribble", "Shot", "Dispossessed", "Miscontrol", "Clearance"}
EPSILON = .001


def role_group(position_id):
    if position_id is None or pd.isna(position_id):
        return None
    p = int(position_id)
    if p == 1:
        return "goalkeeper"
    for lo, hi, name in [(2, 8, "defender"), (9, 11, "defensive_midfielder"),
                         (12, 16, "midfielder"), (17, 21, "attacking_midfielder_winger"),
                         (22, 25, "forward")]:
        if lo <= p <= hi:
            return name
    return "unknown_position"


def valid_point(x, y):
    return x is not None and y is not None and np.isfinite([x, y]).all() and 0 <= x <= 120 and 0 <= y <= 80


def pitch_zone(x, y, columns=6, rows=3):
    """Zero-based, row-major zones. Exact outer boundaries stay in the pitch."""
    if not valid_point(x, y):
        return None
    return min(columns - 1, int(x * columns / 120)) + columns * min(rows - 1, int(y * rows / 80))


def point(value):
    return tuple(value[:2]) if isinstance(value, list) and len(value) >= 2 else (None, None)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inventory(root=ROOT):
    """Verify the existing 3+48 caches against their pinned input manifests."""
    root = Path(root)
    original = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    training = json.loads((root / "data/xt_training_manifest.json").read_text(encoding="utf-8"))
    if original["commit"] != SOURCE_COMMIT or training["commit"] != SOURCE_COMMIT:
        raise ValueError("Input caches must use the pinned source commit")
    metadata_path = root / "data/open-data/data/matches/55/282.json"
    if sha256(metadata_path) != original["sha256"]["data/matches/55/282.json"]:
        raise ValueError("Match metadata checksum mismatch")
    matches = sorted(json.loads(metadata_path.read_text(encoding="utf-8")),
                     key=lambda m: (m["match_date"], m["kick_off"], m["match_id"]))
    ids = {m["match_id"] for m in matches}
    if len(matches) != 51 or len(ids) != 51 or ids != set(MATCH_IDS) | set(training["training_match_ids"]):
        raise ValueError("Expected exactly the 51 EURO 2024 matches")
    files = []
    for match in matches:
        mid = match["match_id"]
        if mid in MATCH_IDS:
            path = root / f"data/open-data/data/events/{mid}.json"
            expected = original["sha256"][f"data/events/{mid}.json"]
        else:
            path = root / f"data/xt_training/events/{mid}.json"
            expected = training["files"][str(mid)]["sha256"]
        if not path.is_file():
            raise FileNotFoundError(f"Run the existing download scripts first: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Event checksum mismatch: {mid}")
        files.append({"match_id": mid, "path": path.relative_to(root).as_posix(), "sha256": actual,
                      "bytes": path.stat().st_size,
                      "source_url": f"https://raw.githubusercontent.com/hudl/open-data/{SOURCE_COMMIT}/data/events/{mid}.json"})
    manifest = {"repository": SOURCE_URL, "commit": SOURCE_COMMIT, "competition_id": 55, "season_id": 282,
                "match_metadata": {"path": metadata_path.relative_to(root).as_posix(), "sha256": sha256(metadata_path)},
                "match_ids": [m["match_id"] for m in matches], "files": files}
    return matches, manifest


def is_control(kind, detail, complete):
    if kind in {"Pass", "Carry", "Dribble", "Shot"}:
        return True  # The actor controlled the ball before executing the action.
    if kind == "Ball Receipt*":
        return complete
    if kind == "Ball Recovery":
        return not detail.get("recovery_failure", False)
    if kind == "Interception":
        return detail.get("outcome", {}).get("name") in {"Won", "Success In Play"}
    if kind == "Goal Keeper":
        return (detail.get("type", {}).get("name") in
                {"Collected", "Keeper Sweeper", "Smother", "Shot Saved", "Shot Saved Off Target", "Save", "Penalty Saved"}
                and detail.get("outcome", {}).get("name") in
                {"Success", "Claim", "Collected Twice", "Won", "Success In Play", "Saved Twice"})
    return False


def stoppage_reason(event, kind, detail):
    if kind == "Pass" and detail.get("type", {}).get("name") in RESTARTS:
        return "restart"
    if kind in {"Foul Won", "Foul Committed"} and not detail.get("advantage", False):
        return "foul_stoppage"
    if kind in {"Half End", "Injury Stoppage", "Substitution", "Referee Ball-Drop", "Offside"}:
        return kind.lower().replace(" ", "_")
    if kind in {"Own Goal Against", "Own Goal For"} or (kind == "Shot" and detail.get("outcome", {}).get("name") == "Goal"):
        return "goal"
    if event.get("out", False):
        return "ball_out"
    if kind == "Shot" and detail.get("outcome", {}).get("name") in {"Off T", "Wayward", "Saved Off T"}:
        return "shot_out"
    return None


def normalise_match(raw, match):
    """Flatten records, reconstruct score, and cache roles using past events only."""
    raw = sorted(raw, key=lambda e: e["index"])
    assert len({e["index"] for e in raw}) == len(raw)
    assert len({e["id"] for e in raw}) == len(raw)
    bounds = period_bounds(raw)
    home, away = match["home_team"]["home_team_id"], match["away_team"]["away_team_id"]
    scores = {home: 0, away: 0}
    names = {home: match["home_team"]["home_team_name"], away: match["away_team"]["away_team_name"]}
    roles, participants, records = {}, {}, []
    for e in raw:
        if e["period"] not in (1, 2, 3, 4):
            continue
        kind, tid = e["type"]["name"], e["team"]["id"]
        assert tid in names
        pid = e.get("player", {}).get("id")
        payload_name = {"Ball Receipt*": "ball_receipt", "Goal Keeper": "goalkeeper"}.get(kind, kind.lower().replace(" ", "_"))
        detail = e.get(payload_name, {})
        if kind in {"Starting XI", "Tactical Shift"}:
            for p in e.get("tactics", {}).get("lineup", []):
                pp = p["player"]; pos = p["position"]
                roles[(tid, pp["id"])] = (pos["id"], pos["name"], kind, e["id"])
                participants[(tid, pp["id"])] = pp["name"]
        if pid is not None:
            participants[(tid, pid)] = e["player"]["name"]
            if "position" in e:
                pos = e["position"]
                roles[(tid, pid)] = (pos["id"], pos["name"], "event_position", e["id"])
        if kind == "Substitution":
            replacement = detail["replacement"]
            participants[(tid, replacement["id"])] = replacement["name"]
            previous = roles.get((tid, pid), (None, None, None, None))
            roles[(tid, replacement["id"])] = (previous[0], previous[1], "substitution_inherited_position", e["id"])
        position = roles.get((tid, pid), (None, None, "unobserved", None))
        rec = detail.get("recipient", {}) if kind == "Pass" else {}
        rid = rec.get("id")
        if rid is not None:
            participants[(tid, rid)] = rec["name"]
        rec_position = roles.get((tid, rid), (None, None, "unobserved", None))
        x, y = point(e.get("location")); ex, ey = point(detail.get("end_location"))
        outcome = detail.get("outcome", {}).get("name")
        complete = ((kind in {"Pass", "Carry", "Ball Receipt*"} and outcome is None)
                    or (kind == "Dribble" and outcome == "Complete"))
        if complete and outcome is None:
            outcome = "Complete"
        t = seconds(e)
        duration = float(e.get("duration") or 0.)
        row = {"match_id": match["match_id"], "event_id": e["id"], "index": e["index"],
               "period": e["period"], "scope": "regulation" if e["period"] <= 2 else "extra_time",
               "clock_seconds": t, "elapsed_seconds": bounds[e["period"]]["offset"] + t,
               "duration_seconds": duration, "team_id": tid, "team": names[tid],
               "opponent_id": away if tid == home else home, "opponent": names[away if tid == home else home],
               "possession_id": e["possession"], "possession_team_id": e["possession_team"]["id"],
               "possession_key": f"{match['match_id']}:{e['period']}:{e['possession']}",
               "player_id": pid, "player": e.get("player", {}).get("name"),
               "position_id": position[0], "position": position[1], "role": role_group(position[0]),
               "role_source": position[2], "role_source_event_id": position[3],
               "event_type": kind, "outcome": outcome, "completed": complete,
               "x": x, "y": y, "end_x": ex, "end_y": ey,
               "start_zone_6x3": pitch_zone(x, y), "end_zone_6x3": pitch_zone(ex, ey),
               "start_zone_4x3": pitch_zone(x, y, 4, 3), "end_zone_4x3": pitch_zone(ex, ey, 4, 3),
               "under_pressure": bool(e.get("under_pressure", False)), "pressure_flag_present": "under_pressure" in e,
               "recipient_id": rid, "recipient": rec.get("name"),
               "recipient_position_id": rec_position[0], "recipient_position": rec_position[1],
               "recipient_role": role_group(rec_position[0]), "recipient_role_source": rec_position[2],
               "recipient_role_source_event_id": rec_position[3],
               "pass_type": detail.get("type", {}).get("name") if kind == "Pass" else None,
               "pass_length": detail.get("length") if kind == "Pass" else None,
               "pass_height": detail.get("height", {}).get("name") if kind == "Pass" else None,
               "pass_angle": detail.get("angle") if kind == "Pass" else None,
               "body_part": detail.get("body_part", {}).get("name"),
               "xg": detail.get("statsbomb_xg") if kind == "Shot" else None,
               "shot_type": detail.get("type", {}).get("name") if kind == "Shot" else None,
               "play_pattern": e.get("play_pattern", {}).get("name"),
               "on_ball_action": kind in ON_BALL, "controlled_touch": is_control(kind, detail, complete),
               "stoppage_reason": stoppage_reason(e, kind, detail), "out": bool(e.get("out", False)),
               "off_camera": bool(e.get("off_camera", False)),
               "related_event_ids": ";".join(e.get("related_events", [])),
               "home_score_before": scores[home], "away_score_before": scores[away]}
        row["open_play"] = kind == "Pass" and row["pass_type"] not in RESTARTS
        exclusions = []
        if kind == "Pass":
            if not row["open_play"]: exclusions.append("restart")
            if not complete: exclusions.append("unsuccessful")
            if pid is None: exclusions.append("missing_passer")
            if rid is None: exclusions.append("missing_recipient")
            if pid is not None and pid == rid: exclusions.append("self_pass")
        row["initial_pass_eligible"] = kind == "Pass" and not exclusions
        row["initial_exclusion_reason"] = ";".join(exclusions) if kind == "Pass" else None
        scoring_team = tid if kind == "Shot" and outcome == "Goal" else (row["opponent_id"] if kind == "Own Goal Against" else None)
        row["scoring_team_id"] = scoring_team
        row["goal_beneficiary_team_id"] = tid if kind == "Own Goal For" else scoring_team
        if scoring_team is not None:
            scores[scoring_team] += 1
        row.update(home_score_after=scores[home], away_score_after=scores[away])
        records.append(row)
    expected = (match.get("home_score"), match.get("away_score"))
    if expected != (None, None) and (scores[home], scores[away]) != expected:
        raise ValueError(f"Goal reconstruction mismatch: {match['match_id']}: {scores} versus {expected}")
    players = [{"match_id": match["match_id"], "team_id": tid, "team": names[tid],
                "player_id": pid, "player": name} for (tid, pid), name in sorted(participants.items())]
    return records, players


class PeriodIndex:
    """Indices for real possession runs, controls and restart/stoppage boundaries."""
    def __init__(self, rows):
        self.rows = rows
        self.by_id = {r["event_id"]: i for i, r in enumerate(rows)}
        self.period_end = max(r["clock_seconds"] for r in rows)
        n = len(rows)
        self.run_end = [n] * n
        for i in range(n - 2, -1, -1):
            a, b = rows[i], rows[i + 1]
            self.run_end[i] = i + 1 if (a["possession_id"], a["possession_team_id"]) != (b["possession_id"], b["possession_team_id"]) else self.run_end[i + 1]
        self.controls = [i for i, r in enumerate(rows) if r["controlled_touch"]]
        self.stops = [i for i, r in enumerate(rows) if r["stoppage_reason"]]
        self.team_controls = {tid: [i for i in self.controls if rows[i]["team_id"] == tid]
                              for tid in {r["team_id"] for r in rows}}

    @staticmethod
    def after(indices, i, default):
        j = bisect_right(indices, i)
        return indices[j] if j < len(indices) else default

    def next_control(self, i):
        j = self.after(self.controls, i, len(self.rows))
        return self.rows[j] if j < len(self.rows) else None

    def control_fields(self, i):
        r, following = self.rows[i], self.next_control(i)
        return {"next_control_event_id": following["event_id"] if following else None,
                "next_control_team_id": following["team_id"] if following else None,
                "seconds_to_next_control": following["clock_seconds"] - r["clock_seconds"] if following else None,
                "next_control_is_restart": bool(following and following["stoppage_reason"] == "restart"),
                "next_control_status": "censored_period_end" if following is None else
                    "same_team_control" if following["team_id"] == r["team_id"] else "opponent_control"}

    def boundary(self, i):
        n, r = len(self.rows), self.rows[i]
        opponent = self.team_controls.get(r["opponent_id"], [])
        j = min(self.run_end[i], self.after(opponent, i, n), self.after(self.stops, i, n))
        if j == n:
            return j, "period_end", False, self.period_end
        end = self.rows[j]
        if end["controlled_touch"] and end["team_id"] != r["team_id"]:
            reason, inclusive = "opponent_control", False
        elif end["stoppage_reason"]:
            reason = end["stoppage_reason"]
            inclusive = (end["team_id"] == r["team_id"] and end["on_ball_action"] and reason != "restart")
        else:
            reason, inclusive = "possession_boundary", False
        end_time = end["clock_seconds"] + (end["duration_seconds"] if inclusive else 0.)
        return j, reason, inclusive, end_time

    def receipt(self, i, stop):
        a = self.rows[i]
        def valid(j):
            r = self.rows[j]
            return (j > i and j < stop and r["event_type"] == "Ball Receipt*"
                    and r["team_id"] == a["team_id"] and r["player_id"] == a["recipient_id"]
                    and r["possession_id"] == a["possession_id"])
        direct = [self.by_id[key] for key in a["related_event_ids"].split(";") if key in self.by_id and valid(self.by_id[key])]
        if direct:
            return self.rows[min(direct)], "related_event", len(direct)
        candidates = []
        for j in range(i + 1, stop):
            r = self.rows[j]
            if r["clock_seconds"] > a["clock_seconds"] + a["duration_seconds"] + 2:
                break
            if valid(j):
                candidates.append(r)
            if r["on_ball_action"] or (r["controlled_touch"] and r["player_id"] != a["recipient_id"]):
                break
        if len(candidates) == 1:
            return candidates[0], "local_event_order", 1
        return None, "ambiguous_receipts" if candidates else "pass_duration_fallback", len(candidates)


def receiver_resolution(index, i, stop, inclusive):
    a = index.rows[i]
    carries = dribbles = 0
    for j in range(i + 1, stop + int(inclusive)):
        r = index.rows[j]
        if r["team_id"] != a["team_id"]:
            continue
        handler = r["on_ball_action"] or r["controlled_touch"] or r["event_type"] in {"Foul Won", "Shield"}
        if not handler:
            continue
        if r["player_id"] != a["recipient_id"]:
            return None, "other_teammate_handler", carries, dribbles, r
        if r["event_type"] == "Carry":
            carries += 1
        elif r["event_type"] == "Dribble" and r["completed"]:
            dribbles += 1
        elif r["event_type"] == "Foul Won" and r["stoppage_reason"] is None:
            continue  # Advantage with the same handler does not interrupt play.
        elif r["on_ball_action"] or r["event_type"] == "Foul Won":
            return r, "observed", carries, dribbles, None
    return None, "no_resolution_before_boundary", carries, dribbles, None


def window_summary(a, sequence_id, actions, stop, stop_reason, stop_time, receipt_time, period_end, kind, value):
    if kind == "actions":
        selected = actions[:value]
        reached = len(selected) == value
        termination_in_window = not reached
    elif kind == "seconds":
        deadline = receipt_time + value
        # Event order is authoritative; timestamps have 1 ms rounding tolerance.
        selected = [r for r in actions if receipt_time - EPSILON <= r["clock_seconds"] <= deadline + EPSILON]
        reached = min(period_end, stop_time) + EPSILON >= deadline
        termination_in_window = stop_time <= deadline + EPSILON
    else:
        selected = actions
        reached = False
        termination_in_window = True
    loss = termination_in_window and stop_reason == "opponent_control"
    goal = termination_in_window and stop_reason == "goal"
    beneficiary = stop["goal_beneficiary_team_id"] if goal and stop else None
    complete = reached or loss or goal
    shots = [r for r in selected if r["event_type"] == "Shot"]
    moves = [r for r in selected if r["event_type"] in {"Pass", "Carry"} and r["completed"]]
    points = [(r["end_x"], r["end_y"]) for r in moves if valid_point(r["end_x"], r["end_y"])]
    inside = lambda x, y: valid_point(x, y) and x >= 102 and 18 <= y <= 62
    entries = [r for r in moves if not inside(r["x"], r["y"]) and inside(r["end_x"], r["end_y"])]
    thirds = [r for r in moves if valid_point(r["x"], r["y"]) and valid_point(r["end_x"], r["end_y"])
              and r["x"] < 80 <= r["end_x"]]
    last = selected[-1] if selected else None
    return {"sequence_id": sequence_id, "match_id": a["match_id"], "period": a["period"], "scope": a["scope"],
            "team_id": a["team_id"], "possession_key": a["possession_key"], "horizon_kind": kind,
            "horizon_value": value, "observed_actions": len(selected), "horizon_reached": reached,
            "observation_complete": complete, "censored": not complete,
            "termination_in_window": termination_in_window, "termination_reason": stop_reason if termination_in_window else None,
            "confirmed_loss_in_window": loss, "goal_terminal_in_window": goal,
            "goal_for_team_in_window": bool(goal and beneficiary == a["team_id"]),
            "goal_against_team_in_window": bool(goal and beneficiary == a["opponent_id"]),
            "shot_observed": bool(shots), "shot_outcome_known": bool(shots) or complete,
            "shots": len(shots), "source_xg_sum": sum(r["xg"] or 0. for r in shots),
            "shot_event_ids": ";".join(r["event_id"] for r in shots),
            "box_entry_observed": bool(entries), "box_entry_outcome_known": bool(entries) or complete,
            "box_entries": len(entries), "final_third_entries": len(thirds),
            "max_forward_gain": max([0.] + [x - a["end_x"] for x, _ in points]) if a["end_x"] is not None else None,
            "last_action_event_id": last["event_id"] if last else None,
            "last_action_type": last["event_type"] if last else None,
            "last_action_completed": last["completed"] if last else None,
            "last_action_end_x": last["end_x"] if last else None,
            "last_action_end_y": last["end_y"] if last else None,
            "action_event_ids": ";".join(r["event_id"] for r in selected),
            "receipt_after_boundary": receipt_time > stop_time + EPSILON}


def build_period_sequences(rows, home_id):
    index = PeriodIndex(rows)
    attempts, sequences, two_passes, action_links, windows = [], [], [], [], []
    for i, a in enumerate(rows):
        if a["event_type"] != "Pass":
            continue
        attempts.append({**a, **index.control_fields(i)})
        if not a["initial_pass_eligible"]:
            continue
        j, reason, inclusive, stop_time = index.boundary(i)
        stop = rows[j] if j < len(rows) else None
        end = j + int(inclusive)
        receipt, receipt_source, candidates = index.receipt(i, j)
        rt = receipt["clock_seconds"] if receipt else a["clock_seconds"] + a["duration_seconds"]
        rx, ry = (receipt["x"], receipt["y"]) if receipt and valid_point(receipt["x"], receipt["y"]) else (a["end_x"], a["end_y"])
        resolution, status, carries, dribbles, other = receiver_resolution(index, i, j, inclusive)
        own_actions = [r for r in rows[i + 1:end] if r["team_id"] == a["team_id"] and r["on_ball_action"]]
        sid = f"{a['match_id']}:{a['event_id']}"
        b_role = ({"position_id": receipt["position_id"], "position": receipt["position"], "role": receipt["role"],
                   "role_source": receipt["role_source"], "role_source_event_id": receipt["role_source_event_id"]}
                  if receipt else {k: a["recipient_" + k] for k in ["position_id", "position", "role", "role_source", "role_source_event_id"]})
        score = receipt if receipt else a
        row = {"sequence_id": sid, "match_id": a["match_id"], "a_event_id": a["event_id"], "a_index": a["index"],
               "period": a["period"], "scope": a["scope"], "possession_id": a["possession_id"],
               "possession_key": a["possession_key"], "team_id": a["team_id"], "team": a["team"],
               "opponent_id": a["opponent_id"], "opponent": a["opponent"],
               "a_id": a["player_id"], "a": a["player"], "b_id": a["recipient_id"], "b": a["recipient"],
               **{"a_" + k: a[k] for k in ["clock_seconds", "duration_seconds", "x", "y", "end_x", "end_y",
                    "position_id", "position", "role", "role_source", "role_source_event_id", "under_pressure", "pressure_flag_present",
                    "pass_length", "pass_height", "start_zone_6x3", "end_zone_6x3", "start_zone_4x3", "end_zone_4x3"]},
               "receipt_event_id": receipt["event_id"] if receipt else None, "receipt_source": receipt_source,
               "receipt_candidates": candidates, "receipt_observed": receipt is not None,
               "receipt_completed": receipt["completed"] if receipt else None,
               "receipt_clock_seconds": rt, "receipt_elapsed_seconds": a["elapsed_seconds"] + rt - a["clock_seconds"],
               "receipt_x": rx, "receipt_y": ry,
               "receipt_coordinate_source": "receipt_location" if receipt and valid_point(receipt["x"], receipt["y"]) else "pass_endpoint",
               "receipt_zone_6x3": pitch_zone(rx, ry), "receipt_zone_4x3": pitch_zone(rx, ry, 4, 3),
               **{"b_" + k: v for k, v in b_role.items()},
               "b_under_pressure_at_receipt": receipt["under_pressure"] if receipt else None,
               "b_pressure_flag_present_at_receipt": receipt["pressure_flag_present"] if receipt else None,
               "b_pressure_state": "unobserved_receipt" if receipt is None else "flagged" if receipt["under_pressure"] else "not_flagged",
               "home_score_at_receipt": score["home_score_before"], "away_score_at_receipt": score["away_score_before"],
               "score_context_source": "receipt_pre_event" if receipt else "pass_start_proxy",
               "stop_event_id": stop["event_id"] if stop else None, "stop_index": stop["index"] if stop else None,
               "stop_reason": reason, "stop_clock_seconds": stop_time, "stop_action_included": inclusive,
               "stop_possession_id": stop["possession_id"] if stop else None,
               "receipt_after_boundary": rt > stop_time + EPSILON,
               "provider_possession_team_matches": a["possession_team_id"] == a["team_id"],
               "receiver_status": status, "receiver_carries_before_release": carries,
               "receiver_dribbles_before_release": dribbles,
               "receiver_resolution_event_id": resolution["event_id"] if resolution else None,
               "receiver_resolution_type": resolution["event_type"] if resolution else None,
               "receiver_resolution_outcome": resolution["outcome"] if resolution else None,
               "receiver_resolution_clock_seconds": resolution["clock_seconds"] if resolution else None,
               "other_handler_event_id": other["event_id"] if other else None,
               "other_handler_id": other["player_id"] if other else None,
               "b_next_pass_observed": bool(resolution and resolution["event_type"] == "Pass"),
               "b_next_pass_completed": bool(resolution and resolution["event_type"] == "Pass" and resolution["completed"]),
               "future_actions_before_boundary": len(own_actions),
               "first_future_action_event_id": own_actions[0]["event_id"] if own_actions else None}
        own_score_key = "home_score_before" if a["team_id"] == home_id else "away_score_before"
        opponent_score_key = "away_score_before" if own_score_key == "home_score_before" else "home_score_before"
        row["score_difference_at_receipt"] = score[own_score_key] - score[opponent_score_key]
        row["receiver_next_control_status"] = None
        row["receiver_next_control_event_id"] = None
        row["receiver_loss_confirmed"] = None
        if resolution:
            ri = index.by_id[resolution["event_id"]]
            fields = index.control_fields(ri)
            row["receiver_next_control_status"] = fields["next_control_status"]
            row["receiver_next_control_event_id"] = fields["next_control_event_id"]
            failure = (resolution["event_type"] in {"Miscontrol", "Dispossessed"}
                       or resolution["event_type"] in {"Pass", "Dribble"} and not resolution["completed"])
            row["receiver_loss_confirmed"] = (fields["next_control_status"] == "opponent_control"
                                              if failure and fields["next_control_status"] != "censored_period_end"
                                              else None if failure else False)
        if row["b_next_pass_observed"]:
            b = resolution
            actual_c = b["recipient_id"] if b["completed"] else None
            dx = b["end_x"] - b["x"] if b["end_x"] is not None and b["x"] is not None else None
            target_eligible = (b["completed"] and b["open_play"] and actual_c is not None
                               and actual_c != a["recipient_id"] and not row["receipt_after_boundary"]
                               and row["provider_possession_team_matches"] and row["receipt_completed"] is not False)
            exclusions = []
            for condition, label in [
                (not b["completed"], "unsuccessful_next_pass"),
                (not b["open_play"], "next_pass_restart"),
                (actual_c is None, "actual_c_unobserved"),
                (actual_c == a["recipient_id"], "next_self_pass"),
                (row["receipt_after_boundary"], "receipt_after_boundary"),
                (not row["provider_possession_team_matches"], "provider_possession_team_mismatch"),
                (row["receipt_completed"] is False, "receipt_incomplete"),
            ]:
                if condition:
                    exclusions.append(label)
            noncontrol = [r for r in rows[i + 1:index.by_id[b["event_id"]]]
                          if r["team_id"] != a["team_id"] and r["on_ball_action"] and not r["controlled_touch"]]
            two_passes.append({"sequence_id": sid, "match_id": a["match_id"], "period": a["period"], "scope": a["scope"],
                "team_id": a["team_id"], "team": a["team"], "possession_id": a["possession_id"], "possession_key": a["possession_key"],
                "a_event_id": a["event_id"], "b_event_id": b["event_id"], "a_id": a["player_id"], "b_id": a["recipient_id"],
                "c_id": actual_c, "c": b["recipient"] if actual_c is not None else None,
                "b_intended_recipient_id": b["recipient_id"], "b_intended_recipient": b["recipient"],
                "b_completed": b["completed"], "b_open_play": b["open_play"], "b_outcome": b["outcome"],
                "b_clock_seconds": b["clock_seconds"], "seconds_between_pass_starts": b["clock_seconds"] - a["clock_seconds"],
                "seconds_from_receipt_to_release": b["clock_seconds"] - rt,
                "b_x": b["x"], "b_y": b["y"], "b_end_x": b["end_x"], "b_end_y": b["end_y"],
                "b_end_zone_6x3": b["end_zone_6x3"], "b_end_zone_4x3": b["end_zone_4x3"],
                "b_dx": dx, "b_direction": None if dx is None else "forward" if dx > 5 else "backward" if dx < -5 else "lateral",
                "b_under_pressure_at_release": b["under_pressure"], "b_position_id_at_release": b["position_id"],
                "c_role_at_release": b["recipient_role"], "c_role_source_event_id": b["recipient_role_source_event_id"],
                "receiver_carries_before_release": carries, "receiver_dribbles_before_release": dribbles,
                "memory_target_eligible": target_eligible,
                "memory_target_exclusion_reason": ";".join(exclusions),
                "opponent_noncontrol_annotations": len(noncontrol),
                "opponent_noncontrol_event_ids": ";".join(r["event_id"] for r in noncontrol),
                "spatial_target_eligible": bool(target_eligible and row["a_start_zone_6x3"] is not None
                                                 and row["receipt_zone_6x3"] is not None and b["end_zone_6x3"] is not None),
                "pattern": None if actual_c is None else "ABA" if actual_c == a["player_id"] else "ABC"})
        sequences.append(row)
        for step, action in enumerate(own_actions[:max(ACTION_HORIZONS)], 1):
            action_links.append({"sequence_id": sid, "match_id": a["match_id"], "period": a["period"],
                                 "possession_key": a["possession_key"], "step": step, "event_id": action["event_id"],
                                 "seconds_from_receipt": action["clock_seconds"] - rt})
        for kind, horizons in [("actions", ACTION_HORIZONS), ("seconds", TIME_HORIZONS), ("possession", [None])]:
            for horizon in horizons:
                windows.append(window_summary(a, sid, own_actions, stop, reason, stop_time, rt, index.period_end, kind, horizon))
    return {"pass_attempts": attempts, "receiver_sequences": sequences, "two_pass_sequences": two_passes,
            "sequence_actions": action_links, "future_windows": windows}


def build_match_sequences(raw, match):
    rows, players = normalise_match(raw, match)
    tables = {name: [] for name in ["pass_attempts", "receiver_sequences", "two_pass_sequences", "sequence_actions", "future_windows"]}
    for period in sorted({r["period"] for r in rows}):
        selected = [r for r in rows if r["period"] == period]
        for name, values in build_period_sequences(selected, match["home_team"]["home_team_id"]).items():
            tables[name].extend(values)
    tables["events"] = rows
    tables["players"] = players
    return tables


def dataframe(rows):
    frame = pd.DataFrame(rows)
    for column in frame:
        if (column == "index" or column.endswith("_index") or column in {"step", "horizon_value"}
            or column.endswith("_id") and column not in {"event_id", "sequence_id"} and not column.endswith("_event_id")
            or "zone_" in column or column.endswith("_candidates")):
            frame[column] = frame[column].astype("Int64")
        elif frame[column].dropna().map(lambda x: isinstance(x, (bool, np.bool_))).all() and frame[column].notna().any():
            frame[column] = frame[column].astype("boolean")
    return frame


def zone_definitions():
    return pd.DataFrame([{"grid": f"{columns}x{rows}", "zone_id": x + columns * y, "column": x, "row": y,
                          "x_min": x * 120 / columns, "x_max": (x + 1) * 120 / columns,
                          "y_min": y * 80 / rows, "y_max": (y + 1) * 80 / rows}
                         for columns, rows in [(6, 3), (4, 3)] for y in range(rows) for x in range(columns)])
