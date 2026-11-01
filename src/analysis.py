"""Load local StatsBomb JSON, build directed weighted graphs, and draw results.

The notebook and command-line runner use these same functions. No network access
is made by this module. Player IDs, rather than display names, identify nodes.
"""
from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import Normalize
import networkx as nx
import numpy as np
import pandas as pd

MATCH_IDS = [3942226, 3942752, 3943043]
SOURCE_COMMIT = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
SOURCE_URL = "https://github.com/hudl/open-data"
RESTARTS = {"Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in"}
COLORS = {"Spain": "#cc5148", "Germany": "#3b586d", "France": "#487cb4", "England": "#7c6aa6"}
INK = "#20364c"
ROOT = Path(__file__).resolve().parents[1]
LOGO = ROOT / "assets" / "statsbomb-logo.png"


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "figure.facecolor": "#fafbfd", "axes.facecolor": "#fafbfd",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#ced7df", "grid.color": "#e2e7ed",
        "savefig.facecolor": "#fafbfd", "figure.dpi": 110,
    })


def load_dataset(root: Path = ROOT) -> dict:
    """Flatten passes with pandas; keep the raw records and exclusion flags."""
    base = root / "data" / "open-data"
    all_matches = json.loads((base / "data/matches/55/282.json").read_text(encoding="utf-8"))
    matches = pd.DataFrame([
        {"match_id": m["match_id"], "date": m["match_date"],
         "home": m["home_team"]["home_team_name"], "away": m["away_team"]["away_team_name"],
         "home_score": m["home_score"], "away_score": m["away_score"],
         "stage": m["competition_stage"]["name"]}
        for m in all_matches if m["match_id"] in MATCH_IDS
    ]).sort_values("date").reset_index(drop=True)
    frames, players, events = [], {}, {}
    for match in matches.itertuples():
        mid = match.match_id
        events[mid] = json.loads((base / f"data/events/{mid}.json").read_text(encoding="utf-8"))
        lineups = json.loads((base / f"data/lineups/{mid}.json").read_text(encoding="utf-8"))
        for team in lineups:
            for player in team["lineup"]:
                players[(mid, player["player_id"])] = {
                    "name": player.get("player_nickname") or player["player_name"],
                    "full_name": player["player_name"], "jersey": player["jersey_number"],
                    "team": team["team_name"],
                }
        flat = pd.json_normalize(events[mid], sep=".")
        frame = flat.loc[flat["type.name"].eq("Pass")].copy()
        for column in ["pass.outcome.name", "pass.type.name", "pass.recipient.id",
                       "pass.recipient.name", "under_pressure"]:
            if column not in frame:
                frame[column] = np.nan
        frame = frame.rename(columns={
            "id": "event_id", "team.name": "team", "player.id": "passer_id",
            "player.name": "passer_name", "pass.recipient.id": "recipient_id",
            "pass.recipient.name": "recipient_name", "pass.outcome.name": "outcome",
            "pass.type.name": "pass_type", "pass.length": "length",
        })
        frame["match_id"] = mid
        frame["passer_id"] = frame["passer_id"].astype("Int64")
        frame["recipient_id"] = frame["recipient_id"].astype("Int64")
        frame["regulation"] = frame["period"].isin([1, 2])
        frame["open_play"] = ~frame["pass_type"].isin(RESTARTS)
        frame["completed"] = frame["outcome"].isna()
        frame["has_recipient"] = frame["recipient_id"].notna()
        frame["self_pass"] = frame["passer_id"].eq(frame["recipient_id"]).fillna(False)
        frame["network_pass"] = (frame["regulation"] & frame["open_play"] &
                                 frame["completed"] & frame["has_recipient"] & ~frame["self_pass"])
        frame["x"] = frame["location"].map(lambda v: v[0] if isinstance(v, list) else np.nan)
        frame["y"] = frame["location"].map(lambda v: v[1] if isinstance(v, list) else np.nan)
        frame["end_x"] = frame["pass.end_location"].map(lambda v: v[0] if isinstance(v, list) else np.nan)
        frame["end_y"] = frame["pass.end_location"].map(lambda v: v[1] if isinstance(v, list) else np.nan)
        frame["under_pressure"] = frame["under_pressure"].fillna(False).astype(bool)
        frames.append(frame)
    passes = pd.concat(frames, ignore_index=True)
    return {"matches": matches, "passes": passes, "players": players, "events": events}


def build_graph(passes: pd.DataFrame, players: dict, mid: int, team: str) -> nx.DiGraph:
    """An edge i->j has weight equal to the count of completed eligible passes."""
    graph = nx.DiGraph(match_id=int(mid), team=team, scope="regulation_open_play_completed")
    eligible = passes.loc[passes["network_pass"] & passes["match_id"].eq(mid) & passes["team"].eq(team)]
    # Include everyone who attempted or was named as recipient of an open-play
    # regulation pass, including a player with no completed network involvement.
    attempts = passes.loc[passes["regulation"] & passes["open_play"] &
                          passes["match_id"].eq(mid) & passes["team"].eq(team)]
    node_ids = set(attempts["passer_id"].dropna().astype(int)) | set(attempts["recipient_id"].dropna().astype(int))
    for pid in sorted(node_ids):
        metadata = players[(mid, pid)]
        if metadata["team"] != team:
            raise ValueError(f"Cross-team recipient: {pid} in {mid}")
        graph.add_node(pid, **metadata)
    weights = eligible.groupby(["passer_id", "recipient_id"]).size()
    for (source, target), weight in weights.items():
        graph.add_edge(int(source), int(target), weight=int(weight), distance=1.0 / weight)
    # Average locations combine completed pass origins and reception endpoints.
    # These are ball-event positions, not tracking-derived player positions.
    touches = pd.concat([
        eligible[["passer_id", "x", "y"]].rename(columns={"passer_id": "player_id"}),
        eligible[["recipient_id", "end_x", "end_y"]].rename(columns={"recipient_id": "player_id", "end_x": "x", "end_y": "y"}),
    ])
    if not touches.empty:
        for pid, row in touches.groupby("player_id")[["x", "y"]].mean().iterrows():
            graph.nodes[int(pid)].update(x=float(row.x), y=float(row.y))
    return graph


def compute_tables(dataset: dict) -> dict:
    passes, players = dataset["passes"], dataset["players"]
    graphs, team_rows, player_rows, edge_rows, audit_rows = {}, [], [], [], []
    for match in dataset["matches"].itertuples():
        mid = match.match_id
        for team in [match.home, match.away]:
            graph = build_graph(passes, players, mid, team)
            graphs[(mid, team)] = graph
            raw = passes.loc[passes.match_id.eq(mid) & passes.team.eq(team)]
            regulation = raw.loc[raw.regulation]
            attempts = regulation.loc[regulation.open_play]
            completed = attempts.loc[attempts.completed]
            eligible = raw.loc[raw.network_pass]
            pagerank = nx.pagerank(graph, alpha=0.85, weight="weight")
            between = nx.betweenness_centrality(graph, weight="distance", normalized=True)
            n, e = len(graph), graph.number_of_edges()
            strengths = dict(graph.degree(weight="weight"))
            total = sum(strengths.values())
            team_rows.append({
                "match_id": mid, "date": match.date, "team": team,
                "opponent": match.away if team == match.home else match.home,
                "stage": match.stage, "nodes": n, "directed_edges": e,
                "open_play_attempts": len(attempts), "completed_open_play": len(completed),
                "network_passes": len(eligible),
                "completion_pct": 100 * len(completed) / len(attempts),
                "density": nx.density(graph), "reciprocity": nx.reciprocity(graph) or 0.0,
                "largest_strong_component_pct": 100 * max(map(len, nx.strongly_connected_components(graph)), default=0) / n,
                "top3_involvement_pct": 100 * sum(sorted(strengths.values(), reverse=True)[:3]) / total,
                "mean_completed_pass_length": float(eligible.length.mean()),
            })
            for pid in graph:
                attrs = graph.nodes[pid]
                player_rows.append({
                    "match_id": mid, "team": team, "player_id": pid,
                    "name": attrs["name"], "jersey": attrs["jersey"],
                    "passes_sent": graph.out_degree(pid, weight="weight"),
                    "passes_received": graph.in_degree(pid, weight="weight"),
                    "involvement": strengths[pid],
                    "involvement_pct": 100 * strengths[pid] / total,
                    "pagerank": pagerank[pid], "betweenness": between[pid],
                    "x": attrs.get("x", np.nan), "y": attrs.get("y", np.nan),
                })
            for source, target, attrs in graph.edges(data=True):
                edge_rows.append({"match_id": mid, "team": team, "passer_id": source,
                                  "recipient_id": target, "passer": graph.nodes[source]["name"],
                                  "recipient": graph.nodes[target]["name"], **attrs})
            audit_rows.append({
                "match_id": mid, "team": team, "all_period_passes": len(raw),
                "excluded_extra_time_or_shootout": len(raw) - len(regulation),
                "regulation_passes": len(regulation),
                "excluded_restarts": len(regulation) - len(attempts),
                "open_play_attempts": len(attempts),
                "unsuccessful_open_play": int((~attempts.completed).sum()),
                "completed_without_recipient": int((completed.recipient_id.isna()).sum()),
                "completed_self_passes": int(completed.self_pass.sum()),
                "network_passes": len(eligible),
            })
    return {"graphs": graphs, "teams": pd.DataFrame(team_rows),
            "players": pd.DataFrame(player_rows), "edges": pd.DataFrame(edge_rows),
            "audit": pd.DataFrame(audit_rows)}


def validate_analysis(dataset: dict, tables: dict) -> dict:
    """Check pass accounting and graph invariants against the original events."""
    passes = dataset["passes"]
    assert not passes.duplicated(["match_id", "event_id"]).any(), "Duplicate event IDs"
    selected = passes.loc[passes.network_pass]
    assert selected.period.isin([1, 2]).all()
    assert selected.outcome.isna().all()
    assert selected.recipient_id.notna().all()
    assert not selected.pass_type.isin(RESTARTS).any()
    assert selected[["x", "y", "end_x", "end_y"]].notna().all().all()
    assert selected.x.between(0, 120).all() and selected.end_x.between(0, 120).all()
    assert selected.y.between(0, 80).all() and selected.end_y.between(0, 80).all()
    for key, graph in tables["graphs"].items():
        mid, team = key
        expected = selected.loc[selected.match_id.eq(mid) & selected.team.eq(team)]
        assert sum(nx.get_edge_attributes(graph, "weight").values()) == len(expected)
        assert sum(dict(graph.out_degree(weight="weight")).values()) == len(expected)
        assert sum(dict(graph.in_degree(weight="weight")).values()) == len(expected)
        assert nx.number_of_selfloops(graph) == 0
        expected_weights = expected.groupby(["passer_id", "recipient_id"]).size().to_dict()
        assert {(u, v): a["weight"] for u, v, a in graph.edges(data=True)} == expected_weights
        prs = tables["players"].loc[lambda x: x.match_id.eq(mid) & x.team.eq(team), "pagerank"]
        assert np.isclose(prs.sum(), 1.0)
    for row in tables["audit"].itertuples():
        assert row.all_period_passes == row.regulation_passes + row.excluded_extra_time_or_shootout
        assert row.regulation_passes == row.excluded_restarts + row.open_play_attempts
        assert row.open_play_attempts == (row.unsuccessful_open_play + row.completed_without_recipient +
                                          row.completed_self_passes + row.network_passes)
    return {"status": "passed", "matches": len(dataset["matches"]),
            "team_networks": len(tables["graphs"]), "network_passes": len(selected),
            "checks": ["unique event IDs", "period/outcome/restart filters", "valid coordinates",
                       "team membership", "exact edge weights", "in/out-strength totals",
                       "no self loops", "PageRank normalization", "exclusion accounting"]}


def match_label(match) -> str:
    suffix = " (a.e.t.; analysis: periods 1-2)" if match.match_id == 3942226 else ""
    return f"{match.home} {match.home_score}-{match.away_score} {match.away}{suffix}"


def pitch(ax):
    ax.set_facecolor("#f2f6f3")
    line = "#b8c9bd"
    for rect in [(0, 0, 120, 80), (0, 18, 18, 44), (102, 18, 18, 44),
                 (0, 30, 6, 20), (114, 30, 6, 20)]:
        ax.add_patch(patches.Rectangle(rect[:2], rect[2], rect[3], fill=False, ec=line, lw=0.9))
    ax.plot([60, 60], [0, 80], color=line, lw=0.9)
    ax.add_patch(patches.Circle((60, 40), 9.15, fill=False, ec=line, lw=0.9))
    ax.scatter([12, 108], [40, 40], s=4, color=line)
    ax.set(xlim=(-5, 125), ylim=(88, -6), aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.annotate("ATTACK", xy=(118, 84), xytext=(93, 84), fontsize=7, va="center",
                color="#718778", arrowprops={"arrowstyle": "->", "color": "#718778"})


def network_plot(ax, graph, title, max_weight, names=False, min_passes=1):
    pitch(ax)
    positions = {pid: (attrs["x"], attrs["y"]) for pid, attrs in graph.nodes(data=True) if "x" in attrs}
    # The threshold is visual only. All eligible edges enter every metric.
    visible = [(u, v) for u, v, attrs in graph.edges(data=True)
               if attrs["weight"] >= min_passes and u in positions and v in positions]
    if visible:
        nx.draw_networkx_edges(
            graph, positions, ax=ax, edgelist=visible,
            width=[0.30 * graph[u][v]["weight"] for u, v in visible],
            edge_color=COLORS[graph.graph["team"]], alpha=0.43,
            arrows=True, arrowstyle="-|>", arrowsize=10,
            connectionstyle="arc3,rad=0.14", min_source_margin=9, min_target_margin=11,
            node_size=[65 + 4.0 * graph.degree(pid, weight="weight") for pid in graph],
        )
    for pid, (x, y) in positions.items():
        involvement = graph.degree(pid, weight="weight")
        ax.scatter(x, y, s=65 + 4.0 * involvement, c=COLORS[graph.graph["team"]],
                   ec="white", lw=1.3, zorder=3)
        ax.text(x, y, str(graph.nodes[pid]["jersey"]), ha="center", va="center",
                fontsize=8, color="white", weight="bold", zorder=4)
        if names:
            label = ax.annotate(graph.nodes[pid]["name"], (x, y), xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=7, zorder=5,
                        bbox={"facecolor": "#fafbfd", "alpha": .86, "edgecolor": "none", "pad": 1})
            label.set_gid("player-label")
    ax.set_title(title, fontsize=11, weight="bold", loc="left", pad=5)


def save_figure(fig, path, title, note, footer=None):
    fig.suptitle(title, x=0.04, ha="left", fontsize=19, weight="bold", y=.985)
    extra_lines = max(0, note.count("\n") - 1)
    fig.tight_layout(rect=(.02, .10 + .025*extra_lines, .99, .945), h_pad=2.0, w_pad=2.0)
    # Resolve clustered name labels after layout, preserving actual node positions.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        labels = [text for text in ax.texts if text.get_gid() == "player-label"]
        for _ in range(60):
            changed = False
            for i, first in enumerate(labels):
                first_box = first.get_window_extent(renderer).expanded(1.04, 1.25)
                for second in labels[i+1:]:
                    if first_box.overlaps(second.get_window_extent(renderer)):
                        xoff, yoff = second.get_position()
                        second.set_position((xoff, yoff + 4))
                        changed = True
            if not changed:
                break
    fig.text(.04, .064 + .023*extra_lines, note, fontsize=8, color="#52677b", va="top")
    fig.text(.04, .022, footer or "Data: StatsBomb Open Data  |  EURO 2024  |  Regulation periods, including stoppage time",
             fontsize=8, color="#63788a")
    if LOGO.exists():
        logo_ax = fig.add_axes([.84, .012, .125, .035])
        logo_ax.imshow(plt.imread(LOGO)); logo_ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=170)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return path.with_suffix(".png")


def create_figures(dataset: dict, tables: dict, output_dir: Path) -> list[dict]:
    configure_style()
    graphs, matches = tables["graphs"], list(dataset["matches"].itertuples())
    max_weight = max(a["weight"] for graph in graphs.values() for _, _, a in graph.edges(data=True))
    figures = []
    def save(fig, stem, title, note):
        path = save_figure(fig, output_dir / stem, title, note)
        figures.append({"file": path.name, "title": title, "note": note})
    network_note = ("Arrows: passer to recipient; every link with 1+ passes is drawn. Line width = 0.30 pt per pass.\n"
                    "Node area: passes sent + received (common scale). Numbers: shirts. Positions: mean pass/reception locations.")
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    for i, match in enumerate(matches):
        for j, team in enumerate([match.home, match.away]):
            g = graphs[(match.match_id, team)]
            network_plot(axes[i, j], g, f"{team} vs {match.away if team == match.home else match.home} | {match.stage}", max_weight)
    save(fig, "01_network_overview", "Six passing networks, three knockout matches", network_note)
    for i, match in enumerate(matches):
        fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
        for ax, team in zip(axes, [match.home, match.away]):
            g = graphs[(match.match_id, team)]
            network_plot(ax, g, team, max_weight, names=True)
        save(fig, f"{i+2:02d}_network_{match.away.lower()}", match_label(match), network_note)

    # Directed adjacency matrices share one color scale across all six teams.
    for i, match in enumerate(matches):
        fig, axes = plt.subplots(1, 2, figsize=(17, 10))
        for ax, team in zip(axes, [match.home, match.away]):
            g = graphs[(match.match_id, team)]
            nodes = sorted(g, key=lambda pid: (-g.degree(pid, weight="weight"), pid))
            matrix = nx.to_numpy_array(g, nodelist=nodes, weight="weight")
            im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max_weight)
            labels = [g.nodes[pid]["name"] for pid in nodes]
            ax.set_xticks(range(len(nodes)), labels, rotation=65, ha="right", fontsize=8)
            ax.set_yticks(range(len(nodes)), labels, fontsize=8)
            ax.set_title(team, loc="left", weight="bold")
            ax.set_xlabel("Recipient"); ax.set_ylabel("Passer")
            for row, col in zip(*np.where(matrix > 0)):
                ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=7,
                        color="white" if matrix[row, col] > max_weight * .5 else INK)
            fig.colorbar(im, ax=ax, fraction=.035, pad=.025, label="Completed passes")
        save(fig, f"{i+5:02d}_adjacency_{match.away.lower()}", f"Who passes to whom? | Spain vs {match.away}",
             "Rows send; columns receive. Players sorted by total involvement. All links included; common color scale.")

    max_involvement = tables["players"].involvement.max()
    for i, match in enumerate(matches):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        for ax, team in zip(axes, [match.home, match.away]):
            subset = tables["players"].loc[lambda x: x.match_id.eq(match.match_id) & x.team.eq(team)]
            top = subset.nlargest(8, "involvement").sort_values("involvement")
            y = np.arange(len(top))
            ax.barh(y, top.passes_sent, color=COLORS[team], label="Sent")
            ax.barh(y, top.passes_received, left=top.passes_sent, color=COLORS[team], alpha=.35, label="Received")
            ax.set_yticks(y, top.name, fontsize=10)
            ax.set_xlim(0, max_involvement * 1.26)
            ax.set_xlabel("Completed pass involvement (sent + received)")
            ax.set_title(team, loc="left", weight="bold")
            for yy, row in zip(y, top.itertuples()):
                ax.text(row.involvement + 3, yy, f"{row.involvement} | PR {row.pagerank:.1%}", va="center", fontsize=8)
            ax.legend(loc="lower right", frameon=False)
        save(fig, f"{i+8:02d}_central_players_{match.away.lower()}", f"Central players | Spain vs {match.away}",
             "Top eight by weighted degree. PR: weighted PageRank (incoming prominence), damping 0.85. Common horizontal scale.\n"
             "Full regulation totals include substitutes and are not adjusted for minutes played.")

    summary = tables["teams"]
    labels = [f"{r.team}\nvs {r.opponent}" for r in summary.itertuples()]
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    metrics = [("network_passes", "Completed network passes", False),
               ("completion_pct", "Open-play completion (%)", False),
               ("density", "Directed network density", True),
               ("reciprocity", "Reciprocity of observed links", True),
               ("top3_involvement_pct", "Top three players' involvement share (%)", False),
               ("mean_completed_pass_length", "Mean completed pass length (StatsBomb units)", False)]
    for ax, (column, title, fraction) in zip(axes.flat, metrics):
        values = summary[column].to_numpy()
        ax.bar(np.arange(6), values, color=[COLORS[t] for t in summary.team], width=.65)
        ax.set_xticks(np.arange(6), labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.set_ylim(0, 1.08 if fraction else max(values) * 1.2)
        ax.grid(axis="y", alpha=.6); ax.set_axisbelow(True)
        for j, value in enumerate(values):
            ax.text(j, value + (0.025 if fraction else max(values)*.025),
                    f"{value:.2f}" if fraction else f"{value:.0f}" if column == "network_passes" else f"{value:.1f}",
                    ha="center", fontsize=8)
    save(fig, "11_team_comparison", "How do the six team networks compare?",
         "Density = observed directed links / n(n-1); reciprocity = share of links with a reverse link.\n"
         "Different player counts and minutes affect these full-match totals; descriptive comparisons only.")

    # Compare the same Spain player IDs across games; absent players stay blank.
    spain = tables["players"].loc[tables["players"].team.eq("Spain")]
    selected_ids = spain.groupby("player_id").involvement.sum().nlargest(11).index
    names = spain.drop_duplicates("player_id").set_index("player_id")["name"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 9))
    for ax, column, title in zip(axes, ["involvement_pct", "pagerank", "betweenness"],
                                 ["Share of pass involvement (%)", "Weighted PageRank (%)", "Weighted betweenness"]):
        matrix = spain.pivot(index="player_id", columns="match_id", values=column).reindex(index=selected_ids, columns=MATCH_IDS)
        if column == "pagerank": matrix = matrix * 100
        values = np.ma.masked_invalid(matrix.to_numpy())
        im = ax.imshow(values, cmap="YlOrRd", aspect="auto", vmin=0)
        ax.set_yticks(range(len(selected_ids)), names.reindex(selected_ids), fontsize=9)
        ax.set_xticks(range(3), ["Germany", "France", "England"])
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        for row in range(len(selected_ids)):
            for col in range(3):
                value = matrix.iloc[row, col]
                if pd.notna(value):
                    ax.text(col, row, f"{value:.2f}" if column == "betweenness" else f"{value:.1f}",
                            ha="center", va="center", fontsize=9,
                            color="white" if value > float(np.nanmax(matrix.to_numpy()))*.6 else INK)
        fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
    save(fig, "12_spain_player_evolution", "Spain's central players across three matches",
         "Same 11 players, selected by total involvement across all matches; blanks mean absent from that network.\n"
         "Betweenness uses inverse pass frequency as edge distance. Full-match values are not adjusted for playing time.")

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    for ax, match in zip(axes, matches):
        for team in [match.home, match.away]:
            subset = dataset["passes"].loc[lambda x: x.match_id.eq(match.match_id) & x.team.eq(team) & x.network_pass].copy()
            # Put each half's added time in that half's final five-minute bin.
            clock = subset.minute.to_numpy() + subset.second.to_numpy() / 60
            subset["bin"] = np.where(subset.period.eq(1), np.minimum(clock,44.999), np.minimum(np.maximum(clock,45),89.999)) // 5
            counts = subset.groupby("bin").size().reindex(range(18), fill_value=0)
            ax.plot(np.arange(18)*5+2.5, counts, marker="o", lw=2, ms=4, label=team, color=COLORS[team])
        ax.axvline(45, color="#aebdca", ls="--", lw=1)
        ax.set_title(f"Spain vs {match.away}", loc="left", weight="bold")
        ax.set_ylabel("Completed passes / bin"); ax.grid(axis="y", alpha=.6)
        ax.legend(frameon=False, ncol=2, loc="upper right")
    axes[-1].set(xlabel="Regulation match minute", xlim=(0,90), xticks=range(0,91,15))
    save(fig, "13_pass_timeline", "Passing volume over the match",
         "Five-minute bins. Added time is pooled into the final bin of each half, so those bins cover longer durations.\n"
         "The lines show counts, not a possession estimate or a time-normalized rate.")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for col, match in enumerate(matches):
        for row, period in enumerate([1,2]):
            half_passes = dataset["passes"].loc[dataset["passes"].period.eq(period)]
            g = build_graph(half_passes, dataset["players"], match.match_id, "Spain")
            count = sum(a["weight"] for _,_,a in g.edges(data=True))
            network_plot(axes[row,col], g, f"vs {match.away} | Half {period} | {count} passes", max_weight)
    save(fig, "14_spain_by_half", "Spain: first half versus second half", network_note)

    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    categories = ["Backward", "Lateral", "Forward"]
    for ax, match in zip(axes, matches):
        for j, team in enumerate([match.home, match.away]):
            subset = dataset["passes"].loc[lambda x: x.match_id.eq(match.match_id) & x.team.eq(team) & x.network_pass]
            dx = subset.end_x - subset.x
            counts = pd.Series(np.where(dx > 5, "Forward", np.where(dx < -5, "Backward", "Lateral"))).value_counts()
            pct = counts.reindex(categories, fill_value=0).to_numpy() / len(subset) * 100
            x = np.arange(3) + (j-.5)*.36
            ax.bar(x, pct, width=.34, color=COLORS[team], label=team)
            for xx, value in zip(x,pct): ax.text(xx,value+1,f"{value:.0f}%",ha="center",fontsize=9)
        ax.set_xticks(range(3),categories); ax.set_ylim(0,75)
        ax.set_title(f"Spain vs {match.away}",loc="left",weight="bold")
        ax.set_ylabel("Share of completed open-play passes (%)")
        ax.legend(frameon=False,loc="upper right",fontsize=9)
    save(fig, "15_pass_direction", "Passing direction: progression or recycling?",
         "Forward: end_x - start_x > 5; backward: < -5; lateral: within +/-5 StatsBomb x units.\n"
         "Team-relative coordinates face right. Forward passes are not the same as progressive passes or line-breaking passes.")
    (output_dir / "figure_index.json").write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding="utf-8")
    return figures


def write_findings(dataset: dict, tables: dict, output_dir: Path):
    rows = ["# Football Passing Networks — Findings", "",
            "Data: **StatsBomb Open Data**, via [hudl/open-data](https://github.com/hudl/open-data).", "",
            "Regulation includes stoppage time. Spain–Germany extra time is excluded: the final was 2–1 after extra time, with 1–1 after regulation.", "",
            "Networks include completed passes with a named recipient, excluding direct corners, free kicks, throw-ins, goal kicks and kick-offs.", ""]
    for match in dataset["matches"].itertuples():
        teams = tables["teams"].loc[tables["teams"].match_id.eq(match.match_id)].set_index("team")
        spain, opponent = teams.loc["Spain"], teams.loc[match.away]
        top = tables["players"].loc[lambda x: x.match_id.eq(match.match_id) & x.team.eq("Spain")].nlargest(3,"involvement")
        pair = tables["edges"].loc[lambda x: x.match_id.eq(match.match_id) & x.team.eq("Spain")].nlargest(1,"weight").iloc[0]
        rows.extend([
            f"## Spain – {match.away} ({match.date})", "",
            f"- Network passes: **{int(spain.network_passes)}** for Spain, **{int(opponent.network_passes)}** for the opponent.",
            f"- Open-play completion: **{spain.completion_pct:.1f}%** versus **{opponent.completion_pct:.1f}%**.",
            f"- Spain density: **{spain.density:.3f}**; reciprocity: **{spain.reciprocity:.3f}**.",
            "- Highest involvement (sent + received): " + ", ".join(f"**{r.name} ({r.involvement})**" for r in top.itertuples()) + ".",
            f"- Strongest directed link: **{pair.passer} → {pair.recipient}**, **{int(pair.weight)}** passes.",
            f"- Top-three involvement share: **{spain.top3_involvement_pct:.1f}%**. This describes concentration of circulation.", ""])
    rows.extend(["## Interpretation limits", "",
        "- Density and reciprocity describe observed links, not team quality. Substitutions change the node count and can mechanically lower density.",
        "- Totals are not adjusted for playing time. Each pass contributes once to the sender and once to the recipient.",
        "- PageRank measures incoming prominence. Betweenness uses inverse pass count as distance; aggregate shortest paths are not observed possession chains.",
        "- Locations summarize pass/reception events, not tracking or formations. Full-match players need not have shared the pitch.",
        "- Three selected matches support a descriptive case study, not causal attribution or generalization.", ""])
    text = "\n".join(rows)
    (output_dir / "RESULTS.md").write_text(text,encoding="utf-8")
    return text


def export_tables(dataset: dict, tables: dict, root: Path, validation: dict):
    out = root / "results"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "graphs").mkdir(exist_ok=True)
    dataset["matches"].to_csv(out / "tables/matches.csv", index=False)
    for name in ["teams", "players", "edges", "audit"]:
        tables[name].to_csv(out / f"tables/{name}.csv", index=False)
    columns = ["match_id", "event_id", "index", "team", "period", "minute", "second", "passer_id",
               "recipient_id", "passer_name", "recipient_name", "x", "y", "end_x", "end_y", "length", "under_pressure"]
    dataset["passes"].loc[dataset["passes"].network_pass, columns].to_csv(out / "tables/completed_passes.csv",index=False)
    for (mid,team), graph in tables["graphs"].items():
        nx.write_graphml(graph, out / f"graphs/{mid}_{team.lower()}.graphml")
    (out / "validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    packages = {name:importlib.metadata.version(name) for name in ["pandas","numpy","networkx","matplotlib","scipy","nbformat","nbclient","ipykernel"]}
    (out / "environment.json").write_text(json.dumps(packages,indent=2),encoding="utf-8")
    manifest_path = root / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        assert hashlib.sha256((root/"data/open-data"/relative).read_bytes()).hexdigest() == expected, f"Source changed: {relative}"


def write_gallery(root: Path, figures: list[dict], tables: dict):
    """Write a local, responsive results document with relative asset links."""
    cards = "\n".join(
        f'<article><h2>{i:02d}. {html.escape(item["title"])}</h2>'
        f'<a href="figures/{item["file"]}"><img loading="lazy" src="figures/{item["file"]}" alt="{html.escape(item["title"])}"></a>'
        f'<p>{html.escape(item["note"])}</p><a href="figures/{Path(item["file"]).stem}.svg">SVG</a></article>'
        for i,item in enumerate(figures,1))
    summary = tables["teams"][["team","opponent","network_passes","completion_pct","density","reciprocity"]].copy()
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Football Passing Networks — Results</title><style>
body{{margin:0;background:#f1f5f8;color:#20364c;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1250px;margin:auto;padding:35px 22px}}
h1{{font-size:clamp(30px,5vw,52px);line-height:1.12;margin-bottom:12px}}h2{{font-size:22px}}article,section{{background:white;border-radius:14px;padding:24px;margin:24px 0;box-shadow:0 3px 20px #20364c0a}}
img{{width:100%;height:auto}}.brand{{width:185px}}a{{color:#27628d}}p{{max-width:1000px}}table{{border-collapse:collapse;font-size:14px;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #dce4eb}}.scroll{{overflow:auto}}
</style><main>

<img class="brand" src="../assets/statsbomb-logo.png" alt="StatsBomb"><h1>Football Passing Networks</h1>
<p>Spain against Germany, France and England · EURO 2024 · three matches, six directed networks, 15 figures.</p>
<p>Nodes identify players; arrows identify sender → recipient. Edge weights count completed open-play passes in regulation, including stoppage time. Extra time is excluded from these aggregate networks.</p>
<p><a href="index.html">All research results</a> · <a href="fixed_lineups/report.html">Stable lineups: every link from one pass</a> · <a href="RESULTS.md">Findings and limitations</a> · <a href="../notebooks/football_passing_networks.ipynb">Executed notebook</a></p>
<section><h2>Network summary</h2><div class="scroll">{summary.to_html(index=False,float_format=lambda x:f"{x:.3f}",border=0)}</div>
<p>Centrality and density depend on substitutions and playing time. Locations summarize ball events and are not tracking positions or measured formations.</p></section>
{cards}<p>Data: <a href="https://github.com/hudl/open-data">StatsBomb Open Data</a>. Source commit: {SOURCE_COMMIT}.</p>
</main></html>'''
    (root/"results/report.html").write_text(body,encoding="utf-8")


def run_analysis(root: Path = ROOT) -> tuple[dict, dict, list[dict]]:
    dataset = load_dataset(root)
    tables = compute_tables(dataset)
    validation = validate_analysis(dataset,tables)
    export_tables(dataset,tables,root,validation)
    figures = create_figures(dataset,tables,root/"results/figures")
    write_findings(dataset,tables,root/"results")
    write_gallery(root,figures,tables)
    from .lineups import run_fixed_lineups
    run_fixed_lineups(root)
    return dataset,tables,figures
