"""Build an English read-only preview of the case-study schemas and source JSON."""
from pathlib import Path
import html
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = dict(
    match_id='Source match ID.', event_id='Unique source-event UUID.',
    index='Event order within the match, across all event types.',
    team='Player/pass team.', period='1–2 regulation; 3–4 extra time.',
    minute='Provider match minute.', second='Second within the recorded minute.',
    passer_id='Sender player ID.', recipient_id='Recipient player ID.',
    passer_name='Source sender name.', recipient_name='Source recipient name.',
    x='Horizontal pass origin or player-node location.',
    y='Vertical pass origin or player-node location.',
    end_x='Horizontal pass endpoint.', end_y='Vertical pass endpoint.',
    length='StatsBomb pass.length in provider coordinate units.',
    under_pressure='Provider pressure flag; unflagged events are False in this cleaned table.',
    date='Match date.', home='Home team.', away='Away team.',
    home_score='Final home score, including extra time.',
    away_score='Final away score, including extra time.', opponent='Opponent.', stage='Competition stage.',
    nodes='Aggregate graph node count.', directed_edges='Distinct directed links.',
    open_play_attempts='All regulation open-play attempts.',
    completed_open_play='Completed open-play passes before recipient/self-pass checks.',
    network_passes='Completed passes included in this network.',
    completion_pct='Open-play completion percentage.', density='E / [N(N−1)].',
    reciprocity='Share of directed links with an observed reverse link.',
    largest_strong_component_pct='Share of nodes in the largest strongly connected component.',
    top3_involvement_pct='Top-three share of total team involvement.',
    mean_completed_pass_length='Mean source length of network passes.',
    player_id='Player ID.', name='Display name.', full_name='Full lineup name.', jersey='Shirt number.',
    passes_sent='Completed passes sent.', passes_received='Completed passes received.',
    involvement='passes_sent + passes_received.', involvement_pct='Share of team involvement (%).',
    pagerank='Weighted PageRank; sums to one per graph.',
    betweenness='Normalized directed betweenness using distance 1/weight.',
    passer='Sender display name.', recipient='Recipient display name.',
    weight='Completed passes from this sender to this recipient.',
    distance='Inverse weight for shortest-path calculations.',
    segment_id='Stable-lineup interval ID; join key.', scope='regulation or extra_time.',
    number='Interval number within match/team/scope.',
    start_index='Included first event index.', end_index='Excluded upper event-index boundary.',
    start_label='Start time with period reference.', end_label='End time with period reference.',
    duration_seconds='Duration including stoppage time and excluding breaks.',
    reason='Membership change causing the interval to start.', players='Roster player count.',
    links='Distinct directed links in the interval.',
    zero_involvement='Roster players without completed-pass involvement.',
    roster_ids='Semicolon-separated roster player IDs.', roster_names='Roster names.',
    change_event_ids='Source UUIDs of membership-change events.',
    position_source='Completed-pass locations, other interval events, or schematic tactical reference.',
    all_period_passes='All source pass attempts including extra time.',
    excluded_extra_time_or_shootout='Attempts outside periods 1–2.',
    regulation_passes='All attempts in periods 1–2.', excluded_restarts='Excluded direct restarts.',
    unsuccessful_open_play='Unsuccessful open-play attempts.',
    completed_without_recipient='Completed passes without a named recipient.',
    completed_self_passes='Excluded completed passes to the sender themself.',
    temporary_absence_seconds='Total player-seconds temporarily absent within the nominal interval.',
    display_eligible='Meets the declared duration and pass-count thresholds for a main figure.',
    display_exclusion_reason='Reason a nominal interval is retained only in the audit.',
    active_audit_segment_ids='Exact-active interval IDs intersecting this nominal interval.',
    absent_player_ids='Semicolon-separated temporarily absent player IDs.',
    absent_player_names='Names of temporarily absent nominal-roster players.')

def compact(value):
    if isinstance(value, (dict, list)): value = json.dumps(value, ensure_ascii=False)
    return str(value)[:180]

def main():
    sections, overview = [], []
    md = ['# Case-study data dictionary', '',
          'Original inputs are nested event JSON. The following CSV files are derived analytical tables.', '']
    paths = [ROOT/'results/tables/completed_passes.csv', ROOT/'results/fixed_lineups/tables/segments.csv']
    paths += [p for folder in [ROOT/'results/tables', ROOT/'results/fixed_lineups/tables']
              for p in sorted(folder.glob('*.csv')) if p not in paths]
    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        schema = pd.DataFrame([{'column': c, 'dtype': str(df[c].dtype),
            'description': DESCRIPTION[c],
            'missing': int(df[c].isna().sum())} for c in df])
        relative = path.relative_to(ROOT/'results').as_posix()
        overview.append({'table': relative, 'rows': len(df), 'columns': len(df.columns)})
        sections.append(f'<section id="table-{i}"><h2>{html.escape(relative)}</h2>'
            f'<p>{len(df):,} rows × {len(df.columns)} columns. <a href="{relative}">CSV</a></p>'
            f'<h3>Columns</h3><div class="scroll">{schema.to_html(index=False,border=0)}</div>'
            f'<h3>First five rows</h3><div class="scroll">{df.head().to_html(index=False,border=0)}</div></section>')
        md += [f'## {relative}', '', f'{len(df):,} rows × {len(df.columns)} columns.', '',
               '| Column | Type | Description |', '|---|---|---|']
        md += [f'| {r.column} | {r.dtype} | {r.description} |' for r in schema.itertuples()]
        md.append('')
    frames, counts, example = [], [], None
    for path in sorted((ROOT/'data/open-data/data/events').glob('*.json')):
        events = json.loads(path.read_text(encoding='utf-8'))
        flat = pd.json_normalize(events, sep='.')
        flat.insert(0, 'match_id', int(path.stem)); frames.append(flat)
        counts.append({'match_id': int(path.stem), 'events': len(events), 'flattened_fields': len(flat.columns)})
        if path.stem == '3943043':
            example = next(e for e in events if e['type']['name'] == 'Pass' and e['team']['name'] == 'Spain')
    if not frames: raise FileNotFoundError('Run scripts/download_data.py first.')
    raw = pd.concat(frames, ignore_index=True)
    raw_schema = pd.DataFrame([{'column': c, 'dtype': str(raw[c].dtype),
        'non_null': int(raw[c].notna().sum()),
        'example': compact(raw[c].dropna().iloc[0]) if raw[c].notna().any() else ''} for c in raw])
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data preview — Football Passing Networks</title><style>
body{margin:0;background:#f1f5f8;color:#20364c;font:16px/1.6 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:26px}section,header{background:white;padding:25px;border-radius:12px;margin:22px 0}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dce4eb;text-align:left}.scroll,pre{overflow:auto}a{color:#27628d}.brand{width:180px}</style><main>'''
    document += '<header><img class="brand" src="../assets/statsbomb-logo.png" alt="StatsBomb"><h1>What do the data look like?</h1>'
    document += '<p>Nested JSON → cleaned passes → stable-lineup intervals → network players and edges. These are events, not frame-by-frame tracking. IDs and ball coordinates come from StatsBomb; weights and centralities are derived.</p>'
    document += '<p>Coordinates span 120 × 80 provider units, with teams attacking right; these are not pitch dimensions in metres. Missing event-specific fields often mean not applicable. Missing pass.outcome marks completion.</p>'
    document += '<p>completed_passes has 2,160 regulation passes; pass_assignments adds 241 extra-time passes. <a href="index.html">All results</a> · <a href="data_dictionary.md">Markdown dictionary</a></p>'
    document += '<div class="scroll">'+pd.DataFrame(overview).to_html(index=False,border=0)+'</div></header>'
    document += '<section><h2>Original event JSON</h2><div class="scroll">'+pd.DataFrame(counts).to_html(index=False,border=0)+'</div>'
    document += '<h3>One recorded pass</h3><pre>'+html.escape(json.dumps(example,ensure_ascii=False,indent=2))+'</pre>'
    document += f'<details><summary>All {len(raw_schema)} flattened columns</summary><div class="scroll">'+raw_schema.to_html(index=False,border=0)+'</div></details></section>'
    document += ''.join(sections)+'</main></html>'
    (ROOT/'results/data_preview.html').write_text(document,encoding='utf-8')
    (ROOT/'results/data_dictionary.md').write_text('\n'.join(md),encoding='utf-8')
    print(f'Preview: {len(paths)} derived tables, {len(raw):,} raw events, {len(raw.columns)} flattened columns.')

if __name__ == '__main__': main()
