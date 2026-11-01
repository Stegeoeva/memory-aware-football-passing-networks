"""Create the readable notebook source; run execute_notebook.py to execute it."""
from pathlib import Path
import sys
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

def build():
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))
    md("# Football Passing Networks\n## Spain's last three matches at EURO 2024\n\n**Python · pandas · NetworkX · Matplotlib**\n\nThree matches, six directed weighted networks and 15 figures from [StatsBomb Open Data](https://github.com/hudl/open-data).\n\n![StatsBomb](../assets/statsbomb-logo.png)\n\nWhich players are most involved? Which links are strongest? How does Spain's connectivity change by opponent and half?")
    md('### 1. Scope and reproducibility\n\nSpain faced Germany on 5 July, France on 9 July and England on 14 July 2024. We analyse periods 1–2 including stoppage time. Germany extra time is excluded: the final was 2–1 after extra time, with 1–1 after regulation.\n\nSource JSON is unchanged. data/manifest.json records the commit and SHA256. This notebook reads local inputs; first run `python scripts/download_data.py` if needed.')
    code("""from pathlib import Path
import sys
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
assert (ROOT / 'src/analysis.py').is_file(), 'Run from the project folder or notebooks/.'
sys.path.insert(0, str(ROOT))
import pandas as pd
import networkx as nx
from IPython.display import display, Image, Markdown
from src.analysis import (
    load_dataset, compute_tables, validate_analysis, export_tables,
    create_figures, write_findings, write_gallery,
)
dataset = load_dataset(ROOT)
passes = dataset['passes']
display(dataset['matches'])
print(f'{len(passes):,} pass events in the three source files, including extra time.')""")
    md('### 2. Clean and filter pass events with pandas\n\npd.json_normalize flattens nested JSON. Missing pass.outcome marks completion. Network passes require a named recipient, regulation time and different sender/recipient. Exclude direct Corner, Free Kick, Goal Kick, Kick Off and Throw-in passes; retain Recovery/Interception and subsequent ordinary passes. A restart play_pattern alone does not exclude the whole possession. The audit accounts for each filter.')
    code("""tables = compute_tables(dataset)
display(tables['audit'])
eligible = passes.loc[passes['network_pass']].copy()
display(eligible[['match_id', 'team', 'passer_name', 'recipient_name', 'x', 'y', 'end_x', 'end_y']].head(10))
validation = validate_analysis(dataset, tables)
export_tables(dataset, tables, ROOT, validation)
display(validation)""")
    md('### 3. A directed, weighted graph in a few lines\n\nNodes are player IDs. An edge i → j represents at least one completed pass; its weight is the count. Names are labels only. The following NetworkX construction makes the aggregation explicit.')
    code("""sample = eligible.loc[eligible['match_id'].eq(3943043) & eligible['team'].eq('Spain')]
weights = sample.groupby(['passer_id', 'recipient_id']).size().reset_index(name='weight')
G = nx.DiGraph()
for row in weights.itertuples(index=False):
    G.add_edge(int(row.passer_id), int(row.recipient_id), weight=int(row.weight))
assert sum(nx.get_edge_attributes(G, 'weight').values()) == len(sample)
print(f'Spain vs England: {G.number_of_nodes()} nodes, {G.number_of_edges()} directed links, {len(sample)} passes.')
display(weights.nlargest(8, 'weight'))""")
    md('### 4. Metrics and definitions\n\n- Out/in-strength: completed passes sent/received.\n- Involvement: their sum; each pass contributes twice across the team.\n- Density: E / [N(N−1)].\n- Reciprocity: directed links with an observed reverse link.\n- PageRank: weighted incoming prominence, damping 0.85.\n- Betweenness: normalized directed shortest paths with distance 1/weight.\n\nParticipants include substitutes and are not adjusted for playing time.')
    code("""display(tables['teams'].round(3))
display(tables['players'].loc[tables['players']['team'].eq('Spain')]
        .sort_values(['match_id', 'involvement'], ascending=[True, False])
        .groupby('match_id').head(5)
        [['match_id', 'name', 'passes_sent', 'passes_received', 'involvement', 'pagerank', 'betweenness']]
        .round(3))""")
    md('### 5. Passing networks on the pitch\n\nPositions average completed-pass origins and receipt endpoints in StatsBomb 120 × 80 coordinates; teams attack right. These are ball-event locations, not tracking formations. Node area represents involvement. Every link from one pass is drawn at 0.30 pt per pass, on common scales. For stable rosters see the [lineup notebook](fixed_lineups.ipynb) and gallery (`../results/fixed_lineups/report.html`, generated locally).')
    code("""figures = create_figures(dataset, tables, ROOT / 'results/figures')
def show_figures(start, stop):
    for item in figures[start:stop]:
        display(Markdown(f"**{item['title']}**"))
        display(Image(filename=str(ROOT / 'results/figures' / item['file'])))
show_figures(0, 4)""")
    md('### 6. Directed adjacency matrices\n\nRows are senders, columns recipients, cells pass counts. All six teams use the same colour scale; all observed links are included.')
    code("show_figures(4, 7)")
    md('### 7. Central players\n\nThe eight most involved players per team; bars separate sent and received passes. PR is PageRank. High passing volume alone does not establish effectiveness.')
    code("show_figures(7, 10)")
    md("### 8. Compare teams and Spain's player roles\n\nCompare volume, completion, density, reciprocity, top-three concentration and mean pass length. The second figure follows the same Spain players across matches. Different substitutions and minutes affect interpretation.")
    code("show_figures(10, 12)")
    md("### 9. Time, halves and pass direction\n\nStoppage time is pooled into each half's last five-minute bin, so 40–45 and 85–90 cover longer actual durations. Direction uses end_x − start_x: forward >5, backward <−5, lateral within ±5. This is not a progressive- or line-breaking-pass definition.")
    code("show_figures(12, 15)")
    md('### 10. Findings and limitations')
    code("""findings = write_findings(dataset, tables, ROOT / 'results')
write_gallery(ROOT, figures, tables)
display(Markdown(findings))
print('All results, tables, graphs and figures saved under results/.')""")
    nb.cells = cells
    nb.metadata = {"kernelspec":{"display_name":"Python 3", "language":"python", "name":"python3"},
                   "language_info":{"name":"python", "version":"3.12"}}
    path = ROOT / "notebooks/football_passing_networks.ipynb"
    path.parent.mkdir(exist_ok=True)
    nbf.write(nb,path)
    print(path)

if __name__ == "__main__":
    build()
