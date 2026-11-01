"""Create the fixed-lineup notebook source, then use execute_notebook.py."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]

def build():
    nb=nbf.v4.new_notebook()
    cells=[]
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text): cells.append(nbf.v4.new_code_cell(text))
    md('# Football Passing Networks — Stable lineups\n\nNominal rosters, every link from one pass, and a separate exact-active audit. Spain–Germany, Spain–France and Spain–England at EURO 2024.\n\nData: StatsBomb Open Data / hudl. ![StatsBomb](../assets/statsbomb-logo.png)\n\nOwn-team substitutions/dismissals change the nominal lineup. Temporary departures are disclosed without splitting the main roster: all nodes need not be on the pitch at every instant. Exact active intervals and extra time remain separate. Main figures require at least five minutes and 20 completed open-play passes; all intervals remain in CSV/GraphML.')
    md('## 1. Reconstruct on-field membership\n\n- Starting XI supplies initial membership.\n- Substitution removes the outgoing player and adds the replacement.\n- Same-stoppage batches are atomic when no football event intervenes.\n- Player Off/On changes the exact-active audit, not the nominal interval.\n- Red/second-yellow dismissals remove a player without replacement.\n- Tactical Shift without membership changes updates role references only.\n\nHalf-open event indices [start_index, end_index) assign passes. Period-relative times include stoppage time and exclude breaks. A substitution batch uses its first event as the boundary; opponent substitutions do not split the lineup.')
    code("""from pathlib import Path
import sys
ROOT=Path.cwd().resolve()
if ROOT.name=='notebooks': ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
import pandas as pd
from IPython.display import display, Image, Markdown
from src.analysis import load_dataset
from src.lineups import compute_fixed_lineups, validate_fixed_lineups, export_fixed_lineups
dataset=load_dataset(ROOT)
result=compute_fixed_lineups(dataset)
validation=validate_fixed_lineups(dataset,result)
display(validation)
summary=pd.DataFrame([{k:s[k] for k in ['segment_id','team','opponent','scope','start_label','end_label','players','network_passes','duration_seconds','temporary_absence_seconds','display_eligible','display_exclusion_reason']} for s in result['segments']])
display(summary)
display(result['temporary_absences'])""")
    md('## 2. Draw every link from one pass\n\nUse completed open-play passes with identified recipients, including periods 3–4 separately. Width = 0.30 pt × pass count on a common scale. Nodes represent sent+received involvement.\n\nAll roster members appear. Filled nodes have completed-pass involvement; hollow nodes use other interval events or schematic role references. The CSV/GraphML position_source records this distinction. Small visual offsets separate overlapping labels; grey ticks mark original locations. Coordinates in data exports stay unchanged. Sparse intervals remain in tables and GraphML; different lineups are never merged to meet display thresholds.')
    code("""figures, validation=export_fixed_lineups(dataset,result,ROOT)
print(f"{len(result['segments'])} intervals; {len(figures)} PNG + {len(figures)} SVG figures.")
overview=[item for item in figures if item['kind']=='overview']
individual=[item for item in figures if item['kind']=='individual']
def show(items):
    for item in items:
        display(Markdown(f"**{item['title']}**"))
        display(Image(filename=str(ROOT/'results/fixed_lineups/figures'/item['file'])))""")
    md("## 3. Selected stable lineups — team overview pages")
    for mid in [3942226,3942752,3943043]:
        md({3942226:"### Spain–Germany: regulation and separate extra time",3942752:"### Spain–France",3943043:"### Spain–England"}[mid])
        code(f"show([f for f in overview if f['match_id']=={mid}])")
    md('## 4. Eligible intervals as separate figures\n\nThese intervals meet the declared five-minute/20-pass thresholds. Figures show raw counts, not rates per minute. Temporary absences are labelled. Omitted intervals remain in the first table and HTML report.')
    for mid in [3942226,3942752,3943043]:
        code(f"show([f for f in individual if f['segment_id'].startswith('{mid}_')])")
    md('## 5. Verification and exports\n\nEvery completed pass belongs to exactly one interval, edge endpoints belong to its roster, and summing all interval edge weights reproduces the original network. Repeat the checks for exact active membership. Regulation retains 2,160 passes and extra time 241; synthetic tests cover substitution batches, returns and dismissals.')
    code("""display(validate_fixed_lineups(dataset,result))
display(pd.read_csv(ROOT/'results/fixed_lineups/tables/segments.csv')
        .groupby(['team','opponent','scope']).agg(intervals=('segment_id','size'),passes=('network_passes','sum')))
print('results/fixed_lineups/report.html')""")
    nb.cells=cells
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
    path=ROOT/'notebooks/fixed_lineups.ipynb'
    nbf.write(nb,path)
    print(path)

if __name__=='__main__': build()
