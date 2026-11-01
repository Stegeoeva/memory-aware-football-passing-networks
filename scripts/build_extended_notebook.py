"""Generate the event and downstream-pass analysis notebook."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]

def main():
    nb=nbf.v4.new_notebook();cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md("# Beyond passing networks: from A to B to C\n\nShots/xG, possessions, progression, pressure, losses/recoveries, zones, before/after windows, xT and temporal motifs.\n\nData: StatsBomb Open Data / hudl. ![StatsBomb](../assets/statsbomb-logo.png)\n\nBackward passes are not automatically bad. Later outcomes do not isolate the initial sender's responsibility. The xT baseline trains on the other 48 EURO matches, excluding the three evaluated matches.")
    md('## 1. Reproduce exported datasets\n\nRun `python scripts/download_xt_training.py` if the approximately 151 MB cache is missing. Execution then uses local files only. The same functions reproduce 23 CSV tables and 15 PNG/SVG figure pairs.')
    code('''from pathlib import Path
import sys
ROOT=Path.cwd().resolve()
if ROOT.name=='notebooks': ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
import pandas as pd
from IPython.display import display,Image,Markdown
from src.extended import run_extended
dataset,tables,model,figures,validation=run_extended(ROOT)
display(validation)
display(pd.DataFrame([{'table':name,'rows':len(df),'columns':len(df.columns)} for name,df in tables.items()]))''')
    md('## 2. Shots, possession sequences and progression\n\nxG comes from source shots. Possessions use provider IDs with period separation. Passes/carries retain origin, endpoint, success, direction and final-third/box entries. Ten coordinate units is an explicit operational progression threshold.')
    code('''display(tables['shots'][['match_id','team','player','scope','x','y','outcome','xg']].head())
display(tables['possessions'].head())
display(tables['progression_players'].head())
def show(start,stop):
    for item in figures[start:stop]:
        display(Markdown('**'+item['title']+'**'))
        display(Image(filename=str(ROOT/'results/extended/figures'/item['file'])))
show(0,4)''')
    md('## 3. Pressure, losses, recoveries, zones and event windows\n\nCompletion uses all attempts. Not flagged does not prove pressure was absent. An unsuccessful pass is not automatically a loss of control: inspect the next controlled touch. Before/after windows can overlap and do not establish causality.')
    code("show(4,8)")
    md('## 4. Expected threat from a separate training set\n\nThe 12 × 8 spatial xT model uses V = p_shot*p_goal + diag(1−p_shot)*T*V. Training excludes evaluated matches. This is retrospective match separation, not chronological forecasting. Smoothing and zero continuation after unsuccessful moves are recorded in model JSON. No VAEP/OBV or calibrated xT claim is made.')
    code("display(model.metadata)\nshow(8,9)")
    md("## 5. The receiver's next action and the initial pass's future\n\nFor each A→B, record B's resolution, actual B→C, completion, direction, loss and xT. B may carry or complete take-ons first; other handlers and opponent control are not skipped. Paths do not bridge possessions/periods. Time horizons start at receipt and flag early period ends. through_receiver_release_xt_delta spans A's origin to B's completed-pass endpoint: a shared path outcome, not exclusive credit to A.")
    code('''cols=['passer','recipient','receiver_resolution','receiver_loss_observed','b_next_pass_completed',
      'b_next_pass_direction','c_name','initial_xt_delta','b_next_pass_xt_delta',
      'through_receiver_release_xt_delta','future_possession_xg']
display(tables['followup_passes'][cols].head(12))
show(9,12)''')
    md("## 6. Multiple horizons and backward passes\n\nOne future chance can follow several passes, so future-xG fields cannot be summed as independent opportunities. Action horizons stop at possession boundaries and retain observed-action coverage. B's pass direction and future value remain separate measures.")
    code("show(12,14)")
    md('## 7. Shot-prefix xGChain / xGBuildup and literature\n\nEach successful pre-shot participant receives the full shot xG once. Buildup excludes the shooter and key-pass provider. Credits overlap and are not divided among players. Only the prefix before each shot is used, avoiding retrospective credit to later participants.\n\nRelated methods include VAEP, xT, EPV, xGChain/xGBuildup, OBV, football flow motifs and memory networks. Exact definitions, literature and limits (`../results/extended/METHODOLOGY.md`, generated locally).')
    code("show(14,15)\ndisplay(tables['xg_player_credits'].head())\ndisplay(validation)")
    nb.cells=cells;nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
    path=ROOT/'notebooks/extended_analysis.ipynb';nbf.write(nb,path);print(path)
if __name__=='__main__':main()
