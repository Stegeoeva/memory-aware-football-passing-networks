"""Create the English overview notebook from locally generated study outputs; no training."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

def main():
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(s))
    def code(s): cells.append(nbf.v4.new_code_cell(s))
    md('''# Memory-Aware Football Passing Networks

## An exploratory EURO 2024 study

**Does knowing the previous sender help explain the next pass and its attacking continuation?**

All 51 matches · real event-UUID paths · held-out prediction · uncertainty-aware continuation · source-shot xG.

Data: [StatsBomb Open Data](https://github.com/hudl/open-data). ![StatsBomb](../assets/statsbomb-logo.png)

This notebook reads locally generated tables and figures. Run the reproduction pipeline first; data-bearing outputs are not distributed. The overview itself does not train models. See the [reproduction guide](../docs/REPRODUCIBILITY.md) for a full rebuild.''')
    code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
assert (ROOT/'results/memory').is_dir(), 'Run from the repository root or notebooks/.'
def read(path): return pd.read_csv(ROOT/path)
def show(path): display(Image(filename=str(ROOT/path)))
coverage = json.loads((ROOT/'results/euro_sequences/validation.json').read_text())
display(pd.DataFrame([{'matches':51,'teams':24,'events':coverage['events_periods_1_to_4'],
    'pass_attempts':coverage['pass_attempts'],'memory_paths':coverage['successful_memory_targets']}]))''')
    md('''## 1. From ordinary networks to real sequences

Ordinary directed edges aggregate completed A→B passes. They describe circulation, not the order of a possession. Here actual event links reconstruct A→B→C; failures, other handlers and opponent control are not skipped. Regulation/extra time remain labelled and shootouts are excluded. Initial-receipt context is separate from future labels.''')
    code('''examples = read('results/euro_sequences/sequence_examples.csv')
display(examples)
show('results/figures/01_network_overview.png')''')
    md('''## 2. Does the previous state improve prediction?

Compare P(C | B) with P(C | A,B), on identical held-out paths and common candidate sets. Five outer whole-match folds keep both teams together; three inner folds tune models. Three repeats are sensitivities, not independent samples. The main intervals use 2,000 paired whole-match bootstrap draws conditional on fitted models.

Spatial history improves prediction beyond common receipt context. Player-identity effects are smaller and baseline-sensitive; rolling evaluation uses different history/tuning.''')
    code('''comparison = read('results/memory/tables/memory_comparisons.csv')
display(comparison.loc[comparison.comparison.isin(['M2_vs_M1','M2_context_vs_M1_context'])])
show('results/memory/figures/01_memory_gain.png')''')
    md('''## 3. What follows a player's completed pass?

R3 records the first shot's source xG within three later ball actions, confirmed loss as zero, or training-only xT at a retained endpoint. Unknowns remain censored. q and xT exclude the evaluated match, including in inner tuning. This is a mixed exploratory proxy, not pure expected goals.

Continuation excess is 100 × mean[R3 − OOF q(receipt context)]. Read support, coverage, role and intervals together. No supported positive primary player excess has an interval wholly above zero; a definitive ranking is not supported. Backward passes are not automatically penalised and B's error is not automatically A's fault.''')
    code('''players = read('results/continuation/tables/player_effects.csv')
primary = players.loc[players.metric.eq('R3') & players.ranking_eligible]
cols = ['a','team','role','observed_passes','observed_matches','raw_per100',
        'adjusted_excess_per100','ci95_low','ci95_high']
display(primary.sort_values('adjusted_excess_per100',ascending=False)[cols].head(12))''')
    md('''## 4. Completed motifs and later source-xG outcomes

ABA/ABC contain two passes. ABAB/ABAC/ABCA/ABCB/ABCD contain three. Outcomes start after the final receipt; required motif-forming passes are not counted as subsequent success. This conditions on completed formation.

For the xG comparison, follow the remainder of the same provider possession, stopping earlier at opponent control, goal or observed period end. Only complete no-shot windows receive zero. Source xG is not mixed with xT. Multiple shots are summed; xG/shot is a ratio of sums. Shared shots create overlapping credits, so unique-shot sensitivity and joint match/possession bootstrap are included.''')
    code('''xg = read('results/motif_xg/tables/primary_comparison.csv')
display(xg)
show('results/motif_xg/figures/01_frequency_value_quality.png')
contrasts = read('results/motif_xg/tables/pattern_contrasts.csv')
display(contrasts.loc[contrasts.order.eq(3) & contrasts.horizon.eq('possession')
    & contrasts.policy.eq('all_linked') & contrasts.scope_filter.eq('all_periods')
    & contrasts.context.eq('raw') & contrasts.metric.eq('xg_per_motif')])''')
    md('''## 5. What can we conclude?

ABCD has higher point estimates of shot frequency and xG/motif than ABAB, with close linked-shot quality. Its xG/motif difference is +0.00328 [95% CI −0.00391, +0.00931]: no reliable advantage is established. Context-adjusted full-possession xG models do not consistently outperform a training-mean baseline; treat their residual comparisons as exploratory.

The strongest finding is useful spatial memory in held-out next-pass prediction. Causal sender value, unseen alternatives and definitive player rankings remain unresolved. Intervals are pointwise and conditional on fitted models; one tournament and repeated teams limit generalization. No significance-test family or FDR claim is made.''')
    code("show('results/motif_xg/figures/03_context_adjusted_xg.png')\nshow('results/motif_xg/figures/05_overlap_weighting.png')")
    md('## Further reading and reproducibility\n\n- [Study guide](../docs/STUDY_GUIDE.md), [memory methods](../docs/MEMORY_EVALUATION.md), [motifs](../docs/MOTIFS.md), [source-xG definitions](../docs/MOTIF_XG.md).\n- All local reports (`../results/index.html`, generated locally), checks (`../results/delivery_check.json`, generated locally), [public release provenance](../provenance/public_release.json).\n- Rosvall et al. (2014), Håland et al. (2020), VAEP and Expected Threat motivate the method; this study is an adaptation, not a reproduction.\n- Infomap communities and additional competitions remain future work.\n\nRun `python -m unittest discover -s tests -v` and `python scripts/check_repository.py` for the publication checks. Full source rebuild commands are in the reproduction guide.')
    nb=nbf.v4.new_notebook(cells=cells)
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                 'language_info':{'name':'python','version':'3.12'}}
    path=ROOT/'notebooks/memory_aware_analysis.ipynb';nbf.write(nb,path);print(path)

if __name__=='__main__':main()
