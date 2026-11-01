# Reproduction guide

The public repository contains source code, methodology, source-only notebooks and twelve aggregate figure pairs. Detailed tables, saved predictions, trained xT grids and executed notebook outputs are **generated locally**, not redistributed.

## 1. Review the source terms

Read the [StatsBomb agreement](https://github.com/hudl/open-data/blob/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/LICENSE.pdf), [upstream README](https://github.com/hudl/open-data/blob/master/README.md), and [publication policy](PUBLICATION_POLICY.md). The dataset and resulting analysis have redistribution/commercial-use restrictions. The MIT license applies only to project source code.

## 2. Set up Python

Tested with Python 3.12. Run from the repository root. Create an environment with `python -m venv .venv`, activate using `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux, then:

~~~shell
python -m pip install -r requirements-memory-lock.txt
python -m unittest discover -s tests -v
python scripts/check_repository.py
~~~

The 82 synthetic tests and public-file checks need no downloaded source data. No dataset or model is obtained from this project. The four requirements files separate direct dependencies and complete tested environments.

## 3. Inspect public findings

Browse the README images, or run `python -m http.server 8000` and open `http://localhost:8000/results/public/index.html`. Stop with Ctrl+C. GitHub renders images and notebook source, while ordinary repository HTML views show source.

## 4. Download inputs directly and rebuild locally

After reviewing the upstream terms:

~~~shell
python scripts/download_data.py
python scripts/download_xt_training.py
python scripts/run_analysis.py
python scripts/run_extended.py
python scripts/build_euro_sequences.py
python scripts/run_memory_study.py
python scripts/run_motifs.py
python scripts/run_motif_xg.py
~~~

The downloaders obtain pinned files directly from the official hudl/open-data repository, verify SHA256 and skip matching files. The second retrieves approximately 151 MB for the other 48 matches. Their union supplies all 51 matches. Only download commands access the network; the source terms remain applicable.

The aggregate runner also creates stable-lineup outputs; `run_fixed_lineups.py` runs that part alone. The original extension's xT excludes its three evaluated matches. Tournament continuation separately fits training-only xT within outer/inner splits.

Nested fitting is the expensive stage: 45 outer memory experiments, 20 training-only xT fits, motif q models, another 80 outer source-xG q models and 2,000-draw uncertainty estimates. Runtime varies with hardware/BLAS; no portable benchmark is claimed. Matching ignored `.runtime/` checkpoints allow resuming. Memory/motif/motif-xG runners accept `--stage report` after their numerical exports exist.

## 5. Validate and execute notebooks locally

~~~shell
python scripts/check_euro_sequences.py
python scripts/check_memory_study.py
python scripts/check_motifs.py
python scripts/check_motif_xg.py
python scripts/execute_notebook.py memory_aware_analysis.ipynb
python scripts/execute_notebook.py football_passing_networks.ipynb
python scripts/execute_notebook.py fixed_lineups.ipynb
python scripts/execute_notebook.py extended_analysis.ipynb
~~~

The scientific checkers verify regenerated source paths, targets, folds, saved probabilities, denominators, intervals and file hashes. The research overview reads the locally generated study outputs; the case-study notebooks run their own analyses. None is delivered with data-bearing outputs in the public package. Full local reports and dictionaries appear under ignored `results/` subdirectories.

The `.gitattributes` file preserves committed bytes, avoiding checkout newline conversion in hash-verified files. Reproduction across platforms does not promise byte-identical model fitting.

## 6. Prepare a public package after local analysis

~~~shell
python scripts/strip_notebook_outputs.py
python scripts/check_repository.py
python scripts/package_repository.py
~~~

Executed notebook backups remain in ignored `.runtime/private_notebooks/`. The public ZIP uses an explicit allowlist and omits detailed results, learned parameters and all notebook outputs. Never upload the complete local analysis directory or its old full-delivery archive. [Upload instructions](UPLOAD.md).
