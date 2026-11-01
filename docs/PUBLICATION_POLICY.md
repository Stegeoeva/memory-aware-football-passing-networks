# Public publication policy

This is a **non-commercial research publication**, not a redistribution of the StatsBomb dataset. The public package is deliberately narrower than the complete local analysis.

## Source terms checked

The StatsBomb Public Data User Agreement at the pinned source commit is last updated 8 September 2023. Its PDF SHA256 is `a5462e69aeb71a39268b760b110c5c2190a2e0cee3015ba976f66bd71f6c2bb4`.

- The introduction permits public sharing of analysis/conclusions, without implying StatsBomb endorsement.
- Section 1.2.1 prohibits distribution/reproduction/provision of the data to third parties.
- Section 1.2.2 prohibits commercial exploitation of the data or analysis derived from the service.
- Section 1.4 requires the StatsBomb brand logo on publications of analysis.
- Section 7 reserves proprietary rights and refers to prior written consent for otherwise unpermitted uses.

Primary sources: [pinned agreement](https://github.com/hudl/open-data/blob/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/LICENSE.pdf), [current agreement](https://github.com/hudl/open-data/blob/master/LICENSE.pdf), and [upstream README](https://github.com/hudl/open-data/blob/master/README.md). The README also requests source attribution and the logo.

These are summaries, not substitute license terms. This packaging decision is an interpretation for a non-commercial research publication; it is not a legal opinion, written permission or a guarantee for every use. Obtain a qualified review or written clarification from Hudl/StatsBomb for commercial use or publication of additional data-derived artifacts. No request has been sent on your behalf.

## Included publicly

- Original source code, synthetic tests, untrained configurations and methods.
- Source-download instructions and checksum manifests; users obtain inputs directly from the official repository under its terms.
- Four notebooks as **source only**, with execution counts, outputs, widget state and attachments cleared.
- Twelve explicitly selected **aggregate analytical figure pairs** and a static gallery. They show model-level comparisons, motif definitions/frequencies, and aggregate source-xG estimates/uncertainty, rather than event paths or player-level data tables.
- The original StatsBomb attribution logo and clear source/usage notices.

The public gallery shows the logo alongside each analytical figure. If sharing a figure separately, include StatsBomb attribution and its logo alongside it; a standalone image does not acquire an MIT license.

## Excluded from the public package

Raw event/lineup JSON; normalized or detailed CSV/CSV.gz; event UUIDs and actual sequence records; per-pass and OOF prediction tables; saved probability matrices; adjacency matrices/GraphML exports; source-event examples; shot freeze frames; serialized learned parameters including xT grids; full run manifests embedding local artifacts; and all executed notebook outputs.

These remain available **locally** after running the pipeline. Public figures/insights are not relicensed under MIT. The MIT license covers only the project's original source code and does not waive StatsBomb restrictions on data or analyses.

## Enforcement

`publication_allowlist.json` and `scripts/publication_policy.py` define the files copied into the public ZIP. Packaging does not recursively include all research results. `scripts/check_repository.py` rejects notebook outputs, excluded data formats, links to unpublished files, and non-allowlisted figures. CI runs the public checks and synthetic unit tests without downloading source data or fitting models.

The `.gitignore` excludes detailed local outputs and source caches. After local notebook execution, clear outputs again before committing. Run `python scripts/package_repository.py` to create the public archive. Review the staged file list before pushing; an allowlist-based package is preferable to uploading the complete working directory.

The earlier approximately 274 MB archive was a full local research delivery and is superseded for public upload. Use the **PUBLIC** archive and its clean extracted directory.
