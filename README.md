# Independent evaluation with XGBoost early stopping

A focused, AI-assisted documentation contribution for CSCI 566. It separates early-stopping
validation from independent test evaluation in the XGBoost scikit-learn guide. The change is
to documentation examples, not XGBoost training code or its API.

## Start here

- [Four-part, four-page contribution report (PDF)](CSCI566_XGBoost_Contribution_Report.pdf)
- [Editable report](report.md)
- [Complete reproducible contribution archive](XGBoost_early_stopping_contribution.zip)
- [Inspectable upstream patch](xgboost-early-stopping.patch)

Download and extract the ZIP to get the full directory layout, source documents, scripts,
requirements, reference evidence, prior-work review, and rendered documentation preview.
The root of this GitHub repository provides the report, patch, and archive for easy review;
the nested runnable project is inside the archive.

## What changed

The introduction reserves an independent final test partition before creating training and
early-stopping validation partitions. The cross-validation example similarly creates inner
validation data from each outer training fold and scores the untouched outer fold afterward.
The text distinguishes fixed-procedure CV evaluation from using those scores for tuning.

Upstream: https://github.com/dmlc/xgboost

Pinned commit: `293cad1a89ab749e3266e190c68118af2db09982`

Changed file: `doc/python/sklearn_estimator.rst` (44 insertions, 23 deletions).
The original examples execute successfully; the issue is evaluation independence and naming,
not a runtime failure. A separate test set is needed when an independent final estimate is
required, not for every training-only workflow.

## Reproduce after extracting the archive

```bash
cd xgboost-early-stopping-contribution
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_examples.py
```

Use Python 3.12. On Windows, activate with `.venv\Scripts\Activate.ps1` instead.
Successful verification prints `"status": "PASS"` and writes fresh evidence under
`evidence/rerun/`. Tests use only scikit-learn's bundled public dataset.

The archive README also explains how to restore the pinned upstream checkout, apply the
patch, and build the isolated Sphinx preview. No script pushes code or opens a pull request.

## Evidence and attribution

The contributor reports that local verification passed. Their local commands, software
versions, operating system, and numerical logs were not supplied. Detailed results under
`evidence/assistant_run/` in the archive are a separately recorded AI-assisted reference run,
not logs attributed to the contributor's own machine.

The reference checks passed for all four revised RST code blocks, disjoint data partitions,
callback consistency, the held-out-label perturbation check, patch application, and an
isolated Sphinx preview. The reference run used Python 3.12.14, xgboost-cpu 3.4.1,
scikit-learn 1.8.0, NumPy 2.3.5, and SciPy 1.17.0. Portable requirements use xgboost 3.4.1.

AI assisted background research, patch and script drafting, reference execution, and report
preparation. The report explicitly separates that assistance from the contributor's own
local-verification statement. See `review/AI_ASSISTANCE.md` inside the archive.

Not performed: source compilation, GPU tests, the full upstream test suite, the complete
upstream documentation build, or public CI. No upstream issue or pull request was opened,
and no course submission was made. No maintainer acceptance or guaranteed credit is claimed.

## Contents inside the archive

- `doc/python/sklearn_estimator.rst`: revised upstream documentation.
- `upstream/sklearn_estimator.rst`: original pinned source.
- `scripts/`: reproducibility, checkout restoration, and preview helpers.
- `evidence/`: separately labeled AI-assisted reference results and contributor confirmation.
- `review/`: prior-work search and AI-assistance disclosure.
- `preview/`: isolated rendered HTML documentation preview.
- `report/`: four-page PDF and editable Markdown.

The XGBoost files retain their Apache-2.0 license; see [LICENSE-XGBoost](LICENSE-XGBoost).
