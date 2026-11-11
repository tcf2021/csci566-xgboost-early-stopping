# XGBoost early stopping: validation and test data

This is my CSCI 566 contribution to the XGBoost documentation. I used XGBoost in an
SDSS photometric-redshift project, and chose to work on its early-stopping examples.

The original CV example uses the same held-out fold to choose the stopping iteration
and report a test score. The patch separates those two steps: an inner validation split
selects the stopping iteration, then the outer test fold evaluates the fitted model.
The introductory example and callback example use the same distinction.

The change is in `doc/python/sklearn_estimator.rst`: 44 lines added and 23 removed.
It is based on upstream commit `293cad1a89ab749e3266e190c68118af2db09982`.

## Report and files

- [Report (PDF)](CSCI566_XGBoost_Contribution_Report.pdf)
- [LaTeX source](report.tex)
- [Patch](xgboost-early-stopping.patch)
- [Full project ZIP](XGBoost_early_stopping_contribution.zip)

The ZIP contains the original and revised documentation, verification scripts, reference
results, prior-work notes, and an HTML preview. Download and extract it before running
the commands below, then enter the `xgboost-early-stopping-contribution` directory.

## Run the checks

Use Python 3.12 in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_examples.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. A passing run prints
`"status": "PASS"` and writes its output to `evidence/rerun/`. The example uses the public
breast-cancer dataset bundled with scikit-learn, not data from my SDSS project.

The reference environment used `xgboost-cpu==3.4.1` on Linux. The requirements file uses
standard `xgboost==3.4.1` for portability; the other package versions are pinned too.

The README inside the ZIP also has commands for restoring the upstream checkout and
building the documentation preview. To rebuild the report with a standard TeX Live
or MiKTeX installation, run twice in the folder containing `report.tex`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

## Checks

I confirmed that local verification passed. The detailed results included here are from
a separate AI-assisted reference run, not my own local logs.

That run checks the code directly from the revised RST file. All four code blocks run in
order, and training, validation, and test rows are disjoint. With the outer split fixed,
flipping only the test labels leaves the revised model unchanged. The original helper
changes its selected stopping iteration under the same check.

The patch also applies cleanly to the pinned upstream revision, and the isolated Sphinx
preview builds. These checks used a CPU wheel; they do not cover a source build, GPU
tests, the full XGBoost test suite, or the complete documentation site.

## AI assistance

I used AI for background research, drafting the patch and helper scripts, the recorded
reference run, and editing the report and README. The report explains the change and
the evidence in more detail.

Upstream: https://github.com/dmlc/xgboost

The included XGBoost files retain their Apache-2.0 license; see `LICENSE-XGBoost`.
