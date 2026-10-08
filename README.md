# Spoken Language Quality Regression

A reproducible Python workflow for predicting a continuous spoken-language quality score from WAV recordings.

This repository is intentionally data-free. Supply audio and CSV manifests that you are permitted to use.

## Input layout

```text
your_data/
├── train.csv            # filename,label
├── test.csv             # filename
├── train/               # training WAV files
└── test/                # prediction WAV files
```

The column names are configurable through command-line options.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python audio_regression_workflow.py --data-dir /path/to/your_data --output-dir outputs
```

This writes `predictions.csv`, `metrics.json`, and `feature_importance.csv` in the chosen output directory.

## Method

The workflow extracts duration, energy, zero-crossing rate, spectral descriptors, MFCC summaries, chroma summaries, and pitch summaries. An `ExtraTreesRegressor` is evaluated with fixed-seed K-fold cross-validation, then refit on all labeled rows to generate predictions.

Audio descriptors alone do not directly represent syntax or semantics. For language-quality tasks, transcript features can be a useful extension when they are permitted and responsibly sourced.

## Reproducibility

Set `--seed` and `--folds` explicitly when comparing experiments. Results are deterministic for a given input dataset and dependency environment.
