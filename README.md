# NBA Game Prediction

A reproducible, leakage-aware model that estimates the probability that the home team wins an NBA regular-season game.

## Current result

The primary model is scaled logistic regression. Hyperparameter selection uses the 2019–2020 seasons; the 2021–2022 seasons are held out until final evaluation.

| Evaluation | Accuracy | ROC-AUC | Log loss | Brier score |
|---|---:|---:|---:|---:|
| 2019–2020 validation | 64.59% | 0.6960 | 0.6312 | 0.2204 |
| 2021–2022 holdout | 64.18% | 0.6766 | 0.6431 | 0.2251 |

The holdout home-team baseline is 55.78%. The supplied 2022 data ends on December 22, so that season is partial.

## Method

- Completed regular-season games only (`GAME_ID` type `002`)
- Exact duplicate games removed before feature generation
- Five- and ten-game form calculated with a one-game shift and reset each season
- Pregame Elo with 100-point home advantage and 25% offseason regression toward 1500
- Rest capped at seven days, with explicit back-to-back indicators
- Whole-season validation rather than random cross-validation
- Standardization learned from training data only
- Accuracy, ROC-AUC, log loss, and Brier score reported

The model uses only information available before tipoff. Player-level and standings files are intentionally not loaded because the current model does not use them.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Place `games.csv` in the repository root. The supplied dataset originates from Kaggle; record the exact dataset URL and license before redistributing the data.

## Run

Open and run `main.ipynb` from top to bottom, or train from the command line:

```bash
python -m nba_prediction.train --games games.csv
```

The command saves `artifacts/nba_logistic.joblib` with the pipeline, ordered feature names, training cutoff, and holdout metrics. Artifacts are ignored by Git because they are reproducible.

Run tests with:

```bash
pytest
```

## Repository layout

- `nba_prediction/features.py` — data validation and pregame feature engineering
- `nba_prediction/modeling.py` — season splits, model pipeline, and probability metrics
- `nba_prediction/train.py` — reproducible training entry point
- `tests/` — leakage and season-boundary tests
- `main.ipynb` — reader-facing analysis and calibration plots

## Known limitations

- Data ends in December 2022 and needs an update pipeline for current predictions.
- The 2022 holdout is a partial season.
- Elo and rolling form do not encode injuries, roster changes, travel distance, or starting lineups.
- A deployable future-game predictor still needs a schedule feed and a persisted team-state snapshot generated as of the prediction date.
- Historical data provenance is not encoded in the CSV files; add a source manifest before publishing the dataset.
