"""Command-line training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .features import FEATURE_COLUMNS, build_model_dataset, load_games
from .modeling import evaluate_predictions, train_logistic_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, default=Path("games.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/nba_logistic.joblib"))
    args = parser.parse_args()

    data = build_model_dataset(load_games(args.games))
    development = data[data["SEASON"] <= 2020]
    holdout = data[data["SEASON"] >= 2021]
    model = train_logistic_model(c=0.01).fit(development[FEATURE_COLUMNS], development["home_team_win"])
    probabilities = pd.Series(model.predict_proba(holdout[FEATURE_COLUMNS])[:, 1], index=holdout.index)
    metrics = evaluate_predictions(holdout["home_team_win"], probabilities)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "training_through_season": 2020,
        "holdout_seasons": sorted(int(value) for value in holdout["SEASON"].unique()),
        "holdout_metrics": metrics,
    }
    joblib.dump(payload, args.output)
    print(json.dumps({"artifact": str(args.output), **metrics}, indent=2))


if __name__ == "__main__":
    main()
