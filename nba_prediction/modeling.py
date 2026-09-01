"""Season-based model selection and probability evaluation."""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


def train_logistic_model(c: float = 0.1) -> Pipeline:
    """Return a scale-aware, regularized probability model."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, max_iter=2000, random_state=42)),
    ])


def season_splits(data: pd.DataFrame, minimum_train_seasons: int = 5):
    """Yield expanding-window train/validation indices with whole-season boundaries."""
    seasons = sorted(data["SEASON"].unique())
    for position in range(minimum_train_seasons, len(seasons)):
        validation_season = seasons[position]
        yield data.index[data["SEASON"] < validation_season], data.index[data["SEASON"] == validation_season]


def evaluate_predictions(actual, probabilities: pd.Series) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(actual, predictions),
        "roc_auc": roc_auc_score(actual, probabilities),
        "log_loss": log_loss(actual, probabilities),
        "brier_score": brier_score_loss(actual, probabilities),
    }


def walk_forward_scores(data: pd.DataFrame, model: Pipeline | None = None) -> pd.DataFrame:
    """Evaluate a model on successive unseen seasons."""
    estimator = model or train_logistic_model()
    rows = []
    for train_index, validation_index in season_splits(data):
        fitted = clone(estimator).fit(data.loc[train_index, FEATURE_COLUMNS], data.loc[train_index, "home_team_win"])
        probabilities = pd.Series(
            fitted.predict_proba(data.loc[validation_index, FEATURE_COLUMNS])[:, 1],
            index=validation_index,
        )
        rows.append({"season": int(data.loc[validation_index, "SEASON"].iloc[0]), **evaluate_predictions(data.loc[validation_index, "home_team_win"], probabilities)})
    return pd.DataFrame(rows)
