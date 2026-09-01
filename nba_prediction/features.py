"""Leakage-aware feature engineering for NBA games."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "elo_diff",
    "rolling_win_pct_5_diff",
    "rolling_win_pct_10_diff",
    "rolling_point_diff_5_diff",
    "rolling_point_diff_10_diff",
    "rest_days_diff",
    "home_back_to_back",
    "away_back_to_back",
]


def load_games(path: str | Path = "games.csv") -> pd.DataFrame:
    """Load only columns used by the regular-season model."""
    columns = [
        "GAME_ID", "GAME_DATE_EST", "SEASON", "HOME_TEAM_ID",
        "VISITOR_TEAM_ID", "PTS_home", "PTS_away",
    ]
    return pd.read_csv(path, usecols=columns, parse_dates=["GAME_DATE_EST"])


def prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    """Validate, deduplicate, and retain completed regular-season games."""
    required = {
        "GAME_ID", "GAME_DATE_EST", "SEASON", "HOME_TEAM_ID",
        "VISITOR_TEAM_ID", "PTS_home", "PTS_away",
    }
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    base = games.copy()
    base["GAME_DATE_EST"] = pd.to_datetime(base["GAME_DATE_EST"], errors="raise")
    base = base.dropna(subset=["PTS_home", "PTS_away"])
    base = base.drop_duplicates(subset="GAME_ID", keep="first")
    game_type = base["GAME_ID"].astype("int64").astype(str).str.zfill(10).str[1:3]
    base = base.loc[game_type.eq("02")].copy()
    base["home_team_win"] = (base["PTS_home"] > base["PTS_away"]).astype("int8")
    return base.sort_values(["GAME_DATE_EST", "GAME_ID"]).reset_index(drop=True)


def _team_game_features(base: pd.DataFrame, rest_cap: int) -> pd.DataFrame:
    common = ["GAME_ID", "GAME_DATE_EST", "SEASON", "home_team_win"]
    home = base[common + ["HOME_TEAM_ID", "PTS_home", "PTS_away"]].copy()
    home.columns = common + ["team_id", "points_for", "points_against"]
    home["win"] = home["home_team_win"]

    away = base[common + ["VISITOR_TEAM_ID", "PTS_away", "PTS_home"]].copy()
    away.columns = common + ["team_id", "points_for", "points_against"]
    away["win"] = 1 - away["home_team_win"]

    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.sort_values(["team_id", "SEASON", "GAME_DATE_EST", "GAME_ID"])
    team_games["point_diff"] = team_games["points_for"] - team_games["points_against"]
    grouped = team_games.groupby(["team_id", "SEASON"], sort=False)

    for window in (5, 10):
        team_games[f"rolling_win_pct_{window}"] = grouped["win"].transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).mean()
        )
        team_games[f"rolling_point_diff_{window}"] = grouped["point_diff"].transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).mean()
        )

    previous_date = grouped["GAME_DATE_EST"].shift(1)
    raw_rest = (team_games["GAME_DATE_EST"] - previous_date).dt.days
    team_games["rest_days"] = raw_rest.clip(lower=0, upper=rest_cap)
    team_games["back_to_back"] = raw_rest.eq(1).astype("int8")
    return team_games


def _elo_features(
    base: pd.DataFrame,
    initial_elo: float,
    k_factor: float,
    home_advantage: float,
    offseason_regression: float,
) -> pd.DataFrame:
    ratings: dict[int, float] = {}
    previous_season: int | None = None
    records: list[dict[str, float | int]] = []

    for row in base.itertuples(index=False):
        season = int(row.SEASON)
        if previous_season is not None and season != previous_season:
            ratings = {
                team: initial_elo + offseason_regression * (rating - initial_elo)
                for team, rating in ratings.items()
            }
        previous_season = season

        home = int(row.HOME_TEAM_ID)
        away = int(row.VISITOR_TEAM_ID)
        home_elo = ratings.get(home, initial_elo)
        away_elo = ratings.get(away, initial_elo)
        probability = 1 / (1 + 10 ** ((away_elo - home_elo - home_advantage) / 400))
        outcome = float(row.home_team_win)
        ratings[home] = home_elo + k_factor * (outcome - probability)
        ratings[away] = away_elo + k_factor * ((1 - outcome) - (1 - probability))
        records.append({
            "GAME_ID": int(row.GAME_ID),
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
        })
    return pd.DataFrame(records)


def build_model_dataset(
    games: pd.DataFrame,
    *,
    rest_cap: int = 7,
    initial_elo: float = 1500,
    k_factor: float = 20,
    home_advantage: float = 100,
    offseason_regression: float = 0.75,
) -> pd.DataFrame:
    """Build features known before tipoff; rows lacking 10-game history are omitted."""
    base = prepare_games(games)
    team_games = _team_game_features(base, rest_cap)
    elo = _elo_features(base, initial_elo, k_factor, home_advantage, offseason_regression)

    values = [
        "rolling_win_pct_5", "rolling_win_pct_10",
        "rolling_point_diff_5", "rolling_point_diff_10",
        "rest_days", "back_to_back",
    ]
    home = team_games[["GAME_ID", "team_id", *values]].rename(
        columns={"team_id": "HOME_TEAM_ID", **{value: f"home_{value}" for value in values}}
    )
    away = team_games[["GAME_ID", "team_id", *values]].rename(
        columns={"team_id": "VISITOR_TEAM_ID", **{value: f"away_{value}" for value in values}}
    )
    model = base.merge(home, on=["GAME_ID", "HOME_TEAM_ID"], validate="one_to_one")
    model = model.merge(away, on=["GAME_ID", "VISITOR_TEAM_ID"], validate="one_to_one")
    model = model.merge(elo, on="GAME_ID", validate="one_to_one")

    for metric in ("rolling_win_pct_5", "rolling_win_pct_10", "rolling_point_diff_5", "rolling_point_diff_10", "rest_days"):
        model[f"{metric}_diff"] = model[f"home_{metric}"] - model[f"away_{metric}"]
    return model.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
