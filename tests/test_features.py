from datetime import timedelta

import pandas as pd

from nba_prediction.features import build_model_dataset, prepare_games


def _games(number=14):
    rows = []
    for index in range(number):
        rows.append({
            "GAME_ID": 20000000 + index,
            "GAME_DATE_EST": pd.Timestamp("2020-01-01") + timedelta(days=index),
            "SEASON": 2019,
            "HOME_TEAM_ID": 1 if index % 2 == 0 else 2,
            "VISITOR_TEAM_ID": 2 if index % 2 == 0 else 1,
            "PTS_home": 100 + (index % 3),
            "PTS_away": 99,
        })
    return pd.DataFrame(rows)


def test_prepare_games_filters_non_regular_and_duplicates():
    games = _games(2)
    duplicate = games.iloc[[0]]
    preseason = games.iloc[[1]].assign(GAME_ID=10000001)
    result = prepare_games(pd.concat([games, duplicate, preseason], ignore_index=True))
    assert result["GAME_ID"].tolist() == games["GAME_ID"].tolist()


def test_rolling_features_exclude_current_result():
    games = _games()
    original = build_model_dataset(games)
    changed = games.copy()
    changed.loc[10, ["PTS_home", "PTS_away"]] = [1, 200]
    revised = build_model_dataset(changed)
    row_original = original.loc[original.GAME_ID == changed.loc[10, "GAME_ID"]]
    row_revised = revised.loc[revised.GAME_ID == changed.loc[10, "GAME_ID"]]
    pd.testing.assert_series_equal(
        row_original.filter(like="rolling").iloc[0],
        row_revised.filter(like="rolling").iloc[0],
    )


def test_rolling_history_resets_each_season():
    games = _games()
    next_season = _games().assign(
        GAME_ID=lambda frame: frame.GAME_ID + 100000,
        GAME_DATE_EST=lambda frame: frame.GAME_DATE_EST + pd.DateOffset(years=1),
        SEASON=2020,
    )
    result = build_model_dataset(pd.concat([games, next_season], ignore_index=True))
    assert result.groupby("SEASON").size().to_dict() == {2019: 4, 2020: 4}
