import pandas as pd

from nba_prediction.modeling import evaluate_predictions, season_splits


def test_season_splits_never_mix_future_into_training():
    data = pd.DataFrame({"SEASON": [2018, 2018, 2019, 2019, 2020, 2020]})
    for train, validation in season_splits(data, minimum_train_seasons=1):
        assert data.loc[train, "SEASON"].max() < data.loc[validation, "SEASON"].min()


def test_probability_metrics_are_reported():
    metrics = evaluate_predictions(pd.Series([0, 1]), pd.Series([0.2, 0.8]))
    assert set(metrics) == {"accuracy", "roc_auc", "log_loss", "brier_score"}
