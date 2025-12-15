import pandas as pd
import pytest

from src.features.preprocessing import temporal_train_test_split

def test_temporal_train_test_split_sizes_and_order():
    df = pd.DataFrame({
        "feat": [1, 2, 3, 4, 5],
        "Color": [10, 11, 12, 13, 14],
    })

    feature_cols = ["feat"]
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df=df,
        feature_cols=feature_cols,
        target_col="Color",
        test_size=0.4
    )

    # tamanhos
    assert len(X_train) == 3
    assert len(X_test) == 2

    # ordem temporal preservada (último do treino vem antes do primeiro do teste)
    assert y_train.iloc[-1] == 12
    assert y_test.iloc[0] == 13


def test_temporal_train_test_split_invalid_test_size():
    df = pd.DataFrame({"feat": [1, 2], "Color": [10, 11]})

    with pytest.raises(ValueError):
        temporal_train_test_split(df, ["feat"], "Color", test_size=1.0)

    with pytest.raises(ValueError):
        temporal_train_test_split(df, ["feat"], "Color", test_size=0.0)
