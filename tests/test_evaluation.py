import numpy as np

from src.models.evaluation import eval_regression

def test_eval_regression_returns_expected_keys_and_types():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    out = eval_regression(y_true, y_pred, label="test", verbose=False)

    assert set(out.keys()) == {"mae", "rmse", "r2"}
    assert isinstance(out["mae"], float)
    assert isinstance(out["rmse"], float)
    assert isinstance(out["r2"], float)

    assert out["mae"] >= 0
    assert out["rmse"] >= 0
