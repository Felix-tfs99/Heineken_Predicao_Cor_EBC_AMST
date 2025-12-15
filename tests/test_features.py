import pandas as pd
import numpy as np

from src.features.preprocessing import add_domain_features

def test_add_domain_features_creates_expected_columns():
    df = pd.DataFrame({
        "Roast amount (kg)": [10, 0],
        "1st (base) malt amount (kg)": [20, 10],
        "2nd (base) malt amount (kg)": [30, 10],
        "MT – Temperature (°C)": [60, 65],
        "MT – Time (s)": [100, 200],
        "WK – Temperature (°C)": [90, 95],
        "WK – Time (s)": [300, 400],
    })

    out = add_domain_features(df)

    for col in ["total_malt", "roast_pct", "mt_energy", "wk_energy"]:
        assert col in out.columns

    # checagens básicas
    assert np.all(out["total_malt"] >= 0)
    assert np.all(out["mt_energy"] >= 0)
    assert np.all(out["wk_energy"] >= 0)

    # roast_pct deve ser finito (sem inf/nan por divisão por zero)
    assert np.isfinite(out["roast_pct"]).all()
