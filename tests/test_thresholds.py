"""Keep total producer carbon separate from its repeated skill rows."""

import numpy as np
import pandas as pd

from carbon_labour.thresholds import build_sector_base_table, compute_threshold_package


def test_skill_rows_do_not_triple_total_carbon():
    rows = []
    for sector, carbon in zip(["a", "b", "c", "d"], [10.0, 20.0, 30.0, 40.0]):
        for skill in ["Low", "Medium", "High"]:
            rows.append({
                "Region": "AA", "Sector": sector, "Skill": skill,
                "Total_Carbon": carbon, "Carbon_Attributed": carbon / 3,
                "Employment": 1.0, "Hours": 2.0, "Wage": 3.0,
            })
    data = pd.DataFrame(rows)
    sectors, _ = build_sector_base_table(data)
    assert sectors["Total_Direct_Carbon"].sum() == 100.0
    package = compute_threshold_package(data)
    total = package["high_carbon_total_carbon_threshold_robustness.csv"]
    selected = total.loc[total["Threshold_Percentile"].eq(80)]
    assert np.allclose(selected["Threshold_Value"], 34.0)
    assert np.allclose(selected["Carbon_Total_In_HighCarbonSet"], 40.0)
    assert np.allclose(selected["Carbon_Share_In_HighCarbonSet"], 0.4)
