"""Synthetic tests for bilateral transfers and their network interpretation."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd

from carbon_labour import corridors as MODULE


def test_corridor_merge_aggregates_duplicates_and_keeps_weighted_identities() -> None:
    """重复 corridor-skill 行应先聚合，再计算 share-normalized ratio。"""
    carbon = pd.DataFrame(
        {
            "From_Region": ["A", "A", "B", "A"],
            "To_Region": ["B", "B", "A", "A"],
            "Skill": ["Low", "Low", "High", "Low"],
            "Carbon_Transfer": [10.0, 5.0, 5.0, 99.0],
        }
    )
    employment = pd.DataFrame(
        {
            "From_Region": ["A", "B"],
            "To_Region": ["B", "A"],
            "Skill": ["Low", "High"],
            "Employment_Transfer": [5.0, 5.0],
        }
    )
    wage = pd.DataFrame(
        {
            "From_Region": ["A", "B"],
            "To_Region": ["B", "A"],
            "Skill": ["Low", "High"],
            "Wage_Transfer": [8.0, 2.0],
        }
    )
    hours = pd.DataFrame(
        {
            "From_Region": ["A", "B"],
            "To_Region": ["B", "A"],
            "Skill": ["Low", "High"],
            "Hours_Transfer": [4.0, 6.0],
        }
    )

    merged, audit = MODULE.compute_corridor_skill_misalignment(
        {
            "carbon": carbon,
            "employment": employment,
            "wage": wage,
            "hours": hours,
        }
    )

    assert len(merged) == 2
    assert float(merged.loc[merged["From_Region"].eq("A"), "Carbon_Transfer"].iloc[0]) == 15.0
    assert not merged.duplicated(["From_Region", "To_Region", "Skill"]).any()
    assert (audit["Check"].eq("no duplicate corridor-skill rows after aggregation")).any()

    identity_specs = {
        "R_CE": "Employment_Share",
        "R_CW": "Wage_Share",
        "R_CH": "Hours_Share",
        "R_WH": "Hours_Share",
    }
    for ratio_col, weight_col in identity_specs.items():
        valid = merged[ratio_col].notna()
        observed = float((merged.loc[valid, weight_col] * merged.loc[valid, ratio_col]).sum())
        assert np.isclose(observed, 1.0, rtol=1e-10, atol=1e-10), ratio_col



def test_corridor_ratios_have_flags_and_no_infinite_values() -> None:
    """corridor 分类允许多标签，ratio 和 log ratio 不能出现 infinity。"""
    frames = {
        "carbon": pd.DataFrame(
            {
                "From_Region": ["A", "B"],
                "To_Region": ["B", "A"],
                "Skill": ["Low", "High"],
                "Carbon_Transfer": [90.0, 10.0],
            }
        ),
        "employment": pd.DataFrame(
            {
                "From_Region": ["A", "B"],
                "To_Region": ["B", "A"],
                "Skill": ["Low", "High"],
                "Employment_Transfer": [30.0, 70.0],
            }
        ),
        "wage": pd.DataFrame(
            {
                "From_Region": ["A", "B"],
                "To_Region": ["B", "A"],
                "Skill": ["Low", "High"],
                "Wage_Transfer": [20.0, 80.0],
            }
        ),
        "hours": pd.DataFrame(
            {
                "From_Region": ["A", "B"],
                "To_Region": ["B", "A"],
                "Skill": ["Low", "High"],
                "Hours_Transfer": [60.0, 40.0],
            }
        ),
    }

    merged, _ = MODULE.compute_corridor_skill_misalignment(frames)
    ratio_cols = ["R_CE", "R_CW", "R_CH", "R_WH"]
    log_cols = [f"log_{col}" for col in ratio_cols]

    a_row = merged.loc[merged["From_Region"].eq("A")].iloc[0]
    assert "Carbon-heavy corridor" in a_row["Corridor_Flags"]
    assert "Burden-over-return corridor" in a_row["Corridor_Flags"]
    assert not np.isinf(merged[ratio_cols + log_cols].to_numpy(dtype=float)).any()



def test_corridor_scale_weighted_scores_and_summaries_are_available() -> None:
    """corridor 结果应同时支持纯 ratio 与按规模加权的排序。"""
    frames = {
        "carbon": pd.DataFrame(
            {
                "From_Region": ["A", "C", "B"],
                "To_Region": ["B", "B", "A"],
                "Skill": ["Low", "Low", "High"],
                "Carbon_Transfer": [90.0, 9.0, 1.0],
            }
        ),
        "employment": pd.DataFrame(
            {
                "From_Region": ["A", "C", "B"],
                "To_Region": ["B", "B", "A"],
                "Skill": ["Low", "Low", "High"],
                "Employment_Transfer": [30.0, 30.0, 40.0],
            }
        ),
        "wage": pd.DataFrame(
            {
                "From_Region": ["A", "C", "B"],
                "To_Region": ["B", "B", "A"],
                "Skill": ["Low", "Low", "High"],
                "Wage_Transfer": [20.0, 1.0, 79.0],
            }
        ),
        "hours": pd.DataFrame(
            {
                "From_Region": ["A", "C", "B"],
                "To_Region": ["B", "B", "A"],
                "Skill": ["Low", "Low", "High"],
                "Hours_Transfer": [60.0, 10.0, 30.0],
            }
        ),
    }

    merged, audit = MODULE.compute_corridor_skill_misalignment(frames)
    summaries = MODULE.build_corridor_summaries(merged)

    for column in [
        "Weighted_Log_R_CW",
        "Weighted_Log_R_CE",
        "Weighted_Burden_Return_Score",
        "Weighted_Carbon_Employment_Score",
        "Corridor_Type",
        "Combined_Corridor_Type",
    ]:
        assert column in merged.columns
    assert "summary_by_skill" in summaries
    assert "summary_by_type" in summaries
    assert not summaries["summary_by_type"].empty
    assert audit["Passed"].all()
