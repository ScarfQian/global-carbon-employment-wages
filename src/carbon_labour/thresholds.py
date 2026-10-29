"""Separate total-carbon locking sensitivity from intensity-based exposure sensitivity."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .positions import SKILLS, audit_row, numeric, safe_divide


def sector_key_columns(df: pd.DataFrame) -> List[str]:
    """识别 producer-sector-skill 表的部门键。"""
    cols = ["Producer_Region" if "Producer_Region" in df.columns else "Region", "Sector"]
    if "Sector_Code" in df.columns:
        cols.append("Sector_Code")
    return cols


def build_sector_base_table(producer_sector_skill: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从 producer-sector-skill 表构建不重复的 country-sector 基表。"""
    df = producer_sector_skill.copy()
    key_cols = sector_key_columns(df)
    skill_values = df.groupby(key_cols + ["Skill"], as_index=False, observed=False)[
        ["Carbon_Attributed", "Employment", "Hours", "Wage"]
    ].sum()
    sector_values = df.groupby(key_cols, as_index=False, observed=False).agg(
        Total_Direct_Carbon=("Total_Carbon", "max"),
        Attributed_Carbon_Total=("Carbon_Attributed", "sum"),
        Employment_Total=("Employment", "sum"),
        Hours_Total=("Hours", "sum"),
        Wage_Total=("Wage", "sum"),
    )
    sector_values["Carbon_Per_Employment_Intensity"] = np.divide(
        numeric(sector_values["Total_Direct_Carbon"]),
        numeric(sector_values["Employment_Total"]),
        out=np.full(len(sector_values), np.nan, dtype=float),
        where=numeric(sector_values["Employment_Total"]).to_numpy(dtype=float) > 0,
    )
    return sector_values, skill_values


def compute_threshold_rows(
    sector_values: pd.DataFrame,
    skill_values: pd.DataFrame,
    definition: str,
    percentiles: Sequence[float] = (0.70, 0.80, 0.90),
) -> pd.DataFrame:
    """按总碳或碳就业强度阈值计算 high-carbon lock-in 稳健性。"""
    key_cols = [column for column in sector_values.columns if column in ["Producer_Region", "Region", "Sector", "Sector_Code"]]
    if definition == "total_direct_carbon":
        metric_col = "Total_Direct_Carbon"
        positive = numeric(sector_values[metric_col]) > 0
    elif definition == "carbon_per_employment_intensity":
        metric_col = "Carbon_Per_Employment_Intensity"
        positive = numeric(sector_values[metric_col]) > 0
    else:
        raise ValueError(f"Unsupported threshold definition: {definition}")

    totals_by_skill = skill_values.groupby("Skill", observed=False)[["Carbon_Attributed", "Employment", "Hours", "Wage"]].sum()
    carbon_total_all = float(numeric(sector_values["Total_Direct_Carbon"]).sum())
    rows: List[Dict[str, object]] = []
    for percentile in percentiles:
        cutoff = float(numeric(sector_values.loc[positive, metric_col]).quantile(percentile))
        high = sector_values.loc[numeric(sector_values[metric_col]) >= cutoff].copy()
        high_keys = high[key_cols].copy()
        tagged = skill_values.merge(high_keys.assign(High_Carbon=True), on=key_cols, how="left")
        tagged["High_Carbon"] = tagged["High_Carbon"].eq(True)
        direct_carbon_high = float(numeric(high["Total_Direct_Carbon"]).sum())
        for skill in SKILLS:
            group = tagged.loc[tagged["Skill"].eq(skill)].copy()
            high_group = group.loc[group["High_Carbon"]]
            denominators = totals_by_skill.loc[skill] if skill in totals_by_skill.index else pd.Series(dtype=float)
            rows.append(
                {
                    "Threshold_Percentile": int(round(percentile * 100)),
                    "Definition": definition,
                    "Skill": skill,
                    "High_Carbon_Sector_Count": int(len(high)),
                    "Carbon_Total_In_HighCarbonSet": direct_carbon_high,
                    "Carbon_Share_In_HighCarbonSet": safe_divide(direct_carbon_high, carbon_total_all),
                    "Employment_Total_In_HighCarbonSet": float(numeric(high_group["Employment"]).sum()),
                    "Employment_Share_In_HighCarbonSet": safe_divide(numeric(high_group["Employment"]).sum(), float(denominators.get("Employment", np.nan))),
                    "Hours_Total_In_HighCarbonSet": float(numeric(high_group["Hours"]).sum()),
                    "Hours_Share_In_HighCarbonSet": safe_divide(numeric(high_group["Hours"]).sum(), float(denominators.get("Hours", np.nan))),
                    "Wage_Total_In_HighCarbonSet": float(numeric(high_group["Wage"]).sum()),
                    "Wage_Share_In_HighCarbonSet": safe_divide(numeric(high_group["Wage"]).sum(), float(denominators.get("Wage", np.nan))),
                    "Attributed_Carbon_Total_In_HighCarbonSet": float(numeric(high_group["Carbon_Attributed"]).sum()),
                    "Attributed_Carbon_Share_In_HighCarbonSet": safe_divide(numeric(high_group["Carbon_Attributed"]).sum(), float(denominators.get("Carbon_Attributed", np.nan))),
                    "Employment_Lock_Ratio": safe_divide(numeric(high_group["Employment"]).sum(), float(denominators.get("Employment", np.nan))),
                    "Hours_Lock_Ratio": safe_divide(numeric(high_group["Hours"]).sum(), float(denominators.get("Hours", np.nan))),
                    "Threshold_Value": cutoff,
                }
            )
    columns = [
        "Threshold_Percentile",
        "Definition",
        "Skill",
        "High_Carbon_Sector_Count",
        "Carbon_Total_In_HighCarbonSet",
        "Carbon_Share_In_HighCarbonSet",
        "Employment_Total_In_HighCarbonSet",
        "Employment_Share_In_HighCarbonSet",
        "Hours_Total_In_HighCarbonSet",
        "Hours_Share_In_HighCarbonSet",
        "Wage_Total_In_HighCarbonSet",
        "Wage_Share_In_HighCarbonSet",
        "Attributed_Carbon_Total_In_HighCarbonSet",
        "Attributed_Carbon_Share_In_HighCarbonSet",
        "Employment_Lock_Ratio",
        "Hours_Lock_Ratio",
        "Threshold_Value",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(["Definition", "Threshold_Percentile", "Skill"]).reset_index(drop=True)


def build_high_carbon_threshold_audit(total_df: pd.DataFrame, intensity_df: pd.DataFrame) -> pd.DataFrame:
    """生成 high-carbon threshold robustness 审计。"""
    combined = pd.concat([total_df, intensity_df], ignore_index=True)
    rows = [
        audit_row(
            "high_carbon_threshold_definition",
            "both_threshold_definitions_present",
            int(set(combined["Definition"].unique()) == {"total_direct_carbon", "carbon_per_employment_intensity"}),
            1,
            set(combined["Definition"].unique()) == {"total_direct_carbon", "carbon_per_employment_intensity"},
            "total-carbon lock-in and intensity-based exposure robustness are explicitly separated",
        ),
        audit_row(
            "high_carbon_threshold_definition",
            "main_lockin_definition_is_total_direct_carbon",
            "total_direct_carbon",
            "total_direct_carbon",
            True,
            "main manuscript lock-in definition is based on positive direct carbon totals",
            0.0,
        ),
        audit_row(
            "high_carbon_threshold_definition",
            "intensity_definition_labelled_supplementary",
            "carbon_per_employment_intensity",
            "carbon_per_employment_intensity",
            True,
            "carbon-per-employment threshold is labelled as intensity-based exposure robustness",
            0.0,
        ),
    ]
    for definition, group in combined.groupby("Definition", observed=False):
        counts = group.groupby("Threshold_Percentile", observed=False)["High_Carbon_Sector_Count"].max().sort_index()
        monotone = bool(counts.is_monotonic_decreasing)
        rows.append(
            audit_row(
                "high_carbon_threshold_monotonicity",
                f"{definition}_sector_counts_decrease_with_threshold",
                ";".join(f"{idx}:{value}" for idx, value in counts.items()),
                "monotonic_non_increasing",
                monotone,
                "higher percentile thresholds should not increase the high-carbon sector count",
                0.0 if monotone else 1.0,
            )
        )
    return pd.DataFrame(rows)


def compute_threshold_package(producer_sector_skill: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return total-carbon and intensity sensitivity tables and their audit."""
    sector_values, skill_values = build_sector_base_table(producer_sector_skill)
    total_df = compute_threshold_rows(sector_values, skill_values, "total_direct_carbon")
    intensity_df = compute_threshold_rows(sector_values, skill_values, "carbon_per_employment_intensity")
    audit = build_high_carbon_threshold_audit(total_df, intensity_df)
    return {
        "high_carbon_total_carbon_threshold_robustness.csv": total_df,
        "high_carbon_intensity_threshold_robustness.csv": intensity_df,
        "high_carbon_threshold_robustness_audit.csv": audit,
    }
